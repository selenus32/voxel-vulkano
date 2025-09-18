use vulkano::{
    shader, 
    VulkanLibrary
};
use vulkano::instance::{
    Instance, 
    InstanceCreateFlags, 
    InstanceCreateInfo
};
use vulkano::device::{
    Device, 
    DeviceCreateInfo, 
    QueueCreateInfo, 
    QueueFlags, 
    Queue, 
    DeviceExtensions,
    physical::{PhysicalDevice, PhysicalDeviceType}
};
use vulkano::swapchain::{
    Surface, 
    Swapchain, 
    SwapchainCreateInfo, 
    SwapchainPresentInfo
};

use vulkano::memory::allocator::{
    StandardMemoryAllocator, 
    AllocationCreateInfo, 
    MemoryTypeFilter
};
use vulkano::buffer::{
    Buffer, 
    BufferCreateInfo, 
    BufferUsage, 
    BufferContents
};
use vulkano::format::Format;
use vulkano::image::{
    Image, 
    ImageCreateInfo, 
    ImageType, 
    ImageUsage,
    view::ImageView
};
use vulkano::command_buffer::{
    //CopyImageToBufferInfo,
    AutoCommandBufferBuilder, 
    CommandBufferUsage,
    allocator::StandardCommandBufferAllocator,
    allocator::CommandBufferAllocator,
    BlitImageInfo
};
use vulkano::pipeline::{
    compute::ComputePipelineCreateInfo,
    layout::PipelineDescriptorSetLayoutCreateInfo,
    ComputePipeline,
    Pipeline,
    PipelineLayout, 
    PipelineShaderStageCreateInfo,
    PipelineBindPoint
};
use vulkano::descriptor_set::{
    DescriptorSet, 
    WriteDescriptorSet,
    allocator::StandardDescriptorSetAllocator
};
use vulkano::sync::GpuFuture;
use winit::{
    event::{WindowEvent, DeviceEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
    application::ApplicationHandler,
};
use std::{
    fs,
    time::SystemTime,
    sync::Arc
};
use shaderc;
use nalgebra as na;
use na::{
    Isometry3, 
    Matrix4, 
    Point3, 
    UnitQuaternion, 
    Vector3,
    Perspective3
};

struct Camera {
    isom: Isometry3<f32>,
    fov: f32, // deg
    near: f32,
    far: f32,
    yaw: f32, // deg
    pitch: f32, // deg
    mouse_sens: f32,
    first_mouse: bool
}

impl Default for Camera {
    fn default() -> Self {
        Self { 
            isom: Isometry3::identity(), 
            fov: 90.0, 
            near: 0.1, 
            far: 100.0, 
            yaw: -90.0, 
            pitch: 0.0, 
            mouse_sens: 0.1,
            first_mouse: true
        }
    }
}

impl Camera {
    fn get_view(&self) -> Matrix4<f32> {
        self.isom.inverse().to_homogeneous()
    }
    fn get_proj(&self, width: u32, height: u32) -> Matrix4<f32> {
        Perspective3::new(
            width as f32 / height as f32,
            self.fov.to_radians(),
            self.near,
            self.far
        ).to_homogeneous()
    }
    fn update_isometry(&mut self) {
        let yaw = self.yaw.to_radians();
        let pitch = self.pitch.to_radians();
        let rotation = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), yaw)
            * UnitQuaternion::from_axis_angle(&Vector3::x_axis(), pitch);
        self.isom.rotation = rotation
    }
    fn update_orientation(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * self.mouse_sens;
        self.pitch += dy * self.mouse_sens;
        self.pitch = self.pitch.clamp(-89.9,89.9);
        self.update_isometry();
    }
}

struct App {
    window: Option<Arc<Window>>,
    surface: Option<Arc<Surface>>,
    physical_device: Option<Arc<PhysicalDevice>>,
    device: Option<Arc<Device>>,
    queue: Option<Arc<Queue>>,
    memory_allocator: Option<Arc<StandardMemoryAllocator>>,
    swapchain: Option<Arc<Swapchain>>,
    swapchain_images: Vec<Arc<Image>>,
    compute_pipeline: Option<Arc<ComputePipeline>>,
    computed_image: Option<Arc<Image>>,
    computed_image_view: Option<Arc<ImageView>>,
    compute_shader_path: String,
    compute_shader_last_modified: SystemTime,
    width: u32,
    height: u32,
    camera: Camera
}

impl Default for App {
    fn default() -> Self {
        App {
            window: None,
            surface: None,
            physical_device: None,
            device: None,
            queue: None,
            memory_allocator: None,
            swapchain: None,
            swapchain_images: Vec::new(),
            compute_pipeline: None,
            computed_image: None,
            computed_image_view: None,
            compute_shader_path: "src/shaders/compute.comp".to_string(),
            compute_shader_last_modified: SystemTime::UNIX_EPOCH,
            width: 1024,
            height: 768,
            camera: Camera::default()
        }
    }
}

impl App {
    fn select_physical_device(
        &mut self,
        instance: &Arc<Instance>,
        device_extensions: &DeviceExtensions,
    ) -> (Arc<PhysicalDevice>, u32) {
        instance
            .enumerate_physical_devices()
            .expect("could not enumerate devices")
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.contains(QueueFlags::GRAPHICS)
                            && p.surface_support(i as u32, self.surface.as_ref().unwrap()).unwrap_or(false)
                    })
                    .map(|q| (p, q as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                _ => 4,
            })
            .expect("no device available")
    }

    fn update_swapchain(&mut self) {
        let surface_caps = self.physical_device.as_ref().unwrap()
            .surface_capabilities(self.surface.as_ref().unwrap(), Default::default())
            .expect("failed to get surface capabilities");
        
        let dimensions = self.window.as_ref().unwrap().inner_size();
        let composite_alpha = surface_caps.supported_composite_alpha.into_iter().next().unwrap();
        let image_format = self.physical_device.as_ref().unwrap()
            .surface_formats(self.surface.as_ref().unwrap(), Default::default())
            .unwrap()[0]
            .0;

        let swapchain_create_info = SwapchainCreateInfo {
            min_image_count: surface_caps.min_image_count + 1,
            image_format,
            image_extent: dimensions.into(),
            image_usage: ImageUsage::TRANSFER_DST,
            composite_alpha,
            ..Default::default()
        };

        let (swapchain, images) = if let Some(existing) = &self.swapchain {
                Swapchain::recreate(existing, swapchain_create_info).unwrap()
            } else {
                Swapchain::new(
                    self.device.as_ref().unwrap().clone(),
                    self.surface.as_ref().unwrap().clone(),
                    swapchain_create_info,
                )
            .unwrap()
        };

        self.swapchain = Some(swapchain);
        self.swapchain_images = images;
    }

    fn update_compute_pipeline_shaderc(&mut self) { // needs to be rewritten to handle errors properly without expects
        let source = fs::read_to_string(self.compute_shader_path.clone()).expect("failed to read shader file");
        let entry_point_name = "main";

        let shaderc_compiler = shaderc::Compiler::new().unwrap();
        
        let spirv_bin = shaderc_compiler.compile_into_spirv(
            &source,
            shaderc::ShaderKind::Compute,
            &self.compute_shader_path,
            entry_point_name,
            None,
        ).expect("failed to compile glsl shader into spirv");
        
        assert_eq!(Some(&0x07230203), spirv_bin.as_binary().first());

        let shader_module = unsafe {shader::ShaderModule::new(
            self.device.as_ref().unwrap().clone(), 
            shader::ShaderModuleCreateInfo::new(spirv_bin.as_binary()),
        ).expect("failed to create shader module")};

        let shader_entry_point = shader_module.entry_point(entry_point_name).unwrap(); // need to look into this further. probably doing something wrong.
        let stage = PipelineShaderStageCreateInfo::new(shader_entry_point);
        let layout = PipelineLayout::new(
            self.device.as_ref().unwrap().clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(self.device.as_ref().unwrap().clone())
                .unwrap(),
        )
        .unwrap();
        let compute_pipeline = ComputePipeline::new(
            self.device.as_ref().unwrap().clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .expect("failed to create compute pipeline");

        self.compute_pipeline = Some(compute_pipeline);
    }

    fn update_compute_pipeline_vulkano(&mut self) {
        mod cs {
            vulkano_shaders::shader!{
                ty: "compute",
                path: "src/shaders/compute.comp",
                //vulkan_version: "1.3",
            }
        }
        let shader = cs::load(self.device.as_ref().unwrap().clone()).expect("failed to create shader module");
        let cs = shader.entry_point("main").unwrap();
        let stage = PipelineShaderStageCreateInfo::new(cs);
        let layout = PipelineLayout::new(
            self.device.as_ref().unwrap().clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(self.device.as_ref().unwrap().clone())
                .unwrap(),
        )
        .unwrap();
        let compute_pipeline = ComputePipeline::new(
            self.device.as_ref().unwrap().clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .expect("failed to create compute pipeline");

        self.compute_pipeline = Some(compute_pipeline);
    }

    fn update_frame(&mut self) {
        let computed_image = self.computed_image.get_or_insert_with(|| {
            Image::new(
                self.memory_allocator.as_ref().unwrap().clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: Format::R8G8B8A8_UNORM,
                    extent: [self.width, self.height, 1],
                    usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
            ).unwrap()
        });

        let computed_image_view = self.computed_image_view.get_or_insert_with(|| {
            ImageView::new_default(computed_image.clone()).unwrap()
        });
        
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            self.device.as_ref().unwrap().clone(),
            Default::default(),
        ));
        let pipeline_layout = self.compute_pipeline.as_ref().unwrap().layout();
        let descriptor_set_layouts = pipeline_layout.set_layouts();
        let descriptor_set_layout_index = 0;
        let descriptor_set_layout = descriptor_set_layouts
            .get(descriptor_set_layout_index)
            .unwrap();
        let descriptor_set = DescriptorSet::new(
            descriptor_set_allocator,
            descriptor_set_layout.clone(),
            [WriteDescriptorSet::image_view(0, computed_image_view.clone())], // binding 0
            [],
        ).unwrap();

        let command_buffer_allocator: Arc<dyn CommandBufferAllocator> =
            Arc::new(StandardCommandBufferAllocator::new(self.device.as_ref().unwrap().clone(), Default::default()));
        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator,
            self.queue.as_ref().unwrap().queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        ).unwrap();
        let (swapchain_image_index, _suboptimal, acquire_future) = 
            vulkano::swapchain::acquire_next_image(self.swapchain.as_ref().unwrap().clone(), None).unwrap();
        let target_swapchain_image = self.swapchain_images[swapchain_image_index as usize].clone();
        let compute_blit_info = BlitImageInfo::images(computed_image.clone(), target_swapchain_image);
        unsafe {
            command_buffer_builder
            .bind_pipeline_compute(self.compute_pipeline.as_ref().unwrap().clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.compute_pipeline.as_ref().unwrap().layout().clone(),
                descriptor_set_layout_index as u32,
                descriptor_set,
            )
            .unwrap()
            .dispatch([(self.width + 7) / 8, (self.height + 7) / 8, 1])
            .unwrap()
            .blit_image(compute_blit_info)
            .unwrap();
        }
        let command_buffer = command_buffer_builder.build().unwrap();
        let future = acquire_future
            .then_execute(self.queue.as_ref().unwrap().clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(self.queue.as_ref().unwrap().clone(),SwapchainPresentInfo::swapchain_image_index(self.swapchain.as_ref().unwrap().clone(), swapchain_image_index))
            .then_signal_fence_and_flush()
            .unwrap();
        future.wait(None).unwrap();
    }

}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = Window::default_attributes()
            .with_title("Voxel Vulkano")
            .with_inner_size(winit::dpi::PhysicalSize::new(self.width,self.height));
        self.window = Some(Arc::new(event_loop
            .create_window(window_attributes)
            .unwrap()));
        let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");

        let required_extensions = Surface::required_extensions(&event_loop);
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions.unwrap(),
                ..Default::default()
            },
        )
        .expect("failed to create instance");

        let surface = Surface::from_window(instance.clone(), self.window.as_ref().unwrap().clone()).unwrap();
        self.surface = Some(surface);

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) = self.select_physical_device(
            &instance,
            &device_extensions,
        );
        self.physical_device = Some(physical_device);

        for family in self.physical_device.as_ref().unwrap().queue_family_properties() {
            println!("found a queue family with {:?} queue(s)", family.queue_count);
        }

        let (device, mut queues) = Device::new(
            self.physical_device.as_ref().unwrap().clone(),
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: device_extensions,
                ..Default::default()
            },
        ).expect("failed to create device");

        let queue = queues.next().unwrap();

        self.device = Some(device);
        self.queue = Some(queue);
        self.memory_allocator = Some(Arc::new(StandardMemoryAllocator::new_default(self.device.as_ref().unwrap().clone())));

        self.update_swapchain();

        self.compute_shader_last_modified = std::fs::metadata(&self.compute_shader_path)
            .and_then(|meta| meta.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);

        //self.update_compute_pipeline_vulkano();
        self.update_compute_pipeline_shaderc();
        self.update_frame();

        println!("Device: {}", self.device.as_ref().unwrap().physical_device().properties().device_name);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                println!("close requested. stopping.");
                event_loop.exit();
            },
            WindowEvent::Resized(physical_size) => {
                if physical_size.width > 0 && physical_size.height > 0 {
                    self.width = physical_size.width;
                    self.height = physical_size.height;

                    self.computed_image = None;
                    self.computed_image_view = None;

                    self.update_swapchain();
                    println!("window resized to {}x{}",self.width,self.height);
                }
            },
            WindowEvent::RedrawRequested => {
                let compute_shader_modified = fs::metadata(&self.compute_shader_path).unwrap().modified().unwrap();
                if compute_shader_modified > self.compute_shader_last_modified {
                    //self.update_compute_pipeline_vulkano();
                    self.update_compute_pipeline_shaderc();
                    self.compute_shader_last_modified = compute_shader_modified;
                }

                let view_mat = self.camera.get_view();
                let proj_mat = self.camera.get_proj(self.width, self.height);

                self.update_frame();
                self.window.as_ref().unwrap().request_redraw();
            },
            WindowEvent::KeyboardInput { device_id, event, is_synthetic } => {

            },
            _ => (),
        }
    }

    fn device_event(&mut self, event_loop: &ActiveEventLoop, _id: winit::event::DeviceId, event: DeviceEvent) {
        match event {
            DeviceEvent::MouseMotion{delta: (dx, dy)} => {
                self.camera.update_orientation(dx as f32, dy as f32);
            },
            _ => (),
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::default();

    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.set_control_flow(ControlFlow::Wait);

    let _ = event_loop.run_app(&mut app);
}