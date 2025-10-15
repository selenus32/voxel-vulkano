use vulkano::command_buffer::PrimaryCommandBufferAbstract;
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
    Subbuffer, 
    BufferCreateInfo, 
    BufferUsage, 
    BufferContents,
    allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo}
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
    event::{WindowEvent, DeviceEvent, ElementState, MouseButton},
    event_loop::{ActiveEventLoop},
    window::{Window, WindowId},
    application::ApplicationHandler,
    keyboard::{PhysicalKey, KeyCode},
};
use std::{
    fs,
    time::SystemTime,
    sync::Arc
};
use std::collections::HashSet;
use shaderc;
use nalgebra as na;
use na::{
    Vector3,
};

use crate::player::Player;

use crate::voxels::brickmap::*;
use crate::voxels::tree64;

#[derive(Hash, Eq, PartialEq, Debug)]
enum InputButton {
    Key(KeyCode),
    Mouse(MouseButton),
}

pub struct App {
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
    descriptor_set_allocator: Option<Arc<StandardDescriptorSetAllocator>>,
    uniform_allocator: Option<SubbufferAllocator>,
    compute_shader_path: String,
    compute_shader_last_modified: SystemTime,
    width: u32,
    height: u32,
    player: Player,
    device_local_buffer: Option<Subbuffer<[u32]>>,
    brickmap_data_size: u32,
    pressed: HashSet<InputButton>,
    instant: std::time::Instant,
    time: f32,
    last_time: f32,
    dt: f32,
    fps_accum: u32,
    fps_time_accum: f32
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
            descriptor_set_allocator: None,
            uniform_allocator: None,
            compute_shader_path: "src/shaders/crude_vox_test.comp.glsl".to_string(),
            compute_shader_last_modified: SystemTime::UNIX_EPOCH,
            width: 1024,
            height: 1024,
            player: Player::default(),
            device_local_buffer: None,
            brickmap_data_size: 0,
            pressed: HashSet::new(),
            instant: std::time::Instant::now(),
            time: 0.0,
            last_time: 0.0,
            dt: 0.0,
            fps_accum: 0,
            fps_time_accum: 0.0
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

    fn update_compute_pipeline_shaderc(&mut self) {
        let source = fs::read_to_string(self.compute_shader_path.clone()).expect("failed to read shader file");
        let entry_point_name = "main";

        let shaderc_compiler = shaderc::Compiler::new().unwrap();
        let mut options = shaderc::CompileOptions::new().unwrap();
        options.set_target_env(shaderc::TargetEnv::Vulkan, shaderc::EnvVersion::Vulkan1_3 as u32);
        options.set_include_callback(|include, include_type, source, _| {
            let path = match include_type {
                shaderc::IncludeType::Relative => std::path::Path::new(source).parent().unwrap().join(include),
                shaderc::IncludeType::Standard => std::path::Path::new("src/shaders/include").join(include),
            };
            let content = std::fs::read_to_string(&path).map_err(|e| format!("failed to include {}, {}", include, e))?;
            Ok(shaderc::ResolvedInclude {
                resolved_name: path.to_string_lossy().into_owned(),
                content: content,
            })
        });
        options.set_forced_version_profile(460 as u32, shaderc::GlslProfile::Core);

        let spirv_bin = shaderc_compiler.compile_into_spirv(
            &source,
            shaderc::ShaderKind::Compute,
            &self.compute_shader_path,
            entry_point_name,
            Some(&options),
        ).unwrap_or_else(|err| { // so it doesn't crash while live-debugging shader
            eprintln!("{}: {}\n falling back to default shader", self.compute_shader_path, err);
            let default_path = "src/shaders/default.comp.glsl";
            let default_src = std::fs::read_to_string(default_path)
                .expect("failed to read default shader");
            shaderc_compiler.compile_into_spirv(
                &default_src,
                shaderc::ShaderKind::Compute,
                default_path,
                entry_point_name,
                Some(&options),
            ).expect("failed to compile glsl shader into spirv")
        });
        
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

    /*fn update_compute_pipeline_vulkano(&mut self) {
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
    }*/

    fn upload_brickmap(&mut self) {
        //let gpu_brickmap = generate_brickmap();
        let gpu_brickmap = read_brickmap("assets/nuke.brk").unwrap();
        self.brickmap_data_size = (gpu_brickmap.bytes.len() / std::mem::size_of::<u32>()) as u32;

        let temporary_accessible_buffer = Buffer::from_iter(
            self.memory_allocator.as_ref().unwrap().clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            gpu_brickmap.bytes,
        )
        .unwrap();

        let device_local_buffer = Buffer::new_slice::<u32>(
            self.memory_allocator.as_ref().unwrap().clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            self.brickmap_data_size as vulkano::DeviceSize,
        )
        .unwrap();

        self.device_local_buffer = Some(device_local_buffer.clone());

        let command_buffer_allocator: Arc<dyn CommandBufferAllocator> =
            Arc::new(StandardCommandBufferAllocator::new(self.device.as_ref().unwrap().clone(), Default::default()));

        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator.clone(),
            self.queue.as_ref().unwrap().queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        command_buffer_builder.copy_buffer(vulkano::command_buffer::CopyBufferInfo::buffers(
            temporary_accessible_buffer,
            device_local_buffer.clone(),
        ))
        .unwrap();
        let command_buffer = command_buffer_builder.build().unwrap();

        command_buffer.execute(self.queue.as_ref().unwrap().clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None /* timeout */)
            .unwrap()
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

        // player
        let view_mat = self.player.get_view();
        let proj_mat = self.player.get_proj(self.width, self.height);
        let player_flags = 
            (self.player.is_placing_voxel as u32) << 1 | 
            (self.player.is_breaking_voxel as u32) << 0;

        #[repr(C)]
        #[derive(Copy, Clone, Default, BufferContents)]
        struct GlobalUniforms {
            view: [[f32; 4]; 4],
            proj: [[f32; 4]; 4],
            time: f32,
            brickmap_data_size: u32,
            player_flags: u32,
            _padding: [f32; 1],
        }

        let uniform_contents = GlobalUniforms{
            view: view_mat.clone().into(),
            proj: proj_mat.clone().into(),
            time: self.time,
            brickmap_data_size: self.brickmap_data_size,
            player_flags: player_flags,
            _padding: [0.0; 1],
        };

        self.uniform_allocator = Some(SubbufferAllocator::new(
            self.memory_allocator.as_ref().unwrap().clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        ));

        let uniform_subbuffer = self.uniform_allocator.as_ref().unwrap().allocate_sized().unwrap();
        *uniform_subbuffer.write().unwrap() = uniform_contents.clone();

        self.descriptor_set_allocator = Some(Arc::new(StandardDescriptorSetAllocator::new(
            self.device.as_ref().unwrap().clone(),
            Default::default(),
        )));
        let pipeline_layout = self.compute_pipeline.as_ref().unwrap().layout();
        let descriptor_set_layout = pipeline_layout.set_layouts().get(0).unwrap();
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.as_ref().unwrap().clone(),
            descriptor_set_layout.clone(),
            [
                WriteDescriptorSet::image_view(0, computed_image_view.clone()),
                WriteDescriptorSet::buffer(1, uniform_subbuffer.clone()),
                WriteDescriptorSet::buffer(2, self.device_local_buffer.as_ref().unwrap().clone()),
            ],
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
                0,
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
            .unwrap())
        );

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

        self.upload_brickmap();

        //self.update_compute_pipeline_vulkano();
        self.update_compute_pipeline_shaderc();
        self.update_frame();

        if let Some(window) = self.window.as_ref() {
            let _ = window.set_cursor_grab(winit::window::CursorGrabMode::Locked);
            window.set_cursor_visible(false);
        }

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
                    //println!("window resized to {}x{}",self.width,self.height);
                }
            },
            WindowEvent::RedrawRequested => {
                self.time = self.instant.elapsed().as_secs_f32();
                self.dt = self.time - self.last_time;
                self.last_time = self.time;

                self.fps_accum += 1;
                self.fps_time_accum += self.dt;

                if self.fps_time_accum >= 1.0 {
                    let fps = self.fps_accum as f32 / self.fps_time_accum;
                    let title = format!("Voxel Vulkano: {:.2}", fps);
                    self.window.as_ref().unwrap().set_title(&title);

                    self.fps_accum = 0;
                    self.fps_time_accum = 0.0;
                }

                let rot = self.player.isom.rotation;
                let movements = [
                    (InputButton::Key(KeyCode::KeyW), rot * Vector3::z_axis()),
                    (InputButton::Key(KeyCode::KeyS), -(rot * Vector3::z_axis())),
                    (InputButton::Key(KeyCode::KeyA), -(rot * Vector3::x_axis())),
                    (InputButton::Key(KeyCode::KeyD), rot * Vector3::x_axis()),
                    //(InputButton::Key(KeyCode::Space), Vector3::y_axis()),
                    //(InputButton::Key(KeyCode::ShiftLeft), -Vector3::y_axis()),
                ];
                let total_movement: Vector3<f32> = movements.iter()
                    .filter(|(key, _)| self.pressed.contains(key))
                    .map(|(_, dir)| dir.into_inner())
                    .sum();

                let mut input_velocity = Vector3::zeros();

                if self.player.physics_enabled {
                    input_velocity += total_movement;
                    if(input_velocity.norm() > 0.0 && self.player.is_on_ground == true) {
                        input_velocity = input_velocity.normalize() * self.player.speed;
                    }

                    self.player.velocity += input_velocity;
                } else {
                    self.player.isom.translation.vector += total_movement * self.dt * self.player.speed;
                }

                let left_mouse = self.pressed.contains(&InputButton::Mouse(MouseButton::Left));
                let right_mouse = self.pressed.contains(&InputButton::Mouse(MouseButton::Right));

                self.player.is_breaking_voxel = left_mouse && !right_mouse;
                self.player.is_placing_voxel = right_mouse && !left_mouse;
                
                self.player.speed = if self.pressed.contains(&InputButton::Key(KeyCode::ControlLeft)) {200.0} else {50.0};

                if self.pressed.contains(&InputButton::Key(KeyCode::Escape)) {
                    event_loop.exit();
                }
                if self.pressed.contains(&InputButton::Key(KeyCode::KeyP)) {
                    self.player.physics_enabled = !self.player.physics_enabled;
                }

                let compute_shader_modified = fs::metadata(&self.compute_shader_path).unwrap().modified().unwrap();
                if compute_shader_modified > self.compute_shader_last_modified {
                    //self.update_compute_pipeline_vulkano();
                    self.update_compute_pipeline_shaderc();
                    self.compute_shader_last_modified = compute_shader_modified;
                }

                self.player.update_physics(self.dt);
                self.update_frame();
                self.window.as_ref().unwrap().request_redraw();
            },
            WindowEvent::KeyboardInput{device_id: _, event, is_synthetic: false} => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    match event.state {
                        ElementState::Pressed => {
                            self.pressed.insert(InputButton::Key(code));
                        },
                        ElementState::Released => {
                            self.pressed.remove(&InputButton::Key(code));
                        },
                    };
                }
            },
            WindowEvent::MouseInput{device_id: _, state, button} => {
                match state {
                    ElementState::Pressed => {
                        self.pressed.insert(InputButton::Mouse(button));
                    },
                    ElementState::Released => {
                        self.pressed.remove(&InputButton::Mouse(button));
                    },
                };
            },
            _ => (),
        }
    }

    fn device_event(&mut self, event_loop: &ActiveEventLoop, _id: winit::event::DeviceId, event: DeviceEvent) {
        match event {
            DeviceEvent::MouseMotion{delta: (dx, dy)} => {
                self.player.update_orientation(dx as f32, dy as f32);
            },
            _ => (),
        }
    }
}