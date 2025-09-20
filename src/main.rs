use winit::{
    event_loop::{ControlFlow, EventLoop},
};

mod app;
mod camera;

use crate::app::App;

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::default();

    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.set_control_flow(ControlFlow::Wait);

    let _ = event_loop.run_app(&mut app);
}