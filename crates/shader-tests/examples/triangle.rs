//! Vertex-pulling triangle using physical pointers.
//!
//! Requires prebuilt triangle shaders.

use abi_core::TriangleData;
use app::FrameClock;
use asha_assets::load_spv;
use gpu::{
    Gpu, LoadOp, Memory, Queue, RenderAttachment, RenderPassDesc, Semaphore, ShaderTypeGraphics,
    StoreOp,
};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

const FRAMES_IN_FLIGHT: u64 = 3;
const FIXED_DT: f32 = 1.0 / 120.0;
const FIXED_STEPS_MAX: u32 = 8;

struct Scene {
    gpu: Gpu,
    window: Window,
    vert: gpu::Shader,
    frag: gpu::Shader,
    tri: gpu::Ptr<TriangleData>,
    positions: gpu::Ptr<[f32; 2]>,
    colors: gpu::Ptr<[f32; 4]>,
    tint: gpu::Ptr<[f32; 4]>,
    indices: gpu::Ptr<u32>,
    frame_sem: Semaphore,
    next_frame: u64,
    clock: FrameClock,
    time: f32,
}

#[derive(Default)]
struct App {
    scene: Option<Scene>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.scene.is_some() {
            return;
        }
        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title("asha — first pixels")
                    .with_inner_size(winit::dpi::LogicalSize::new(960, 720)),
            )
            .unwrap();

        let gpu = Gpu::new(true).expect("vulkan init");
        let size = window.inner_size();
        gpu.swapchain_init(&window, [size.width, size.height], FRAMES_IN_FLIGHT as u32);

        let vert = gpu.shader_create(
            &load_spv("triangle_vert"),
            ShaderTypeGraphics::Vertex,
            "triangle_vert",
        );
        let frag = gpu.shader_create(
            &load_spv("triangle_frag"),
            ShaderTypeGraphics::Fragment,
            "triangle_frag",
        );

        let positions = gpu.alloc_slice::<[f32; 2]>(3, Memory::Default);
        let colors = gpu.alloc_slice::<[f32; 4]>(3, Memory::Default);
        let tint = gpu.alloc::<[f32; 4]>(Memory::Default);
        let tri = gpu.alloc::<TriangleData>(Memory::Default);
        let indices = gpu.alloc_slice::<u32>(3, Memory::Default);
        unsafe {
            // Vulkan NDC uses a downward positive Y direction here.
            *positions.cpu.add(0) = [0.0, -0.6];
            *positions.cpu.add(1) = [-0.55, 0.5];
            *positions.cpu.add(2) = [0.55, 0.5];
            *colors.cpu.add(0) = [1.0, 0.2, 0.2, 1.0];
            *colors.cpu.add(1) = [0.2, 0.4, 1.0, 1.0];
            *colors.cpu.add(2) = [0.2, 1.0, 0.2, 1.0];
            *tint.cpu = [1.0, 1.0, 1.0, 1.0];
            *tri.cpu = TriangleData {
                positions: positions.gpu,
                colors: colors.gpu,
                tint: tint.gpu,
            };
            for i in 0..3 {
                *indices.cpu.add(i) = i as u32;
            }
        }

        let frame_sem = gpu.semaphore_create(0);
        self.scene = Some(Scene {
            gpu,
            window,
            vert,
            frag,
            tri,
            positions,
            colors,
            tint,
            indices,
            frame_sem,
            next_frame: 0,
            clock: FrameClock::new(),
            time: 0.0,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(scene) = self.scene.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } => {
                if event.logical_key == Key::Named(NamedKey::Escape) {
                    event_loop.exit();
                }
            }
            WindowEvent::Resized(size) => {
                if size.width > 0 && size.height > 0 {
                    scene.gpu.swapchain_resize([size.width, size.height]);
                } else {
                    scene.clock.reset();
                }
            }
            WindowEvent::RedrawRequested => {
                scene.draw();
                scene.window.request_redraw();
            }
            _ => {}
        }
    }
}

impl Scene {
    fn draw(&mut self) {
        // Bound CPU work to completed frames.
        self.next_frame += 1;
        if self.next_frame > FRAMES_IN_FLIGHT {
            self.gpu
                .semaphore_wait(self.frame_sem, self.next_frame - FRAMES_IN_FLIGHT);
        }

        let dt = app::dt_clamped(self.clock.tick(), 0.25);
        let _due = self.clock.advance(dt, false, FIXED_DT, FIXED_STEPS_MAX);
        self.time += dt;

        let backbuffer = self.gpu.swapchain_acquire_next();

        let cb = self.gpu.commands_begin(Queue::Main);
        let t = self.time;
        let clear = [
            0.03 + 0.02 * (t * 0.7).sin().abs(),
            0.03,
            0.06 + 0.03 * (t * 0.4).cos().abs(),
            1.0,
        ];
        self.gpu.cmd_begin_render_pass(
            cb,
            RenderPassDesc {
                color_attachments: &[RenderAttachment {
                    texture: backbuffer,
                    load_op: LoadOp::Clear,
                    store_op: StoreOp::Store,
                    clear_color: clear,
                    ..Default::default()
                }],
                ..Default::default()
            },
        );
        self.gpu.cmd_set_shaders(cb, self.vert, self.frag);
        self.gpu.cmd_draw_indexed_instanced(
            cb,
            self.tri.gpu.cast(),
            self.tri.gpu.cast(),
            self.indices.cast(),
            3,
            1,
        );
        self.gpu.cmd_end_render_pass(cb);
        self.gpu
            .cmd_add_signal_semaphore(cb, self.frame_sem, self.next_frame);
        self.gpu.queue_submit(Queue::Main, &[cb]);
        self.gpu
            .swapchain_present(Queue::Main, self.frame_sem, self.next_frame);
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
    if let Some(scene) = app.scene.take() {
        scene.gpu.wait_idle();
        scene.gpu.shader_destroy(scene.vert);
        scene.gpu.shader_destroy(scene.frag);
        scene.gpu.semaphore_destroy(scene.frame_sem);
        scene.gpu.free(scene.tri);
        scene.gpu.free(scene.positions);
        scene.gpu.free(scene.colors);
        scene.gpu.free(scene.tint);
        scene.gpu.free(scene.indices);
    }
}
