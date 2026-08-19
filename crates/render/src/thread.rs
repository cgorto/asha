//! Render-thread owner of the GPU, swapchain, scene, and frame retirement.
//!
//! It receives extracted packets, returns the next writable packet before
//! recording, then submits and presents the current one. The timeline
//! semaphore gates arena reuse; teardown waits for the GPU before destroying
//! the scene, arena buffers, semaphore, and swapchain-backed GPU state.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{Receiver, SyncSender};

use bevy::window::RawHandleWrapper;
use gpu::{Gpu, Memory, Queue, Semaphore};

use crate::{Arena, Extracted, FRAMES_IN_FLIGHT, Frame, FrameCtx, SceneInit};

pub(crate) struct ThreadConfig {
    pub wrapper: RawHandleWrapper,
    pub extent: [u32; 2],
    pub scene_init: SceneInit,
    pub arena_capacity: u64,
    pub to_main: SyncSender<Frame>,
    pub from_main: Receiver<Frame>,
    pub exit_flag: Arc<AtomicBool>,
}

pub(crate) fn run(cfg: ThreadConfig) {
    let gpu = Gpu::new(true).expect("vulkan init");
    // SAFETY: RawHandleWrapper exists to carry window handles to a render
    // thread — it keeps the winit window alive through an internal Arc, and
    // surface creation off the main thread is well-defined on X11/Wayland.
    // bevy's own renderer performs this same hand-off.
    let handle = unsafe { cfg.wrapper.get_handle() };
    gpu.swapchain_init(&handle, cfg.extent, FRAMES_IN_FLIGHT as u32);

    let mut scene = (cfg.scene_init)(&gpu);
    let frame_sem = gpu.semaphore_create(0);

    let bufs: Vec<gpu::Ptr<u8>> = (0..FRAMES_IN_FLIGHT)
        .map(|_| gpu.alloc_slice::<u8>(cfg.arena_capacity, Memory::Default))
        .collect();
    let mut pool: VecDeque<Frame> = bufs
        .iter()
        .map(|&buf| Frame {
            arena: Arena {
                buf,
                cap: cfg.arena_capacity,
                offset: 0,
            },
            extracted: Extracted::default(),
            frame: 0,
            time: 0.0,
            extent: cfg.extent,
            mesh_uploads: Vec::new(),
            mesh_updates: Vec::new(),
            mesh_removals: Vec::new(),
            material_uploads: Vec::new(),
            shader_group_uploads: Vec::new(),
            proc_texture_uploads: Vec::new(),
        })
        .collect();

    let mut extent = cfg.extent;

    send_writable(&gpu, frame_sem, &cfg.to_main, &mut pool, 1);

    while let Ok(mut frame) = cfg.from_main.recv() {
        let n = frame.frame;
        send_writable(&gpu, frame_sem, &cfg.to_main, &mut pool, n + 1);

        if frame.extent != extent {
            gpu.swapchain_resize(frame.extent);
            extent = frame.extent;
        }

        let backbuffer = gpu.swapchain_acquire_next();
        let cb = gpu.commands_begin(Queue::Main);
        scene.draw(&mut FrameCtx {
            gpu: &gpu,
            cb,
            backbuffer,
            extent: frame.extent,
            frame: n,
            time: frame.time,
            extracted: &frame.extracted,
            arena: &mut frame.arena,
            mesh_uploads: &mut frame.mesh_uploads,
            mesh_updates: &mut frame.mesh_updates,
            mesh_removals: &mut frame.mesh_removals,
            material_uploads: &mut frame.material_uploads,
            shader_group_uploads: &mut frame.shader_group_uploads,
            proc_texture_uploads: &mut frame.proc_texture_uploads,
            exit: &cfg.exit_flag,
        });
        assert!(
            frame.mesh_uploads.is_empty()
                && frame.mesh_updates.is_empty()
                && frame.mesh_removals.is_empty()
                && frame.material_uploads.is_empty()
                && frame.shader_group_uploads.is_empty()
                && frame.proc_texture_uploads.is_empty(),
            "bevy mesh/material/shader-group/texture uploads not consumed — the host scene \
             must drain FrameCtx::mesh_uploads, mesh_updates, mesh_removals, \
             material_uploads, shader_group_uploads, \
             and proc_texture_uploads"
        );
        gpu.cmd_add_signal_semaphore(cb, frame_sem, n);
        gpu.queue_submit(Queue::Main, &[cb]);
        gpu.swapchain_present(Queue::Main, frame_sem, n);
        pool.push_back(frame);
    }

    gpu.wait_idle();
    scene.teardown(&gpu);
    for buf in bufs {
        gpu.free(buf);
    }
    gpu.semaphore_destroy(frame_sem);
}

/// Returns a GPU-idle frame packet to the main thread.
///
/// The semaphore wait proves its mapped arena can be reset and reused. If the
/// channel is closed, the packet remains owned by the render-thread pool.
fn send_writable(
    gpu: &Gpu,
    frame_sem: Semaphore,
    to_main: &SyncSender<Frame>,
    pool: &mut VecDeque<Frame>,
    frame_no: u64,
) {
    if frame_no > FRAMES_IN_FLIGHT {
        gpu.semaphore_wait(frame_sem, frame_no - FRAMES_IN_FLIGHT);
    }
    let mut frame = pool.pop_front().expect("frame pool underflow");
    frame.arena.reset();
    frame.extracted.map.clear();
    frame.frame = frame_no;
    if let Err(returned) = to_main.send(frame) {
        pool.push_front(returned.0);
    }
}
