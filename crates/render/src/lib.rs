//! The Bevy–renderer seam: Bevy owns CPU state; the renderer owns the GPU.
//!
//! Three frame packets circulate by value between the main and render threads.
//! The render thread returns a packet only after a timeline-semaphore wait proves
//! its arena is GPU-idle; the main thread extracts components, time, extent, and
//! registration lanes into it. No ECS or GPU state crosses the channel.
//!
//! The render thread owns the `Gpu`, swapchain, and [`RenderScene`]. It creates
//! the scene after GPU initialization and recreates both when the window/surface
//! lifecycle restarts. Extraction and recording overlap by handing out the next
//! writable packet before recording the current one.

mod effects;
#[cfg(feature = "gltf")]
pub mod gltf;
mod local_lighting;
mod local_shadow;
mod meshes;
mod normals;
mod pacing;
mod proc_texture;
mod thread;
pub use effects::{EffectGroup, ShaderEffect, ShaderEffectAppExt};
pub use local_lighting::{LocalLightPass, MeshShadowMask, MeshShadowPass, MeshSurfaceTargets};
pub use local_shadow::{
    LocalShadowDirectPass, LocalShadowPass, LocalShadowSlots, LocalShadowTemporal,
};
pub use meshes::{
    MaterialUpload, MeshCoat, MeshInstanceColor, MeshInstanceFlags, MeshMaterial, MeshMaterials,
    MeshRemoval, MeshShader, MeshSkin, MeshUpdate, MeshUpload, ShaderGroupDesc, ShaderGroupMode,
    ShaderGroupUpload, ShaderGroups,
};
pub use normals::{SoftenError, soften_normals};
pub use pacing::PacingPlugin;
pub use proc_texture::{
    ProcTexture, ProcTextureBridge, ProcTextureFill, ProcTextureUpload, ProcTextures,
};

use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::ThreadId;

use bevy::app::AppExit;
use bevy::ecs::message::Messages;
use bevy::prelude::*;
use bevy::window::{PrimaryWindow, RawHandleWrapper, Window, WindowCloseRequested};
use bytemuck::Pod;
use gpu::{CommandBuffer, Gpu, GpuPtr, Texture};

pub const FRAMES_IN_FLIGHT: u64 = 3;

/// Default per-slot arena capacity: 16 MiB × FRAMES_IN_FLIGHT.
pub const DEFAULT_ARENA_CAPACITY: u64 = 16 << 20;

/// Bump allocator over persistently mapped GPU memory.
///
/// Each frame owns one arena. It is handed back only after the semaphore
/// proves the GPU has finished reading it; resetting then reuses the mapping.
struct Arena {
    buf: gpu::Ptr<u8>,
    cap: u64,
    offset: u64,
}

// SAFETY: the mapping is persistent for the buffer's lifetime, and an Arena
// is owned by exactly one thread at a time (it moves through the Frame
// channels; it is never aliased). GPU reads are fenced by the frame
// semaphore wait before each hand-off.
unsafe impl Send for Arena {}

impl Arena {
    fn reset(&mut self) {
        self.offset = 0;
    }

    fn alloc_bytes(&mut self, size: u64, align: u64) -> (GpuPtr<u8>, *mut u8) {
        debug_assert!(align.is_power_of_two());
        let start = self.offset.next_multiple_of(align);
        assert!(
            start + size <= self.cap,
            "frame arena overflow: {size}B requested at offset {start}, capacity {}B — raise \
             AshaRenderPlugin::arena_capacity",
            self.cap,
        );
        self.offset = start + size;
        (
            self.buf.gpu.byte_offset(start as i64),
            // SAFETY: start + size <= cap was just asserted; the mapping is
            // as large as the allocation.
            unsafe { self.buf.cpu.add(start as usize) },
        )
    }
}

/// Per-frame extracted arrays: arena addresses and host-lane snapshots.
#[derive(Default)]
pub struct Extracted {
    map: HashMap<TypeId, (u64, u32)>,
    /// Typed host-lane snapshots retained in cached system RAM.
    host: HashMap<TypeId, Box<dyn Any + Send>>,
}

impl Extracted {
    /// Returns `T`'s arena pointer and count.
    ///
    /// Panics when `T` was not registered with [`AshaRenderPlugin::extract`].
    pub fn get<T: 'static>(&self) -> (GpuPtr<T>, u32) {
        let (addr, count) = *self.map.get(&TypeId::of::<T>()).unwrap_or_else(|| {
            panic!("type {} was never .extract()ed", std::any::type_name::<T>())
        });
        (GpuPtr::from_addr(addr), count)
    }

    /// Returns `T`'s retained host-lane snapshot.
    ///
    /// Panics when `T` was not registered with [`AshaRenderPlugin::extract_host`].
    pub fn get_host<T: 'static>(&self) -> &[T] {
        self.host
            .get(&TypeId::of::<T>())
            .unwrap_or_else(|| {
                panic!(
                    "type {} was never .extract_host()ed",
                    std::any::type_name::<T>()
                )
            })
            .downcast_ref::<Vec<T>>()
            .expect("host lane entry holds the Vec of its own TypeId")
    }
}

/// Copies component tables into a retained cached host-lane vector.
fn make_extract_host<T: Component + Pod>() -> ExtractFn {
    let mut query: Option<QueryState<&'static T>> = None;
    Box::new(move |world, frame| {
        let qs = query.get_or_insert_with(|| world.query::<&T>());
        let rows = frame
            .extracted
            .host
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Box::new(Vec::<T>::new()))
            .downcast_mut::<Vec<T>>()
            .expect("host lane entry holds the Vec of its own TypeId");
        rows.clear();
        for slice in qs
            .query(&*world)
            .contiguous_iter()
            .expect("gpu_data components use table storage")
        {
            rows.extend_from_slice(slice);
        }
    })
}

/// Clone-extracts snapshot components into retained host-lane storage.
fn make_extract_host_clone<T: Component + Clone>() -> ExtractFn {
    let mut query: Option<QueryState<&'static T>> = None;
    Box::new(move |world, frame| {
        let qs = query.get_or_insert_with(|| world.query::<&T>());
        let rows = frame
            .extracted
            .host
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Box::new(Vec::<T>::new()))
            .downcast_mut::<Vec<T>>()
            .expect("host lane entry holds the Vec of its own TypeId");
        let mut write = 0;
        for slice in qs
            .query(&*world)
            .contiguous_iter()
            .expect("clone-extracted components use table storage")
        {
            for source in slice {
                if let Some(destination) = rows.get_mut(write) {
                    destination.clone_from(source);
                } else {
                    rows.push(source.clone());
                }
                write += 1;
            }
        }
        rows.truncate(write);
    })
}

/// Copies a resource-owned `Pod` slice into the frame arena.
fn make_extract_resource_slice<R: Resource, T: Pod>(accessor: fn(&R) -> &[T]) -> ExtractFn {
    Box::new(move |world, frame| {
        let slice = accessor(world.resource::<R>());
        let (gpu_base, dst) = frame.arena.alloc_bytes(
            (slice.len() * size_of::<T>()) as u64,
            align_of::<T>().max(4) as u64,
        );
        let bytes = bytemuck::cast_slice::<T, u8>(slice);
        // SAFETY: dst is a fresh arena run sized for exactly `slice.len()` T.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len()) };
        frame
            .extracted
            .map
            .insert(TypeId::of::<T>(), (gpu_base.addr(), slice.len() as u32));
    })
}

/// Clone-copies a resource-owned slice into retained host storage.
fn make_extract_resource_host_clone<R: Resource, H: Clone + Send + 'static>(
    accessor: fn(&R) -> &[H],
) -> ExtractFn {
    Box::new(move |world, frame| {
        let rows = frame
            .extracted
            .host
            .entry(TypeId::of::<H>())
            .or_insert_with(|| Box::new(Vec::<H>::new()))
            .downcast_mut::<Vec<H>>()
            .expect("host lane entry holds the Vec of its own TypeId");
        let source = accessor(world.resource::<R>());
        let mut write = 0;
        for item in source {
            if let Some(destination) = rows.get_mut(write) {
                destination.clone_from(item);
            } else {
                rows.push(item.clone());
            }
            write += 1;
        }
        rows.truncate(write);
    })
}

/// One CPU-to-GPU packet, circulating by value between both threads.
///
/// The render thread timestamps and returns it; the main thread fills its
/// arena, extraction table, extent, and registration lanes before sending it.
struct Frame {
    arena: Arena,
    extracted: Extracted,
    frame: u64,
    time: f32,
    extent: [u32; 2],
    mesh_uploads: Vec<MeshUpload>,
    mesh_updates: Vec<meshes::MeshUpdate>,
    mesh_removals: Vec<meshes::MeshRemoval>,
    material_uploads: Vec<MaterialUpload>,
    shader_group_uploads: Vec<meshes::ShaderGroupUpload>,
    proc_texture_uploads: Vec<ProcTextureUpload>,
}

/// One registered extract fills arena bytes, host data, and upload lanes.
type ExtractFn = Box<dyn FnMut(&mut World, &mut Frame)>;

fn make_extract<T: Component + Pod>() -> ExtractFn {
    let mut query: Option<QueryState<&'static T>> = None;
    Box::new(move |world, frame| {
        let (arena, out) = (&mut frame.arena, &mut frame.extracted);
        let qs = query.get_or_insert_with(|| world.query::<&T>());
        let total: usize = qs.query(&*world).iter().len();
        let (gpu_base, mut dst) = arena.alloc_bytes(
            (total * size_of::<T>()) as u64,
            align_of::<T>().max(4) as u64,
        );
        for slice in qs
            .query(&*world)
            .contiguous_iter()
            .expect("gpu_data components use table storage")
        {
            let bytes = bytemuck::cast_slice::<T, u8>(slice);
            // SAFETY: dst walks an arena run sized for `total` elements above.
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
                dst = dst.add(bytes.len());
            }
        }
        out.map
            .insert(TypeId::of::<T>(), (gpu_base.addr(), total as u32));
    })
}

/// Per-frame recording context owned by the render thread.
///
/// Its command buffer is begun on `Queue::Main`; the render thread submits and
/// presents after [`RenderScene::draw`] returns. The render thread initializes
/// every public field and packet lane before calling `draw`. Every packet
/// lane must be fully consumed before recording completes; mesh lanes are
/// consumed in removals, updates, then adds order.
///
/// Extracted pointers and all arena mappings remain valid only until this
/// frame finishes recording; the semaphore then gates arena reuse.
pub struct FrameCtx<'w> {
    pub gpu: &'w Gpu,
    pub cb: CommandBuffer,
    pub backbuffer: Texture,
    /// Physical swapchain extent in pixels for this frame.
    ///
    /// Derive viewport and projection sizes from this value, not logical
    /// window dimensions; it includes the platform scale factor.
    pub extent: [u32; 2],
    /// Monotonic frame index (first rendered frame is 1).
    pub frame: u64,
    /// `Time` elapsed seconds as of this frame's extract.
    pub time: f32,
    extracted: &'w Extracted,
    arena: &'w mut Arena,
    mesh_uploads: &'w mut Vec<MeshUpload>,
    mesh_updates: &'w mut Vec<meshes::MeshUpdate>,
    mesh_removals: &'w mut Vec<meshes::MeshRemoval>,
    material_uploads: &'w mut Vec<MaterialUpload>,
    shader_group_uploads: &'w mut Vec<meshes::ShaderGroupUpload>,
    proc_texture_uploads: &'w mut Vec<ProcTextureUpload>,
    exit: &'w AtomicBool,
}

impl FrameCtx<'_> {
    /// Returns the extracted component array for `T`.
    pub fn extracted<T: 'static>(&self) -> (GpuPtr<T>, u32) {
        self.extracted.get::<T>()
    }

    /// Places `value` in the frame arena and returns its device address.
    pub fn frame_alloc<T: Pod>(&mut self, value: T) -> GpuPtr<T> {
        let (gpu_base, dst) = self
            .arena
            .alloc_bytes(size_of::<T>() as u64, align_of::<T>().max(4) as u64);
        // SAFETY: dst is a fresh arena run, sized and aligned for T above.
        unsafe { dst.cast::<T>().write(value) };
        gpu_base.cast()
    }

    /// Reserves a mapped arena run and returns GPU and CPU addresses.
    ///
    /// The CPU mapping is valid through this frame's recording. The caller
    /// must initialize all elements before passing the GPU pointer to a pass.
    pub fn frame_alloc_slice<T: Pod>(&mut self, count: u32) -> (GpuPtr<T>, *mut T) {
        let bytes = (count as u64)
            .checked_mul(size_of::<T>() as u64)
            .expect("frame slice byte count overflow");
        let (gpu_base, cpu) = self.arena.alloc_bytes(bytes, align_of::<T>().max(4) as u64);
        (gpu_base.cast(), cpu.cast())
    }

    /// Returns the extracted host-lane array for `T`.
    pub fn extracted_host<T: 'static>(&self) -> &[T] {
        self.extracted.get_host::<T>()
    }

    /// Returns a mutable view of an extracted arena stream.
    ///
    /// The view is valid only while this frame is being recorded; `&mut self`
    /// prevents simultaneous mapped views.
    pub fn extracted_host_mut<T: 'static + Pod>(&mut self) -> &mut [T] {
        let (ptr, count) = self.extracted.get::<T>();
        let offset = ptr.addr() - self.arena.buf.gpu.addr();
        // SAFETY: extraction wrote `count` T values at this arena offset;
        // `&mut self` guarantees no simultaneous mapped view exists.
        unsafe {
            std::slice::from_raw_parts_mut(
                self.arena.buf.cpu.add(offset as usize).cast::<T>(),
                count as usize,
            )
        }
    }

    /// Drains mesh additions in main-thread index order, after removals and
    /// updates. The host must consume every packet lane before recording ends.
    pub fn mesh_uploads(&mut self) -> std::vec::Drain<'_, MeshUpload> {
        self.mesh_uploads.drain(..)
    }

    /// Drains removed mesh slots; consume before updates and additions.
    pub fn mesh_removals(&mut self) -> std::vec::Drain<'_, meshes::MeshRemoval> {
        self.mesh_removals.drain(..)
    }

    /// Drains rewrites of registered mesh slots after removals.
    pub fn mesh_updates(&mut self) -> std::vec::Drain<'_, meshes::MeshUpdate> {
        self.mesh_updates.drain(..)
    }

    /// Drains registered materials in index order.
    pub fn material_uploads(&mut self) -> std::vec::Drain<'_, MaterialUpload> {
        self.material_uploads.drain(..)
    }

    /// Drains registered shader groups in index order.
    pub fn shader_group_uploads(&mut self) -> std::vec::Drain<'_, meshes::ShaderGroupUpload> {
        self.shader_group_uploads.drain(..)
    }

    /// Drains registered procedural textures in index order.
    pub fn proc_texture_uploads(&mut self) -> std::vec::Drain<'_, ProcTextureUpload> {
        self.proc_texture_uploads.drain(..)
    }

    /// Flushes stdout before signaling application exit so verification output
    /// survives teardown.
    pub fn request_exit(&self) {
        use std::io::Write as _;
        let _ = std::io::stdout().flush();
        self.exit.store(true, Ordering::Relaxed);
    }
}

/// Supplies pass dispatch allocations from the current frame arena.
impl gpu::pass::FrameAlloc for FrameCtx<'_> {
    fn frame_alloc<T: Pod>(&mut self, value: T) -> GpuPtr<T> {
        FrameCtx::frame_alloc(self, value)
    }

    fn frame_alloc_slice<T: Pod>(&mut self, values: &[T]) -> GpuPtr<T> {
        if values.is_empty() {
            return GpuPtr::null();
        }
        let count = u32::try_from(values.len()).expect("frame slice count exceeds u32");
        let (gpu, cpu) = FrameCtx::frame_alloc_slice(self, count);
        // SAFETY: the fresh arena run holds exactly `count == values.len()` T.
        unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), cpu, values.len()) };
        gpu
    }
}

/// Render-thread scene that owns and records all GPU resources.
///
/// The factory may be called repeatedly when the window and GPU are rebuilt;
/// [`teardown`](RenderScene::teardown) runs before each GPU is destroyed.
pub trait RenderScene: 'static {
    fn draw(&mut self, ctx: &mut FrameCtx);
    fn teardown(&mut self, gpu: &Gpu);
}

type SceneInit = Arc<dyn Fn(&Gpu) -> Box<dyn RenderScene> + Send + Sync>;

/// Main-thread channel endpoints and extraction closures.
struct RenderLink {
    join: Option<std::thread::JoinHandle<()>>,
    to_render: Option<SyncSender<Frame>>,
    from_render: Receiver<Frame>,
    exit_flag: Arc<AtomicBool>,
    extracts: Vec<ExtractFn>,
    /// Holds a writable packet while the window has zero physical extent.
    /// Minimization pauses extraction and rendering without dropping it.
    stash: Option<Frame>,
}

thread_local! {
    static RENDER: RefCell<Option<RenderLink>> = const { RefCell::new(None) };
}

/// Initialization config retained until the primary window exists.
#[derive(Resource)]
struct PendingInit {
    scene: SceneInit,
    extracts: Vec<fn() -> ExtractFn>,
    resource_extracts: Vec<Arc<dyn Fn() -> ExtractFn + Send + Sync>>,
    arena_capacity: u64,
}

/// Identifies the thread that must run exclusive plugin systems.
#[derive(Resource, Clone, Copy)]
struct MainThread(ThreadId);

pub struct AshaRenderPlugin {
    scene_init: Mutex<Option<SceneInit>>,
    extracts: Vec<fn() -> ExtractFn>,
    resource_extracts: Vec<Arc<dyn Fn() -> ExtractFn + Send + Sync>>,
    meshes: bool,
    /// Reserved procedural-texture block, when configured.
    proc_textures: Option<(u32, u32)>,
    arena_capacity: u64,
}

impl AshaRenderPlugin {
    /// Constructs a plugin whose scene factory runs on the render thread.
    ///
    /// The factory runs after GPU and swapchain creation and may run again
    /// after a window or surface restart; it must be reinitializable.
    pub fn new<S, F>(scene_init: F) -> Self
    where
        S: RenderScene,
        F: Fn(&Gpu) -> S + Send + Sync + 'static,
    {
        Self {
            scene_init: Mutex::new(Some(Arc::new(move |gpu| Box::new(scene_init(gpu)) as _))),
            extracts: Vec::new(),
            resource_extracts: Vec::new(),
            meshes: false,
            proc_textures: None,
            arena_capacity: DEFAULT_ARENA_CAPACITY,
        }
    }

    /// Registers Bevy mesh asset and instance extraction.
    pub fn extract_meshes(mut self) -> Self {
        self.meshes = true;
        self.extracts.push(meshes::make_mesh_extract);
        self
    }

    /// Registers `T` for per-frame arena extraction in ECS table order, not
    /// entity spawn order.
    pub fn extract<T: Component + Pod>(mut self) -> Self {
        self.extracts.push(make_extract::<T>);
        self
    }

    /// Registers `T` for per-frame cached host extraction in ECS table order,
    /// not entity spawn order.
    pub fn extract_host<T: Component + Pod>(mut self) -> Self {
        self.extracts.push(make_extract_host::<T>);
        self
    }

    /// Clone-extracts a host-only snapshot component in ECS table order, not
    /// entity spawn order.
    pub fn extract_host_clone<T: Component + Clone>(mut self) -> Self {
        self.extracts.push(make_extract_host_clone::<T>);
        self
    }

    /// Registers a resource-owned `Pod` slice for arena extraction.
    pub fn extract_resource_slice<R: Resource, T: Pod>(mut self, accessor: fn(&R) -> &[T]) -> Self {
        self.resource_extracts.push(Arc::new(move || {
            make_extract_resource_slice::<R, T>(accessor)
        }));
        self
    }

    /// Registers a resource-owned slice for cached host extraction.
    pub fn extract_resource_host_clone<R: Resource, H: Clone + Send + 'static>(
        mut self,
        accessor: fn(&R) -> &[H],
    ) -> Self {
        self.resource_extracts.push(Arc::new(move || {
            make_extract_resource_host_clone::<R, H>(accessor)
        }));
        self
    }

    /// Registers procedural-texture extraction with a reserved slot block.
    pub fn proc_textures(mut self, base_slot: u32, capacity: u32) -> Self {
        self.proc_textures = Some((base_slot, capacity));
        self.extracts.push(proc_texture::make_extract);
        self
    }

    /// Per-frame-slot arena capacity in bytes (allocated FRAMES_IN_FLIGHT times).
    pub fn arena_capacity(mut self, bytes: u64) -> Self {
        self.arena_capacity = bytes;
        self
    }
}

impl Plugin for AshaRenderPlugin {
    fn build(&self, app: &mut App) {
        let init = self
            .scene_init
            .lock()
            .unwrap()
            .take()
            .expect("AshaRenderPlugin::build ran twice");
        if self.meshes {
            app.init_resource::<MeshMaterials>();
            app.init_resource::<meshes::ShaderGroups>();
            if !app.is_plugin_added::<bevy::mesh::MeshPlugin>() {
                app.add_plugins(bevy::mesh::MeshPlugin);
            }
        }
        if let Some((base_slot, capacity)) = self.proc_textures {
            app.insert_resource(ProcTextures::new(base_slot, capacity));
        }
        app.insert_resource(PendingInit {
            scene: init,
            extracts: self.extracts.clone(),
            resource_extracts: self.resource_extracts.clone(),
            arena_capacity: self.arena_capacity,
        })
        .insert_resource(MainThread(std::thread::current().id()))
        .add_systems(PreUpdate, init_render)
        .add_systems(Last, extract_frame);
    }
}

fn primary_window(world: &mut World) -> Option<(RawHandleWrapper, [u32; 2])> {
    let mut query = world.query_filtered::<(&RawHandleWrapper, &Window), With<PrimaryWindow>>();
    let (wrapper, window) = query.single(world).ok()?;
    Some((
        wrapper.clone(),
        [window.physical_width(), window.physical_height()],
    ))
}

fn init_render(world: &mut World) {
    assert_eq!(
        std::thread::current().id(),
        world.resource::<MainThread>().0
    );
    if RENDER.with_borrow(|r| r.is_some()) {
        return;
    }
    let Some((wrapper, extent)) = primary_window(world) else {
        return;
    };

    let pending = world.resource::<PendingInit>();
    let scene_init = pending.scene.clone();
    let extracts: Vec<ExtractFn> = pending
        .extracts
        .iter()
        .map(|f| f())
        .chain(pending.resource_extracts.iter().map(|f| f()))
        .collect();

    let (to_main, from_render) = std::sync::mpsc::sync_channel(FRAMES_IN_FLIGHT as usize);
    let (to_render, from_main) = std::sync::mpsc::sync_channel(1);
    let exit_flag = Arc::new(AtomicBool::new(false));

    let cfg = thread::ThreadConfig {
        wrapper,
        extent,
        scene_init,
        arena_capacity: pending.arena_capacity,
        to_main,
        from_main,
        exit_flag: exit_flag.clone(),
    };
    let join = std::thread::Builder::new()
        .name("asha-render".into())
        .spawn(move || thread::run(cfg))
        .expect("spawn render thread");

    RENDER.with_borrow_mut(|r| {
        *r = Some(RenderLink {
            join: Some(join),
            to_render: Some(to_render),
            from_render,
            exit_flag,
            extracts,
            stash: None,
        });
    });
}

/// Joins a dead render thread and re-raises its panic on the main thread.
fn propagate_render_death(mut link: RenderLink) -> ! {
    link.to_render = None;
    link.stash = None;
    match link.join.take().expect("join handle").join() {
        Ok(()) => panic!("render thread exited unexpectedly"),
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

fn extract_frame(world: &mut World) {
    assert_eq!(
        std::thread::current().id(),
        world.resource::<MainThread>().0
    );
    RENDER.with_borrow_mut(|render| {
        let Some(link) = render.as_mut() else { return };

        if link.exit_flag.load(Ordering::Relaxed) {
            world.write_message(AppExit::Success);
        }

        let exiting = world
            .get_resource::<Messages<AppExit>>()
            .is_some_and(|m| !m.is_empty())
            || world
                .get_resource::<Messages<WindowCloseRequested>>()
                .is_some_and(|m| !m.is_empty());
        let window = primary_window(world);
        if exiting || window.is_none() {
            let mut link = render.take().expect("link present");
            link.to_render = None;
            link.stash = None;
            if let Err(payload) = link.join.take().expect("join handle").join() {
                std::panic::resume_unwind(payload);
            }
            return;
        }

        let mut frame = match link.stash.take() {
            Some(frame) => frame,
            None => match link.from_render.recv() {
                Ok(frame) => frame,
                Err(_) => propagate_render_death(render.take().expect("link present")),
            },
        };

        let (_, extent) = window.expect("checked above");
        if extent[0] == 0 || extent[1] == 0 {
            link.stash = Some(frame); // minimized: pause the pipeline
            return;
        }

        frame.extent = extent;
        frame.time = world.resource::<Time>().elapsed_secs();
        for extract in &mut link.extracts {
            extract(world, &mut frame);
        }
        let to_render = link.to_render.as_ref().expect("sender present");
        if to_render.send(frame).is_err() {
            propagate_render_death(render.take().expect("link present"));
        }
    });
}
