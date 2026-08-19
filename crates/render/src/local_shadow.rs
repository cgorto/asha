//! Half-resolution local-shadow producers and their shared slot contract.
//!
//! The temporal producer selects top-contributing lights, reuses validated
//! slot history, compacts ray requests, traces indirectly, and optionally
//! reconstructs penumbra. The direct producer uses the same outputs without
//! history. A zero temporal interval disables reuse; zero source radius keeps
//! binary visibility. Storage is bounded by half-resolution texels times the
//! fixed slot count, not full pixels times lights.

use abi_core::GpuPtr;
use abi_core::glam::{Mat4, UVec2};
use abi_light::PointLight;
use abi_light::{
    DepthMarchConfig, LOCAL_SHADOW_SLOT_EMPTY, LOCAL_SHADOW_SLOTS, LocalShadowArgsData,
    LocalShadowCounters, LocalShadowData, depth_march_config_valid,
};
use gpu::pass::{FrameAlloc, Pass};
use gpu::{
    CommandBuffer, DispatchIndirectCommand, Gpu, HazardFlags, HeapSlots, Memory, Queue,
    SampledSlot, Stage,
};
use mesh::{MeshInstances, MeshRasterView, MeshScene, ShadowTlasBuilder, ShadowTlasStats};

use crate::local_lighting::MeshSurfaceTargets;

/// `local_shadow_select`'s workgroup edge over the half-res grid.
const SELECT_GROUP_SIZE: u32 = 8;

/// Temporal reuse dials, validated at record time.
#[derive(Debug, Clone, Copy)]
pub struct LocalShadowTemporal {
    /// Frames visible history remains trusted; zero disables reuse.
    pub refresh_interval: u32,
    /// World-unit thickness for occluded-history validation.
    pub validate_thickness: f32,
    pub near_plane: f32,
    /// Light-motion threshold that invalidates visible history.
    pub light_epsilon: f32,
    /// Contact-march controls; `None` disables the stage.
    pub contact: Option<DepthMarchConfig>,
    /// World-unit reach of the contact march from the receiver.
    pub contact_distance: f32,
    /// Rays serviced per frame; zero means unlimited.
    ///
    /// The high-priority queue runs before the low-priority queue; deferred
    /// slots retain estimates and converge on later frames.
    pub ray_budget: u32,
    /// Visibility-space edge promotion (dithered edge supersampling).
    pub edge_promotion: bool,
    /// Frames an occlusion may live before low-queue reproof.
    ///
    /// Zero disables periodic reproof; dynamic and blind policies still apply.
    pub occluded_refresh: u32,
    /// Light-source radius for penumbra; zero keeps edges hard.
    pub source_radius: f32,
}

/// Resolve input: slot map, packed states, representatives, and fractions.
#[derive(Clone, Copy)]
pub struct LocalShadowSlots {
    slot_map: GpuPtr<u32>,
    slot_state: GpuPtr<u32>,
    slot_fraction: GpuPtr<u32>,
    slot_rep: GpuPtr<u32>,
    half_size: UVec2,
    size: UVec2,
    light_count: u32,
}

impl LocalShadowSlots {
    pub fn slot_map(self) -> GpuPtr<u32> {
        self.slot_map
    }
    pub fn slot_state(self) -> GpuPtr<u32> {
        self.slot_state
    }
    pub fn slot_fraction(self) -> GpuPtr<u32> {
        self.slot_fraction
    }
    pub fn slot_rep(self) -> GpuPtr<u32> {
        self.slot_rep
    }
    pub fn half_size(self) -> UVec2 {
        self.half_size
    }
    pub fn size(self) -> UVec2 {
        self.size
    }
    pub fn light_count(self) -> u32 {
        self.light_count
    }
}

/// Temporal producer with ping-pong history and compacted ray queues.
pub struct LocalShadowPass {
    select_shader: gpu::Shader,
    args_shader: gpu::Shader,
    trace_shader: gpu::Shader,
    blur_shader: gpu::Shader,
    slot_map: [gpu::Ptr<u32>; 2],
    slot_state: [gpu::Ptr<u32>; 2],
    /// Single-buffered fractions derived solely from this frame's field.
    slot_fraction: gpu::Ptr<u32>,
    rep_depth: [gpu::Ptr<f32>; 2],
    slot_rep: gpu::Ptr<u32>,
    requests_high: gpu::Ptr<u32>,
    requests_low: gpu::Ptr<u32>,
    counters: gpu::Ptr<LocalShadowCounters>,
    dispatch_args: gpu::Ptr<DispatchIndirectCommand>,
    size: UVec2,
    half: UVec2,
    max_lights: u32,
    in_flight: usize,
    /// Most recently written ping-pong set; accessors return this set.
    pong: usize,
    /// Previous camera and light correspondence enable history reuse.
    /// `None` disables history for the next record.
    prev_frame: Option<Mat4>,
    /// Retained previous light array for motion validation.
    prev_lights: Vec<PointLight>,
    /// Remap scratch (grow-once): current light index → previous index.
    light_remap: Vec<u32>,
    prev_claimed: Vec<bool>,
    /// Frame counter driving the promoted-texel corner rotation.
    frame_index: u32,
    tlas: ShadowTlasBuilder,
}

impl LocalShadowPass {
    pub const TRACE_GROUP_SIZE: u32 = 64;

    pub fn new(
        gpu: &Gpu,
        size: UVec2,
        max_lights: u32,
        instance_capacity: u32,
        in_flight: usize,
    ) -> Self {
        assert!(
            size.x > 0 && size.y > 0,
            "local shadow size must be positive"
        );
        assert!(max_lights > 0, "local shadow max_lights must be positive");
        assert!(in_flight > 0, "local shadow in_flight must be positive");
        let half = half_size(size);
        let texels = texel_capacity(half);
        let requests = texels * u64::from(LOCAL_SHADOW_SLOTS);
        let counters = gpu.alloc_slice::<LocalShadowCounters>(in_flight as u64, Memory::Default);
        // SAFETY: fresh host-visible ring with exactly `in_flight` rows.
        unsafe {
            for slot in 0..in_flight {
                *counters.cpu.add(slot) = LocalShadowCounters::default();
            }
        }
        Self {
            select_shader: gpu.shader_create_compute(
                &asha_assets::load_spv("local_shadow_select"),
                SELECT_GROUP_SIZE,
                SELECT_GROUP_SIZE,
                1,
                "local_shadow_select",
            ),
            args_shader: gpu.shader_create_compute(
                &asha_assets::load_spv("local_shadow_args"),
                1,
                1,
                1,
                "local_shadow_args",
            ),
            trace_shader: gpu.shader_create_compute(
                &asha_assets::load_spv("local_shadow_trace"),
                Self::TRACE_GROUP_SIZE,
                1,
                1,
                "local_shadow_trace",
            ),
            blur_shader: gpu.shader_create_compute(
                &asha_assets::load_spv("local_shadow_blur"),
                SELECT_GROUP_SIZE,
                SELECT_GROUP_SIZE,
                1,
                "local_shadow_blur",
            ),
            slot_map: [
                gpu.alloc_slice(texels, Memory::Gpu),
                gpu.alloc_slice(texels, Memory::Gpu),
            ],
            slot_state: [
                gpu.alloc_slice(requests, Memory::Gpu),
                gpu.alloc_slice(requests, Memory::Gpu),
            ],
            slot_fraction: gpu.alloc_slice(requests, Memory::Gpu),
            rep_depth: [
                gpu.alloc_slice(texels, Memory::Gpu),
                gpu.alloc_slice(texels, Memory::Gpu),
            ],
            slot_rep: gpu.alloc_slice(texels, Memory::Gpu),
            requests_high: gpu.alloc_slice(requests, Memory::Gpu),
            requests_low: gpu.alloc_slice(requests, Memory::Gpu),
            counters,
            dispatch_args: gpu.alloc_slice(1, Memory::Gpu),
            size,
            half,
            max_lights,
            in_flight,
            pong: 0,
            prev_frame: None,
            prev_lights: Vec::with_capacity(max_lights as usize),
            light_remap: Vec::with_capacity(max_lights as usize),
            prev_claimed: Vec::with_capacity(max_lights as usize),
            frame_index: 0,
            tlas: ShadowTlasBuilder::new(instance_capacity),
        }
    }

    /// Resizes buffers after waiting for in-flight GPU use.
    pub fn resize(&mut self, gpu: &Gpu, size: UVec2) -> bool {
        assert!(
            size.x > 0 && size.y > 0,
            "local shadow size must be positive"
        );
        if size == self.size {
            return false;
        }
        let half = half_size(size);
        let texels = texel_capacity(half);
        let requests = texels * u64::from(LOCAL_SHADOW_SLOTS);
        gpu.queue_wait_idle(Queue::Main);
        for i in 0..2 {
            gpu.free(self.slot_map[i]);
            gpu.free(self.slot_state[i]);
            gpu.free(self.rep_depth[i]);
        }
        gpu.free(self.slot_fraction);
        gpu.free(self.slot_rep);
        gpu.free(self.requests_high);
        gpu.free(self.requests_low);
        self.slot_map = [
            gpu.alloc_slice(texels, Memory::Gpu),
            gpu.alloc_slice(texels, Memory::Gpu),
        ];
        self.slot_state = [
            gpu.alloc_slice(requests, Memory::Gpu),
            gpu.alloc_slice(requests, Memory::Gpu),
        ];
        self.slot_fraction = gpu.alloc_slice(requests, Memory::Gpu);
        self.rep_depth = [
            gpu.alloc_slice(texels, Memory::Gpu),
            gpu.alloc_slice(texels, Memory::Gpu),
        ];
        self.slot_rep = gpu.alloc_slice(texels, Memory::Gpu);
        self.requests_high = gpu.alloc_slice(requests, Memory::Gpu);
        self.requests_low = gpu.alloc_slice(requests, Memory::Gpu);
        self.size = size;
        self.half = half;
        self.prev_frame = None;
        true
    }

    pub fn allocated_bytes(&self) -> u64 {
        let texels = texel_capacity(self.half);
        let slots = texels * u64::from(LOCAL_SHADOW_SLOTS);
        texels * 4 * 2 * 2
            + slots * 4 * 2
            + slots * 4
            + texels * 4
            + slots * 4 * 2
            + self.in_flight as u64 * core::mem::size_of::<LocalShadowCounters>() as u64
            + core::mem::size_of::<DispatchIndirectCommand>() as u64
    }

    pub fn half(&self) -> UVec2 {
        self.half
    }

    /// Most recently written buffers (valid after `record`).
    pub fn slot_map_buffer(&self) -> gpu::Ptr<u32> {
        self.slot_map[self.pong]
    }

    pub fn slot_state_buffer(&self) -> gpu::Ptr<u32> {
        self.slot_state[self.pong]
    }

    pub fn slot_rep_buffer(&self) -> gpu::Ptr<u32> {
        self.slot_rep
    }

    /// Returns the single-buffered fraction field after recording.
    pub fn slot_fraction_ptr(&self) -> gpu::Ptr<u32> {
        self.slot_fraction
    }

    /// Discards history; the next record traces every slot.
    pub fn invalidate_history(&mut self) {
        self.prev_frame = None;
    }

    /// Frame N-in_flight's diagnostics. Call before reusing `slot`.
    pub fn take_counters(&self, slot: usize) -> LocalShadowCounters {
        assert!(slot < self.in_flight);
        // SAFETY: the frame-loop gate proves this host-visible slot GPU-idle.
        unsafe { *self.counters.cpu.add(slot) }
    }

    fn counters_gpu(&self, slot: usize) -> GpuPtr<LocalShadowCounters> {
        assert!(slot < self.in_flight);
        self.counters.gpu.offset(slot as i64)
    }

    #[allow(clippy::too_many_arguments)] // Each argument is a real dependency.
    pub fn record(
        &mut self,
        gpu: &Gpu,
        cb: CommandBuffer,
        fa: &mut impl FrameAlloc,
        heap: &HeapSlots,
        scene: &MeshScene,
        instances: MeshInstances<'_>,
        surfaces: &MeshSurfaceTargets,
        view: MeshRasterView,
        lights: &[PointLight],
        origin_bias: f32,
        destination_bias: f32,
        wrap_w: f32,
        temporal: LocalShadowTemporal,
        depth_slot: SampledSlot,
        counter_slot: usize,
    ) -> (LocalShadowSlots, ShadowTlasStats) {
        assert_eq!(
            surfaces.size(),
            self.size,
            "local shadow buffers and mesh surfaces must have the same size"
        );
        let light_count = u32::try_from(lights.len()).expect("light count fits u32");
        assert!(
            light_count <= self.max_lights,
            "local shadow light count {light_count} exceeds capacity {}",
            self.max_lights
        );
        assert!(
            light_count <= abi_light::LOCAL_SHADOW_MAX_LIGHT_ID + 1,
            "light ids must fit the 8-bit slot lanes"
        );
        assert!(
            origin_bias.is_finite() && origin_bias >= 0.0,
            "local shadow origin bias must be finite and nonnegative"
        );
        assert!(
            destination_bias.is_finite() && destination_bias >= 0.0,
            "local shadow destination bias must be finite and nonnegative"
        );
        assert!(wrap_w.is_finite(), "local shadow wrap must be finite");
        assert!(
            temporal.validate_thickness.is_finite() && temporal.validate_thickness > 0.0,
            "local shadow validation thickness must be finite and positive"
        );
        assert!(
            temporal.near_plane.is_finite() && temporal.near_plane > 0.0,
            "local shadow near plane must be finite and positive"
        );
        assert!(
            temporal.light_epsilon.is_finite() && temporal.light_epsilon >= 0.0,
            "local shadow light epsilon must be finite and nonnegative"
        );
        if let Some(contact) = &temporal.contact {
            assert!(
                depth_march_config_valid(contact),
                "invalid contact-march config"
            );
            assert!(
                temporal.contact_distance.is_finite() && temporal.contact_distance > 0.0,
                "contact march needs a finite positive reach"
            );
        }
        let det = view.world_to_clip.determinant();
        assert!(
            view.world_to_clip.is_finite() && det.is_finite() && det.abs() > 1.0e-8,
            "local shadows reconstruct world position: world_to_clip must be finite and invertible"
        );
        assert!(
            temporal.source_radius.is_finite() && temporal.source_radius >= 0.0,
            "local shadow source radius must be finite and nonnegative"
        );
        let write = self.pong ^ 1;
        let read = self.pong;
        let history = self.prev_frame;

        if history.is_some() {
            match_lights(
                &self.prev_lights,
                lights,
                temporal.light_epsilon,
                &mut self.light_remap,
                &mut self.prev_claimed,
            );
        } else {
            self.light_remap.clear();
            self.light_remap
                .extend(std::iter::repeat_n(LOCAL_SHADOW_SLOT_EMPTY, lights.len()));
        }

        let slots = LocalShadowSlots {
            slot_map: self.slot_map[write].gpu,
            slot_state: self.slot_state[write].gpu,
            slot_fraction: self.slot_fraction.gpu,
            slot_rep: self.slot_rep.gpu,
            half_size: self.half,
            size: self.size,
            light_count,
        };
        let (world, stats) = self.tlas.build_instances(fa, scene, instances);
        let texel_count = self.half.x * self.half.y;
        let request_capacity = texel_count * LOCAL_SHADOW_SLOTS;
        assert!(counter_slot < self.in_flight);
        // SAFETY: caller rotates this slot on the same in-flight gate as the
        // frame allocator. Preserve the previous row for `take_counters`
        // before this call; this initializes the current frame's atomics.
        unsafe {
            *self.counters.cpu.add(counter_slot) = LocalShadowCounters {
                texel_count,
                ..Default::default()
            };
        }

        let counters = self.counters_gpu(counter_slot);
        let lights_gpu = fa.frame_alloc_slice(lights);
        let prev_lights_gpu = if history.is_some() && !self.prev_lights.is_empty() {
            fa.frame_alloc_slice(&self.prev_lights)
        } else {
            lights_gpu
        };
        let light_remap_gpu = fa.frame_alloc_slice(&self.light_remap);
        let data = fa.frame_alloc(LocalShadowData {
            clip_to_world: view.world_to_clip.inverse(),
            world_to_clip: view.world_to_clip,
            prev_world_to_clip: history.unwrap_or(Mat4::IDENTITY),
            world,
            lights: lights_gpu,
            prev_lights: prev_lights_gpu,
            slot_map: self.slot_map[write].gpu,
            slot_map_prev: self.slot_map[read].gpu,
            slot_state: self.slot_state[write].gpu,
            slot_state_prev: self.slot_state[read].gpu,
            slot_fraction: self.slot_fraction.gpu,
            slot_rep: self.slot_rep.gpu,
            rep_depth: self.rep_depth[write].gpu,
            rep_depth_prev: self.rep_depth[read].gpu,
            requests_high: self.requests_high.gpu,
            requests_low: self.requests_low.gpu,
            counters,
            depth_texture_id: depth_slot.index(),
            surface_material_texture_id: surfaces.material_slot().index(),
            surface_normal_texture_id: surfaces.normal_slot().index(),
            history_valid: u32::from(history.is_some()),
            screen_size: self.size.to_array(),
            half_size: self.half.to_array(),
            light_count,
            origin_bias,
            destination_bias,
            wrap_w,
            request_capacity,
            refresh_interval: temporal.refresh_interval,
            validate_thickness: temporal.validate_thickness,
            near_plane: temporal.near_plane,
            light_epsilon: temporal.light_epsilon,
            contact_distance: temporal.contact_distance,
            contact: temporal.contact.unwrap_or_default(),
            ray_budget: temporal.ray_budget,
            edge_promotion: u32::from(temporal.edge_promotion),
            frame_index: self.frame_index,
            occluded_refresh: temporal.occluded_refresh,
            source_radius: temporal.source_radius,
            proj_scale: view.world_to_clip.row(1).truncate().length(),
            light_remap: light_remap_gpu,
            _pad0: [0; 2],
        });

        heap.bind(gpu, cb);
        gpu.cmd_barrier(cb, Stage::All, Stage::Compute, HazardFlags::empty());
        gpu.cmd_set_compute_shader(cb, self.select_shader);
        gpu.cmd_dispatch(
            cb,
            data,
            self.half.x.div_ceil(SELECT_GROUP_SIZE),
            self.half.y.div_ceil(SELECT_GROUP_SIZE),
            1,
        );
        gpu.cmd_barrier(
            cb,
            Stage::Compute,
            Stage::Compute,
            HazardFlags::SHADER_BUFFER,
        );
        let args_data = fa.frame_alloc(LocalShadowArgsData {
            counters,
            dispatch_args: self.dispatch_args.cast::<u32>().gpu,
            queue_capacity: request_capacity,
            group_size: Self::TRACE_GROUP_SIZE,
            ray_budget: temporal.ray_budget,
            _pad0: 0,
        });
        gpu.cmd_set_compute_shader(cb, self.args_shader);
        gpu.cmd_dispatch(cb, args_data, 1, 1, 1);
        gpu.cmd_barrier(
            cb,
            Stage::Compute,
            Stage::All,
            HazardFlags::DRAW_ARGUMENTS | HazardFlags::SHADER_BUFFER,
        );
        gpu.cmd_set_compute_shader(cb, self.trace_shader);
        gpu.cmd_dispatch_indirect(cb, data, self.dispatch_args);
        if temporal.source_radius > 0.0 {
            gpu.cmd_barrier(
                cb,
                Stage::Compute,
                Stage::Compute,
                HazardFlags::SHADER_BUFFER,
            );
            gpu.cmd_set_compute_shader(cb, self.blur_shader);
            gpu.cmd_dispatch(
                cb,
                data,
                self.half.x.div_ceil(SELECT_GROUP_SIZE),
                self.half.y.div_ceil(SELECT_GROUP_SIZE),
                1,
            );
        }

        self.pong = write;
        self.frame_index = self.frame_index.wrapping_add(1);
        self.prev_frame = Some(view.world_to_clip);
        self.prev_lights.clear();
        self.prev_lights.extend_from_slice(lights);
        (slots, stats)
    }
}

impl Pass for LocalShadowPass {
    const NAME: &'static str = "local_shadow";

    fn free(self, gpu: &Gpu) {
        gpu.shader_destroy(self.select_shader);
        gpu.shader_destroy(self.args_shader);
        gpu.shader_destroy(self.trace_shader);
        gpu.shader_destroy(self.blur_shader);
        for i in 0..2 {
            gpu.free(self.slot_map[i]);
            gpu.free(self.slot_state[i]);
            gpu.free(self.rep_depth[i]);
        }
        gpu.free(self.slot_fraction);
        gpu.free(self.slot_rep);
        gpu.free(self.requests_high);
        gpu.free(self.requests_low);
        gpu.free(self.counters);
        gpu.free(self.dispatch_args);
    }
}

/// Stateless direct trace over this frame's TLAS and depth.
///
/// It has no history, queues, budget, or frame-to-frame state, but produces
/// the same [`LocalShadowSlots`] contract as the temporal producer.
pub struct LocalShadowDirectPass {
    select_shader: gpu::Shader,
    args_shader: gpu::Shader,
    trace_shader: gpu::Shader,
    blur_shader: gpu::Shader,
    slot_map: gpu::Ptr<u32>,
    slot_state: gpu::Ptr<u32>,
    slot_fraction: gpu::Ptr<u32>,
    slot_rep: gpu::Ptr<u32>,
    rep_depth: gpu::Ptr<f32>,
    /// Per-frame dense work list of `(texel, slot)` request IDs.
    requests: gpu::Ptr<u32>,
    counters: gpu::Ptr<LocalShadowCounters>,
    dispatch_args: gpu::Ptr<DispatchIndirectCommand>,
    size: UVec2,
    half: UVec2,
    max_lights: u32,
    in_flight: usize,
    tlas: ShadowTlasBuilder,
}

impl LocalShadowDirectPass {
    pub const TRACE_GROUP_SIZE: u32 = 64;

    pub fn new(
        gpu: &Gpu,
        size: UVec2,
        max_lights: u32,
        instance_capacity: u32,
        in_flight: usize,
    ) -> Self {
        assert!(
            size.x > 0 && size.y > 0,
            "local shadow size must be positive"
        );
        assert!(max_lights > 0, "local shadow max_lights must be positive");
        assert!(in_flight > 0, "local shadow in_flight must be positive");
        let half = half_size(size);
        let texels = texel_capacity(half);
        let slots = texels * u64::from(LOCAL_SHADOW_SLOTS);
        let counters = gpu.alloc_slice::<LocalShadowCounters>(in_flight as u64, Memory::Default);
        // SAFETY: fresh host-visible ring with exactly `in_flight` rows.
        unsafe {
            for slot in 0..in_flight {
                *counters.cpu.add(slot) = LocalShadowCounters::default();
            }
        }
        Self {
            select_shader: gpu.shader_create_compute(
                &asha_assets::load_spv("local_shadow_select_direct"),
                SELECT_GROUP_SIZE,
                SELECT_GROUP_SIZE,
                1,
                "local_shadow_select_direct",
            ),
            args_shader: gpu.shader_create_compute(
                &asha_assets::load_spv("local_shadow_args"),
                1,
                1,
                1,
                "local_shadow_args",
            ),
            trace_shader: gpu.shader_create_compute(
                &asha_assets::load_spv("local_shadow_trace_direct"),
                Self::TRACE_GROUP_SIZE,
                1,
                1,
                "local_shadow_trace_direct",
            ),
            blur_shader: gpu.shader_create_compute(
                &asha_assets::load_spv("local_shadow_blur"),
                SELECT_GROUP_SIZE,
                SELECT_GROUP_SIZE,
                1,
                "local_shadow_blur",
            ),
            slot_map: gpu.alloc_slice(texels, Memory::Gpu),
            slot_state: gpu.alloc_slice(slots, Memory::Gpu),
            slot_fraction: gpu.alloc_slice(slots, Memory::Gpu),
            slot_rep: gpu.alloc_slice(texels, Memory::Gpu),
            rep_depth: gpu.alloc_slice(texels, Memory::Gpu),
            requests: gpu.alloc_slice(slots, Memory::Gpu),
            counters,
            dispatch_args: gpu.alloc_slice(1, Memory::Gpu),
            size,
            half,
            max_lights,
            in_flight,
            tlas: ShadowTlasBuilder::new(instance_capacity),
        }
    }

    /// Resizes buffers after waiting for in-flight GPU use.
    pub fn resize(&mut self, gpu: &Gpu, size: UVec2) -> bool {
        assert!(
            size.x > 0 && size.y > 0,
            "local shadow size must be positive"
        );
        if size == self.size {
            return false;
        }
        let half = half_size(size);
        let texels = texel_capacity(half);
        let slots = texels * u64::from(LOCAL_SHADOW_SLOTS);
        gpu.queue_wait_idle(Queue::Main);
        gpu.free(self.slot_map);
        gpu.free(self.slot_state);
        gpu.free(self.slot_fraction);
        gpu.free(self.slot_rep);
        gpu.free(self.rep_depth);
        gpu.free(self.requests);
        self.slot_map = gpu.alloc_slice(texels, Memory::Gpu);
        self.slot_state = gpu.alloc_slice(slots, Memory::Gpu);
        self.slot_fraction = gpu.alloc_slice(slots, Memory::Gpu);
        self.slot_rep = gpu.alloc_slice(texels, Memory::Gpu);
        self.rep_depth = gpu.alloc_slice(texels, Memory::Gpu);
        self.requests = gpu.alloc_slice(slots, Memory::Gpu);
        self.size = size;
        self.half = half;
        true
    }

    pub fn half(&self) -> UVec2 {
        self.half
    }

    /// Returns the fraction field written by the latest record.
    pub fn slot_fraction_ptr(&self) -> gpu::Ptr<u32> {
        self.slot_fraction
    }

    /// Returns diagnostics for an idle in-flight slot.
    pub fn take_counters(&self, slot: usize) -> LocalShadowCounters {
        assert!(slot < self.in_flight);
        // SAFETY: the frame-loop gate proves this host-visible slot GPU-idle.
        unsafe { *self.counters.cpu.add(slot) }
    }

    #[allow(clippy::too_many_arguments)] // Each argument is a real dependency.
    pub fn record(
        &mut self,
        gpu: &Gpu,
        cb: CommandBuffer,
        fa: &mut impl FrameAlloc,
        heap: &HeapSlots,
        scene: &MeshScene,
        instances: MeshInstances<'_>,
        surfaces: &MeshSurfaceTargets,
        view: MeshRasterView,
        lights: &[PointLight],
        origin_bias: f32,
        destination_bias: f32,
        wrap_w: f32,
        source_radius: f32,
        near_plane: f32,
        depth_slot: SampledSlot,
        counter_slot: usize,
    ) -> (LocalShadowSlots, ShadowTlasStats) {
        assert_eq!(
            surfaces.size(),
            self.size,
            "local shadow buffers and mesh surfaces must have the same size"
        );
        let light_count = u32::try_from(lights.len()).expect("light count fits u32");
        assert!(
            light_count <= self.max_lights,
            "local shadow light count {light_count} exceeds capacity {}",
            self.max_lights
        );
        assert!(
            light_count <= abi_light::LOCAL_SHADOW_MAX_LIGHT_ID + 1,
            "light ids must fit the 8-bit slot lanes"
        );
        assert!(
            origin_bias.is_finite() && origin_bias >= 0.0,
            "local shadow origin bias must be finite and nonnegative"
        );
        assert!(
            destination_bias.is_finite() && destination_bias >= 0.0,
            "local shadow destination bias must be finite and nonnegative"
        );
        assert!(wrap_w.is_finite(), "local shadow wrap must be finite");
        assert!(
            source_radius.is_finite() && source_radius >= 0.0,
            "local shadow source radius must be finite and nonnegative"
        );
        assert!(
            near_plane.is_finite() && near_plane > 0.0,
            "local shadow near plane must be finite and positive"
        );
        let det = view.world_to_clip.determinant();
        assert!(
            view.world_to_clip.is_finite() && det.is_finite() && det.abs() > 1.0e-8,
            "local shadows reconstruct world position: world_to_clip must be finite and invertible"
        );

        let slots = LocalShadowSlots {
            slot_map: self.slot_map.gpu,
            slot_state: self.slot_state.gpu,
            slot_fraction: self.slot_fraction.gpu,
            slot_rep: self.slot_rep.gpu,
            half_size: self.half,
            size: self.size,
            light_count,
        };
        let (world, stats) = self.tlas.build_instances(fa, scene, instances);
        let texel_count = self.half.x * self.half.y;
        let request_capacity = texel_count * LOCAL_SHADOW_SLOTS;
        assert!(counter_slot < self.in_flight);
        // SAFETY: caller rotates this slot on the same in-flight gate as the
        // frame allocator; this initializes the current frame's atomics.
        unsafe {
            *self.counters.cpu.add(counter_slot) = LocalShadowCounters {
                texel_count,
                ..Default::default()
            };
        }
        let counters = {
            assert!(counter_slot < self.in_flight);
            self.counters.gpu.offset(counter_slot as i64)
        };
        let lights_gpu = fa.frame_alloc_slice(lights);
        // Direct tracing uses zero-value temporal state: no history, no
        // remap, no low queue, unlimited high-queue work, and current buffers
        // aliased as harmless previous inputs because history is disabled.
        let data = fa.frame_alloc(LocalShadowData {
            clip_to_world: view.world_to_clip.inverse(),
            world_to_clip: view.world_to_clip,
            prev_world_to_clip: Mat4::IDENTITY,
            world,
            lights: lights_gpu,
            prev_lights: lights_gpu,
            slot_map: self.slot_map.gpu,
            slot_map_prev: self.slot_map.gpu,
            slot_state: self.slot_state.gpu,
            slot_state_prev: self.slot_state.gpu,
            slot_fraction: self.slot_fraction.gpu,
            slot_rep: self.slot_rep.gpu,
            rep_depth: self.rep_depth.gpu,
            rep_depth_prev: self.rep_depth.gpu,
            requests_high: self.requests.gpu,
            requests_low: GpuPtr::null(),
            counters,
            depth_texture_id: depth_slot.index(),
            surface_material_texture_id: surfaces.material_slot().index(),
            surface_normal_texture_id: surfaces.normal_slot().index(),
            history_valid: 0,
            screen_size: self.size.to_array(),
            half_size: self.half.to_array(),
            light_count,
            origin_bias,
            destination_bias,
            wrap_w,
            request_capacity,
            refresh_interval: 0,
            validate_thickness: 1.0,
            near_plane,
            light_epsilon: 0.0,
            contact_distance: 0.0,
            contact: DepthMarchConfig::default(),
            ray_budget: 0,
            edge_promotion: 0,
            frame_index: 0,
            occluded_refresh: 0,
            source_radius,
            proj_scale: view.world_to_clip.row(1).truncate().length(),
            light_remap: GpuPtr::null(),
            _pad0: [0; 2],
        });

        heap.bind(gpu, cb);
        gpu.cmd_barrier(cb, Stage::All, Stage::Compute, HazardFlags::empty());
        gpu.cmd_set_compute_shader(cb, self.select_shader);
        gpu.cmd_dispatch(
            cb,
            data,
            self.half.x.div_ceil(SELECT_GROUP_SIZE),
            self.half.y.div_ceil(SELECT_GROUP_SIZE),
            1,
        );
        gpu.cmd_barrier(
            cb,
            Stage::Compute,
            Stage::Compute,
            HazardFlags::SHADER_BUFFER,
        );
        let args_data = fa.frame_alloc(LocalShadowArgsData {
            counters,
            dispatch_args: self.dispatch_args.cast::<u32>().gpu,
            queue_capacity: request_capacity,
            group_size: Self::TRACE_GROUP_SIZE,
            ray_budget: 0,
            _pad0: 0,
        });
        gpu.cmd_set_compute_shader(cb, self.args_shader);
        gpu.cmd_dispatch(cb, args_data, 1, 1, 1);
        gpu.cmd_barrier(
            cb,
            Stage::Compute,
            Stage::All,
            HazardFlags::DRAW_ARGUMENTS | HazardFlags::SHADER_BUFFER,
        );
        gpu.cmd_set_compute_shader(cb, self.trace_shader);
        gpu.cmd_dispatch_indirect(cb, data, self.dispatch_args);
        if source_radius > 0.0 {
            gpu.cmd_barrier(
                cb,
                Stage::Compute,
                Stage::Compute,
                HazardFlags::SHADER_BUFFER,
            );
            gpu.cmd_set_compute_shader(cb, self.blur_shader);
            gpu.cmd_dispatch(
                cb,
                data,
                self.half.x.div_ceil(SELECT_GROUP_SIZE),
                self.half.y.div_ceil(SELECT_GROUP_SIZE),
                1,
            );
        }
        (slots, stats)
    }
}

impl Pass for LocalShadowDirectPass {
    const NAME: &'static str = "local_shadow_direct";

    fn free(self, gpu: &Gpu) {
        gpu.shader_destroy(self.select_shader);
        gpu.shader_destroy(self.args_shader);
        gpu.shader_destroy(self.trace_shader);
        gpu.shader_destroy(self.blur_shader);
        gpu.free(self.slot_map);
        gpu.free(self.slot_state);
        gpu.free(self.slot_fraction);
        gpu.free(self.slot_rep);
        gpu.free(self.rep_depth);
        gpu.free(self.requests);
        gpu.free(self.counters);
        gpu.free(self.dispatch_args);
    }
}

/// Matches current lights to nearest unclaimed predecessors within `epsilon`.
///
/// Reorders and smooth motion preserve history; teleports and new lights map
/// to [`LOCAL_SHADOW_SLOT_EMPTY`]. Each predecessor can be claimed once.
fn match_lights(
    prev: &[PointLight],
    current: &[PointLight],
    epsilon: f32,
    remap: &mut Vec<u32>,
    claimed: &mut Vec<bool>,
) {
    remap.clear();
    claimed.clear();
    claimed.resize(prev.len(), false);
    let epsilon2 = epsilon * epsilon;
    for light in current {
        let position = abi_core::glam::Vec3::from_array(light.position);
        let mut best = LOCAL_SHADOW_SLOT_EMPTY;
        let mut best_distance2 = f32::INFINITY;
        for (j, prev_light) in prev.iter().enumerate() {
            if claimed[j] {
                continue;
            }
            let distance2 =
                (abi_core::glam::Vec3::from_array(prev_light.position) - position).length_squared();
            if distance2 <= epsilon2 && distance2 < best_distance2 {
                best = j as u32;
                best_distance2 = distance2;
            }
        }
        if let Some(claim) = claimed.get_mut(best as usize) {
            *claim = true;
        }
        remap.push(best);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn light(x: f32, z: f32) -> PointLight {
        PointLight {
            position: [x, 2.0, z],
            radius: 14.0,
            color: [1.0, 1.0, 1.0],
            intensity: 5.0,
        }
    }

    /// Slot history follows light identity across reorder and membership changes.
    #[test]
    fn light_matching_survives_reorder_insert_remove_and_motion() {
        let mut remap = Vec::new();
        let mut claimed = Vec::new();
        let a = light(0.0, 0.0);
        let b = light(10.0, 0.0);
        let c = light(0.0, 10.0);

        match_lights(&[a, b, c], &[c, a, b], 1.0, &mut remap, &mut claimed);
        assert_eq!(remap, [2, 0, 1]);

        let a_moved = light(0.4, 0.0);
        match_lights(&[a, b], &[a_moved, b], 1.0, &mut remap, &mut claimed);
        assert_eq!(remap, [0, 1]);

        let a_teleported = light(50.0, 0.0);
        match_lights(&[a, b], &[a_teleported, b], 1.0, &mut remap, &mut claimed);
        assert_eq!(remap, [LOCAL_SHADOW_SLOT_EMPTY, 1]);

        match_lights(&[a, b], &[a, c, b], 1.0, &mut remap, &mut claimed);
        assert_eq!(remap, [0, LOCAL_SHADOW_SLOT_EMPTY, 1]);

        match_lights(&[a, b, c], &[c, a], 1.0, &mut remap, &mut claimed);
        assert_eq!(remap, [2, 0]);

        let a_twin = light(0.3, 0.0);
        let a_near = light(0.1, 0.0);
        match_lights(&[a], &[a_twin, a_near], 1.0, &mut remap, &mut claimed);
        assert_eq!(remap, [0, LOCAL_SHADOW_SLOT_EMPTY]);
    }
}

/// Half resolution rounds up so odd right/bottom edges keep coverage.
fn half_size(size: UVec2) -> UVec2 {
    UVec2::new(size.x.div_ceil(2), size.y.div_ceil(2))
}

fn texel_capacity(half: UVec2) -> u64 {
    u64::from(half.x)
        .checked_mul(u64::from(half.y))
        .expect("local shadow texel capacity overflow")
}
