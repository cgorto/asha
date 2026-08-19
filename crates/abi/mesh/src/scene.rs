//! Scene and draw plumbing for the one geometry path: meshlets, cluster
//! culling, and multi-draw indirect. Layouts cross to the shader through
//! buffer-device-address loads; strides and offsets are part of the ABI.

pub use abi_core::DrawIndexedIndirectCommand;

use crate::{DeformerStack, DualQuat, JointWeights};
use abi_core::{GpuPtr, gpu_data};
use glam::{Mat4, Vec3, Vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

/// Vulkan's `VkDrawIndirectCommand`. Mesh batches use the non-indexed form:
/// every visible cluster has its own meshlet index range, so the vertex stage
/// pulls the real index from [`MeshFrameData::index_data`] instead of asking
/// fixed-function indexing to choose one range for a whole batch.
#[gpu_data]
pub struct DrawIndirectCommand {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}

const _: () = assert!(core::mem::size_of::<DrawIndirectCommand>() == 16);
const _: () = assert!(core::mem::align_of::<DrawIndirectCommand>() == 4);
const _: () = assert!(core::mem::offset_of!(DrawIndirectCommand, vertex_count) == 0);
const _: () = assert!(core::mem::offset_of!(DrawIndirectCommand, instance_count) == 4);
const _: () = assert!(core::mem::offset_of!(DrawIndirectCommand, first_vertex) == 8);
const _: () = assert!(core::mem::offset_of!(DrawIndirectCommand, first_instance) == 12);

pub const ALPHA_MODE_OPAQUE: u32 = 0;
pub const ALPHA_MODE_MASK: u32 = 1;
pub const ALPHA_MODE_BLEND: u32 = 2;

/// Bit flags consumed by mesh shaders.
pub const MESH_FLAG_SKINNED: u32 = 0x1;
pub const MESH_FLAG_HIDDEN: u32 = 0x2;
pub const MESH_FLAG_NO_SHADOW: u32 = 0x4;
pub const MESH_FLAG_STATIC: u32 = 0x8;
pub const MESH_FLAG_NO_ANALYTICAL_SHADOW: u32 = 0x10;
/// Excludes presentation geometry from linework and neighboring halos.
pub const MESH_FLAG_NO_LINEWORK: u32 = 0x20;
/// The instance moved or deforms this frame. Derived, never authored: the
/// bevy extract sets it from transform change detection + deformer
/// presence, and MeshScene's direct movers (stage_world / set_world) set it
/// sticky. Shadow consumers shorten blind-occlusion trust behind it — a
/// hit on something that moves cannot be trusted to keep standing there.
pub const MESH_FLAG_DYNAMIC: u32 = 0x40;

/// Slim mesh lighting payload. `sun_direction` is the L vector, toward the
/// sun, and `sun_tint` is the transmitted sun color already scaled by the
/// host dial.
#[gpu_data]
pub struct MeshShadeLighting {
    pub sun_direction: [f32; 3],
    /// Diffuse wrap dial; zero is exactly Lambert.
    pub wrap_w: f32,
    pub sun_tint: [f32; 3],
    pub _pad1: u32,
    pub sky_ambient: [f32; 3],
    pub _pad2: f32,
    pub ground_ambient: [f32; 3],
    pub _pad3: f32,
}

impl MeshShadeLighting {
    pub const fn zeroed() -> Self {
        Self {
            sun_direction: [0.0; 3],
            wrap_w: 0.0,
            sun_tint: [0.0; 3],
            _pad1: 0,
            sky_ambient: [0.0; 3],
            _pad2: 0.0,
            ground_ambient: [0.0; 3],
            _pad3: 0.0,
        }
    }
}

const _: () = assert!(core::mem::size_of::<MeshShadeLighting>() == 64);
const _: () = assert!(core::mem::align_of::<MeshShadeLighting>() == 4);
const _: () = assert!(core::mem::offset_of!(MeshShadeLighting, sun_direction) == 0);
const _: () = assert!(core::mem::offset_of!(MeshShadeLighting, wrap_w) == 12);
const _: () = assert!(core::mem::offset_of!(MeshShadeLighting, sun_tint) == 16);
const _: () = assert!(core::mem::offset_of!(MeshShadeLighting, _pad1) == 28);
const _: () = assert!(core::mem::offset_of!(MeshShadeLighting, sky_ambient) == 32);
const _: () = assert!(core::mem::offset_of!(MeshShadeLighting, ground_ambient) == 48);

/// Vertex stream table; all streams share the global mesh index space.
#[gpu_data]
pub struct MeshTableEntry {
    /// 16-byte stride; `w = 0`.
    pub positions: GpuPtr<[f32; 4]>,
    /// 16-byte stride; `w = 0`.
    pub normals: GpuPtr<[f32; 4]>,
    /// Tight 8-byte stride.
    pub uvs: GpuPtr<[f32; 2]>,
    /// Null selects the shader ddx/ddy TBN fallback; `w` is the bitangent sign.
    pub tangents: GpuPtr<[f32; 4]>,
    /// Null means static; otherwise points at immutable vertex-parallel weights.
    pub joint_weights: GpuPtr<JointWeights>,
    /// Null means untinted: the vertex stage substitutes `Vec4::ONE` and the
    /// mesh takes exactly its colourless path. Non-null is a MULTIPLICATIVE
    /// tint over `MeshInstance::instance_color`, never a replacement.
    pub colors: GpuPtr<[f32; 4]>,
}

const _: () = assert!(core::mem::size_of::<MeshTableEntry>() == 48);
const _: () = assert!(core::mem::align_of::<MeshTableEntry>() == 4);
const _: () = assert!(core::mem::offset_of!(MeshTableEntry, positions) == 0);
const _: () = assert!(core::mem::offset_of!(MeshTableEntry, normals) == 8);
const _: () = assert!(core::mem::offset_of!(MeshTableEntry, uvs) == 16);
const _: () = assert!(core::mem::offset_of!(MeshTableEntry, tangents) == 24);
const _: () = assert!(core::mem::offset_of!(MeshTableEntry, joint_weights) == 32);
const _: () = assert!(core::mem::offset_of!(MeshTableEntry, colors) == 40);

#[gpu_data]
pub struct MeshData {
    pub idx_count: u32,
    pub first_index: u32,
    pub meshlet_offset: u32,
    pub meshlet_count: u32,
    /// Maximum `Meshlet::tri_count * 3` for this mesh. A batched non-indexed
    /// draw uses this many vertices for every cluster; shorter meshlets pad
    /// their whole-triangle tail with degenerate positions in the shader.
    pub cluster_vertex_count: u32,
}

const _: () = assert!(core::mem::size_of::<MeshData>() == 20);
const _: () = assert!(core::mem::align_of::<MeshData>() == 4);
const _: () = assert!(core::mem::offset_of!(MeshData, idx_count) == 0);
const _: () = assert!(core::mem::offset_of!(MeshData, first_index) == 4);
const _: () = assert!(core::mem::offset_of!(MeshData, meshlet_offset) == 8);
const _: () = assert!(core::mem::offset_of!(MeshData, meshlet_count) == 12);
const _: () = assert!(core::mem::offset_of!(MeshData, cluster_vertex_count) == 16);

#[gpu_data]
pub struct MeshBounds {
    pub aabb_min: [f32; 3],
    pub _pad0: f32,
    pub aabb_max: [f32; 3],
    pub _pad1: f32,
}

const _: () = assert!(core::mem::size_of::<MeshBounds>() == 32);
const _: () = assert!(core::mem::align_of::<MeshBounds>() == 4);
const _: () = assert!(core::mem::offset_of!(MeshBounds, _pad0) == 12);
const _: () = assert!(core::mem::offset_of!(MeshBounds, aabb_max) == 16);
const _: () = assert!(core::mem::offset_of!(MeshBounds, _pad1) == 28);

#[gpu_data]
pub struct DrawTransform {
    pub model_to_world: Mat4,
    pub model_to_world_normal: Mat4,
}

const _: () = assert!(core::mem::size_of::<DrawTransform>() == 128);
const _: () = assert!(core::mem::align_of::<DrawTransform>() == 16);
const _: () = assert!(core::mem::offset_of!(DrawTransform, model_to_world_normal) == 64);

/// One transformable mesh object. It selects a batch rather than embedding a
/// mesh/material draw command; color and outline identity deliberately do not
/// participate in batch identity.
#[gpu_data]
pub struct MeshInstance {
    pub batch_index: u32,
    pub transform_index: u32,
    pub flags: u32,
    /// Zero disables outlines; other values select an outline group.
    pub outline_group: u32,
    /// Multiplicative visual modulation. White is neutral.
    pub instance_color: [f32; 4],
    /// Null means static; otherwise points at this instance's joint palette
    /// of unit dual quaternions.
    pub joint_transforms: GpuPtr<DualQuat>,
    /// Slot index into the extracted deformer array, not a pointer.
    /// 0 = undeformed (ZII).
    pub deformer_slot: u32,
    /// Conservative world-space cull-bound expansion.
    pub bounds_dilation: f32,
}

const _: () = assert!(core::mem::size_of::<MeshInstance>() == 48);
const _: () = assert!(core::mem::align_of::<MeshInstance>() == 4);
const _: () = assert!(core::mem::offset_of!(MeshInstance, batch_index) == 0);
const _: () = assert!(core::mem::offset_of!(MeshInstance, transform_index) == 4);
const _: () = assert!(core::mem::offset_of!(MeshInstance, flags) == 8);
const _: () = assert!(core::mem::offset_of!(MeshInstance, outline_group) == 12);
const _: () = assert!(core::mem::offset_of!(MeshInstance, instance_color) == 16);
const _: () = assert!(core::mem::offset_of!(MeshInstance, joint_transforms) == 32);
const _: () = assert!(core::mem::offset_of!(MeshInstance, deformer_slot) == 40);
const _: () = assert!(core::mem::offset_of!(MeshInstance, bounds_dilation) == 44);

/// Static draw-class metadata. Its `[cluster_base, cluster_base +
/// cluster_capacity)` range is exclusive to this batch for one cull.
#[gpu_data]
pub struct MeshBatch {
    pub mesh_index: u32,
    pub material_index: u32,
    pub cluster_base: u32,
    pub cluster_capacity: u32,
}

const _: () = assert!(core::mem::size_of::<MeshBatch>() == 16);
const _: () = assert!(core::mem::align_of::<MeshBatch>() == 4);
const _: () = assert!(core::mem::offset_of!(MeshBatch, mesh_index) == 0);
const _: () = assert!(core::mem::offset_of!(MeshBatch, material_index) == 4);
const _: () = assert!(core::mem::offset_of!(MeshBatch, cluster_base) == 8);
const _: () = assert!(core::mem::offset_of!(MeshBatch, cluster_capacity) == 12);

/// One visible `(MeshInstance, Meshlet)` pair, compacted by the cull stage.
#[gpu_data]
pub struct ClusterInstance {
    pub instance_id: u32,
    pub meshlet_index: u32,
}

const _: () = assert!(core::mem::size_of::<ClusterInstance>() == 8);
const _: () = assert!(core::mem::align_of::<ClusterInstance>() == 4);
const _: () = assert!(core::mem::offset_of!(ClusterInstance, instance_id) == 0);
const _: () = assert!(core::mem::offset_of!(ClusterInstance, meshlet_index) == 4);

/// Per-frame draw payload shared by the mesh vertex and fragment stages.
/// The host may pass the same allocation as both graphics push slots. The
/// indirect push slot names a batch; `InstanceIndex` then names a compacted
/// cluster, including the command's `first_instance` base.
#[gpu_data]
pub struct MeshFrameData {
    pub world_to_clip: Mat4,
    pub mesh_table: GpuPtr<MeshTableEntry>,
    pub transforms: GpuPtr<DrawTransform>,
    pub materials: GpuPtr<MaterialEntry>,
    pub deformers: GpuPtr<DeformerStack>,
    pub batches: GpuPtr<MeshBatch>,
    pub instances: GpuPtr<MeshInstance>,
    pub clusters: GpuPtr<ClusterInstance>,
    pub meshlets: GpuPtr<Meshlet>,
    /// Global mesh index buffer, read by vertex pulling.
    pub index_data: GpuPtr<u32>,
    pub lighting: MeshShadeLighting,
    /// Null selects the ungated path.
    pub light_field: GpuPtr<f32>,
    pub light_field_dims: [u32; 2],
    pub light_field_cell_size: f32,
    /// 0 = ungated, 1 = full field sample.
    pub light_field_gate: f32,
    /// Host frame time in seconds. Crosses for bespoke shader-group
    /// animation; the standard pair never reads it (ungrouped recorders
    /// pass 0).
    pub time: f32,
    pub _pad4: u32,
    /// Camera world position for view-dependent mesh shading.
    pub eye: [f32; 3],
    /// Sampler used when a material leaves `ramp_sampler` at its default 0.
    pub ramp_default_sampler: u32,
    pub _pad5: [u32; 2],
}

const _: () = assert!(core::mem::size_of::<MeshFrameData>() == 256);
const _: () = assert!(core::mem::align_of::<MeshFrameData>() == 16);
const _: () = assert!(core::mem::offset_of!(MeshFrameData, world_to_clip) == 0);
const _: () = assert!(core::mem::offset_of!(MeshFrameData, mesh_table) == 64);
const _: () = assert!(core::mem::offset_of!(MeshFrameData, transforms) == 72);
const _: () = assert!(core::mem::offset_of!(MeshFrameData, materials) == 80);
const _: () = assert!(core::mem::offset_of!(MeshFrameData, deformers) == 88);
const _: () = assert!(core::mem::offset_of!(MeshFrameData, batches) == 96);
const _: () = assert!(core::mem::offset_of!(MeshFrameData, instances) == 104);
const _: () = assert!(core::mem::offset_of!(MeshFrameData, clusters) == 112);
const _: () = assert!(core::mem::offset_of!(MeshFrameData, meshlets) == 120);
const _: () = assert!(core::mem::offset_of!(MeshFrameData, index_data) == 128);
const _: () = assert!(core::mem::offset_of!(MeshFrameData, lighting) == 136);
const _: () = assert!(core::mem::offset_of!(MeshFrameData, light_field) == 200);
const _: () = assert!(core::mem::offset_of!(MeshFrameData, light_field_dims) == 208);
const _: () = assert!(core::mem::offset_of!(MeshFrameData, light_field_cell_size) == 216);
const _: () = assert!(core::mem::offset_of!(MeshFrameData, light_field_gate) == 220);
const _: () = assert!(core::mem::offset_of!(MeshFrameData, time) == 224);
const _: () = assert!(core::mem::offset_of!(MeshFrameData, _pad4) == 228);
const _: () = assert!(core::mem::offset_of!(MeshFrameData, eye) == 232);
const _: () = assert!(core::mem::offset_of!(MeshFrameData, ramp_default_sampler) == 244);
const _: () = assert!(core::mem::offset_of!(MeshFrameData, _pad5) == 248);

/// Per-frame data for the visibility-buffer linework resolve. The fragment
/// pass reaches the same mesh streams as raster through `frame`; its texture
/// ids name the existing sampled and storage descriptor heaps.
#[gpu_data]
pub struct LineworkData {
    /// Inverse of the exact raster matrix that produced the depth attachment.
    pub clip_to_world: Mat4,
    pub frame: GpuPtr<MeshFrameData>,
    pub eye: [f32; 3],
    pub _pad0: f32,
    /// Sampled-heap D32 view. Texel fetches deliberately bypass filtering.
    pub depth_texture_id: u32,
    /// Storage-heap R32Uint visibility-token view.
    pub visibility_texture_id: u32,
    pub screen_size: [u32; 2],
    /// `cos(normal_threshold)`, prepared by the host from its degree dial.
    pub normal_cos_threshold: f32,
    /// World-space plane distance, including the center token/depth guard.
    pub plane_epsilon: f32,
    pub crease_strength: f32,
    pub step_strength: f32,
    pub fade_near: f32,
    pub fade_far: f32,
    /// Q10's attachment point; 1.0 is neutral until terrain light drives it.
    pub darkness_seat: f32,
    pub _pad1: [u32; 3],
    /// Null selects the ungated path.
    pub light_field: GpuPtr<f32>,
    pub light_field_dims: [u32; 2],
    pub light_field_cell_size: f32,
    /// 0 = ungated, 1 = full field sample.
    pub light_field_gate: f32,
    pub _pad2: [u32; 2],
}

const _: () = assert!(core::mem::size_of::<LineworkData>() == 176);
const _: () = assert!(core::mem::align_of::<LineworkData>() == 16);
const _: () = assert!(core::mem::offset_of!(LineworkData, clip_to_world) == 0);
const _: () = assert!(core::mem::offset_of!(LineworkData, frame) == 64);
const _: () = assert!(core::mem::offset_of!(LineworkData, eye) == 72);
const _: () = assert!(core::mem::offset_of!(LineworkData, _pad0) == 84);
const _: () = assert!(core::mem::offset_of!(LineworkData, depth_texture_id) == 88);
const _: () = assert!(core::mem::offset_of!(LineworkData, visibility_texture_id) == 92);
const _: () = assert!(core::mem::offset_of!(LineworkData, screen_size) == 96);
const _: () = assert!(core::mem::offset_of!(LineworkData, normal_cos_threshold) == 104);
const _: () = assert!(core::mem::offset_of!(LineworkData, plane_epsilon) == 108);
const _: () = assert!(core::mem::offset_of!(LineworkData, crease_strength) == 112);
const _: () = assert!(core::mem::offset_of!(LineworkData, step_strength) == 116);
const _: () = assert!(core::mem::offset_of!(LineworkData, fade_near) == 120);
const _: () = assert!(core::mem::offset_of!(LineworkData, fade_far) == 124);
const _: () = assert!(core::mem::offset_of!(LineworkData, darkness_seat) == 128);
const _: () = assert!(core::mem::offset_of!(LineworkData, _pad1) == 132);
const _: () = assert!(core::mem::offset_of!(LineworkData, light_field) == 144);
const _: () = assert!(core::mem::offset_of!(LineworkData, light_field_dims) == 152);
const _: () = assert!(core::mem::offset_of!(LineworkData, light_field_cell_size) == 160);
const _: () = assert!(core::mem::offset_of!(LineworkData, light_field_gate) == 164);
const _: () = assert!(core::mem::offset_of!(LineworkData, _pad2) == 168);

/// Build the raster matrix from the [`abi_core::View`] camera contract. The
/// resulting clip depth is eye-axis based, matching
/// [`abi_core::hardware_depth`]: a point `camera + ray * t` maps to
/// `near / (t * dot(forward, ray))`.
pub fn mesh_world_to_clip(view: &abi_core::View) -> Mat4 {
    let camera = Vec3::from_array(view.camera_position);
    let forward = Vec3::from_array(view.camera_forward);
    let right = Vec3::from_array(view.camera_right);
    let up = Vec3::from_array(view.camera_up);
    let x_scale = 1.0 / (view.tan_half_fov * view.aspect);
    let y_scale = 1.0 / view.tan_half_fov;

    let row0 = Vec4::new(
        right.x * x_scale,
        right.y * x_scale,
        right.z * x_scale,
        -right.dot(camera) * x_scale,
    );
    let row1 = Vec4::new(
        -up.x * y_scale,
        -up.y * y_scale,
        -up.z * y_scale,
        up.dot(camera) * y_scale,
    );
    let row2 = Vec4::new(0.0, 0.0, 0.0, view.depth_near_plane);
    let row3 = Vec4::new(forward.x, forward.y, forward.z, -forward.dot(camera));
    Mat4::from_cols(row0, row1, row2, row3).transpose()
}

/// The per-batch record consumed by non-indexed indirect multi-draw. The
/// command is first by Vulkan's ABI; the shader follows `batch_index` to the
/// compacted cluster range.
#[gpu_data]
pub struct IndirectData {
    pub cmd: DrawIndirectCommand,
    pub batch_index: u32,
}

const _: () = assert!(core::mem::size_of::<IndirectData>() == 20);
const _: () = assert!(core::mem::align_of::<IndirectData>() == 4);
const _: () = assert!(core::mem::offset_of!(IndirectData, cmd) == 0);
const _: () = assert!(core::mem::offset_of!(IndirectData, batch_index) == 16);

/// Dispatch data shared by `cluster_cull` and `cluster_build_args`. Culling
/// owns per-batch visible counters; argument build turns every candidate
/// batch (including empty ones) into an indirect command.
#[gpu_data]
pub struct ClusterCullData {
    pub instances: GpuPtr<MeshInstance>,
    pub batches: GpuPtr<MeshBatch>,
    pub transforms: GpuPtr<DrawTransform>,
    pub mesh_data: GpuPtr<MeshData>,
    pub meshlets: GpuPtr<Meshlet>,
    pub clusters: GpuPtr<ClusterInstance>,
    pub visible_counts: GpuPtr<u32>,
    pub output_indirect: GpuPtr<IndirectData>,
    pub instance_count: u32,
    pub batch_count: u32,
    /// Dispatch-wide y ceiling; each lane re-bounds against its mesh.
    pub max_meshlets_per_mesh: u32,
    /// Instances with `(flags & cull_mask) != 0` are culled.
    pub cull_mask: u32,
    /// xyz = normal, w = distance; `dot(n, p) + d >= 0` means inside.
    pub frustum_planes: [[f32; 4]; 6],
    pub camera_pos: [f32; 3],
    pub cone_cull_epsilon: f32,
}

const _: () = assert!(core::mem::size_of::<ClusterCullData>() == 192);
const _: () = assert!(core::mem::align_of::<ClusterCullData>() == 4);
const _: () = assert!(core::mem::offset_of!(ClusterCullData, instances) == 0);
const _: () = assert!(core::mem::offset_of!(ClusterCullData, batches) == 8);
const _: () = assert!(core::mem::offset_of!(ClusterCullData, transforms) == 16);
const _: () = assert!(core::mem::offset_of!(ClusterCullData, mesh_data) == 24);
const _: () = assert!(core::mem::offset_of!(ClusterCullData, meshlets) == 32);
const _: () = assert!(core::mem::offset_of!(ClusterCullData, clusters) == 40);
const _: () = assert!(core::mem::offset_of!(ClusterCullData, visible_counts) == 48);
const _: () = assert!(core::mem::offset_of!(ClusterCullData, output_indirect) == 56);
const _: () = assert!(core::mem::offset_of!(ClusterCullData, instance_count) == 64);
const _: () = assert!(core::mem::offset_of!(ClusterCullData, batch_count) == 68);
const _: () = assert!(core::mem::offset_of!(ClusterCullData, max_meshlets_per_mesh) == 72);
const _: () = assert!(core::mem::offset_of!(ClusterCullData, cull_mask) == 76);
const _: () = assert!(core::mem::offset_of!(ClusterCullData, frustum_planes) == 80);
const _: () = assert!(core::mem::offset_of!(ClusterCullData, camera_pos) == 176);
const _: () = assert!(core::mem::offset_of!(ClusterCullData, cone_cull_epsilon) == 188);

/// Sphere-vs-6-planes test. Called by the `cluster_cull` shader AND the
/// CPU verify twin — parity by construction.
pub fn sphere_inside_planes(center: Vec3, radius: f32, planes: &[[f32; 4]; 6]) -> bool {
    let mut i = 0;
    while i < 6 {
        let normal = Vec3::new(planes[i][0], planes[i][1], planes[i][2]);
        if normal.dot(center) + planes[i][3] < -radius {
            return false;
        }
        i += 1;
    }
    true
}

/// Conservative world-space radius scale of `m`'s linear part.
///
/// The largest transformed basis length is exact for orthogonal TRS but can
/// underbound inherited shear. The largest eigenvalue of `AᵀA` is bounded by
/// its maximum absolute row sum (Gershgorin), so this remains conservative for
/// every affine transform while staying cheap in the meshlet cull shader.
pub fn max_world_scale(m: &Mat4) -> f32 {
    let c0 = (*m * Vec4::new(1.0, 0.0, 0.0, 0.0)).truncate();
    let c1 = (*m * Vec4::new(0.0, 1.0, 0.0, 0.0)).truncate();
    let c2 = (*m * Vec4::new(0.0, 0.0, 1.0, 0.0)).truncate();
    let g00 = c0.dot(c0);
    let g11 = c1.dot(c1);
    let g22 = c2.dot(c2);
    let g01 = c0.dot(c1).abs();
    let g02 = c0.dot(c2).abs();
    let g12 = c1.dot(c2).abs();
    (g00 + g01 + g02)
        .max(g01 + g11 + g12)
        .max(g02 + g12 + g22)
        .sqrt()
}

/// Backface cone test: cull when the view-to-meshlet direction sits
/// inside the normal cone's rejection region (`dot >= cutoff + epsilon`). A
/// degenerate world axis (length < 1e-6) never culls. Shared for the same
/// parity reason as `sphere_inside_planes`.
pub fn meshlet_backfacing_to_camera(
    meshlet: &Meshlet,
    model_to_world_normal: &Mat4,
    camera_pos: Vec3,
    world_center: Vec3,
    epsilon: f32,
) -> bool {
    let axis = meshlet.cone_axis;
    let cone_axis_world =
        (*model_to_world_normal * Vec4::new(axis[0], axis[1], axis[2], 0.0)).truncate();
    let axis_len = cone_axis_world.length();
    if axis_len < 1.0e-6 {
        return false;
    }
    let cone_axis_world = cone_axis_world / axis_len;
    let view_to_meshlet = (world_center - camera_pos).normalize();
    view_to_meshlet.dot(cone_axis_world) >= meshlet.cone_cutoff + epsilon
}

/// Positive-vertex frustum test. Called by the `cluster_cull` shader AND
/// the CPU verify twin — parity by construction.
pub fn aabb_in_frustum(aabb_min: Vec3, aabb_max: Vec3, planes: &[[f32; 4]; 6]) -> bool {
    let mut i = 0;
    while i < 6 {
        let normal = Vec3::new(planes[i][0], planes[i][1], planes[i][2]);
        let d = planes[i][3];
        // Find the AABB corner most in the direction of the plane normal
        // (positive vertex).
        let p = Vec3::new(
            if normal.x >= 0.0 {
                aabb_max.x
            } else {
                aabb_min.x
            },
            if normal.y >= 0.0 {
                aabb_max.y
            } else {
                aabb_min.y
            },
            if normal.z >= 0.0 {
                aabb_max.z
            } else {
                aabb_min.z
            },
        );
        // If this corner is outside the plane, the whole AABB is outside.
        if normal.dot(p) + d < 0.0 {
            return false;
        }
        i += 1;
    }
    true
}

/// 8-corner world-AABB rebuild: corners of the local AABB through
/// `model_to_world`, min/max fold, then optional deformation dilation.
/// Shared for the same parity reason as `aabb_in_frustum`.
pub fn cull_world_aabb(bounds: &MeshBounds, model_to_world: &Mat4, dilation: f32) -> (Vec3, Vec3) {
    let corners = [
        Vec3::new(bounds.aabb_min[0], bounds.aabb_min[1], bounds.aabb_min[2]),
        Vec3::new(bounds.aabb_max[0], bounds.aabb_min[1], bounds.aabb_min[2]),
        Vec3::new(bounds.aabb_min[0], bounds.aabb_max[1], bounds.aabb_min[2]),
        Vec3::new(bounds.aabb_max[0], bounds.aabb_max[1], bounds.aabb_min[2]),
        Vec3::new(bounds.aabb_min[0], bounds.aabb_min[1], bounds.aabb_max[2]),
        Vec3::new(bounds.aabb_max[0], bounds.aabb_min[1], bounds.aabb_max[2]),
        Vec3::new(bounds.aabb_min[0], bounds.aabb_max[1], bounds.aabb_max[2]),
        Vec3::new(bounds.aabb_max[0], bounds.aabb_max[1], bounds.aabb_max[2]),
    ];
    let t0 = *model_to_world * corners[0].extend(1.0);
    let mut world_min = t0.truncate();
    let mut world_max = t0.truncate();
    let mut i = 1;
    while i < 8 {
        let transformed = *model_to_world * corners[i].extend(1.0);
        world_min = world_min.min(transformed.truncate());
        world_max = world_max.max(transformed.truncate());
        i += 1;
    }
    let d = dilation.max(0.0);
    (world_min - Vec3::splat(d), world_max + Vec3::splat(d))
}

/// Gribb-Hartmann plane extraction for the VULKAN clip volume
/// (-w<=x<=w, -w<=y<=w, 0<=z<=w). Planes are `{nx, ny, nz, d}`,
/// normalized, with `dot(n, p) + d >= 0` inside.
pub fn extract_frustum_planes(world_to_clip: &Mat4) -> [[f32; 4]; 6] {
    let r0 = world_to_clip.row(0);
    let r1 = world_to_clip.row(1);
    let r2 = world_to_clip.row(2);
    let r3 = world_to_clip.row(3);
    // Near: z >= 0 -> row2 (Vulkan). For infinite projections row2 is
    // degenerate (zero normal), so fall back to the looser OpenGL
    // convention (z + w >= 0 -> row2 + row3).
    let near_normal_len_sq = r2.x * r2.x + r2.y * r2.y + r2.z * r2.z;
    let near = if near_normal_len_sq > 1.0e-12 {
        r2
    } else {
        r2 + r3
    };
    let raw = [
        r0 + r3, // Left:   x + w >= 0 -> row0 + row3
        r3 - r0, // Right:  w - x >= 0 -> row3 - row0
        r1 + r3, // Bottom: y + w >= 0 -> row1 + row3
        r3 - r1, // Top:    w - y >= 0 -> row3 - row1
        near,
        r3 - r2, // Far:    w - z >= 0 -> row3 - row2
    ];
    let mut planes = [[0.0f32; 4]; 6];
    let mut i = 0;
    while i < 6 {
        let normal_len = raw[i].truncate().length();
        assert!(normal_len > 0.0);
        planes[i] = (raw[i] / normal_len).to_array();
        i += 1;
    }
    planes
}

/// Meshlet ranges live in the one global index buffer after each mesh's normal
/// range. `first_primitive` indexes the u32 remap table from meshlet-local
/// primitive to original mesh-local triangle; meshoptimizer reorders triangles
/// while clustering, so analytical consumers need the remap.
#[gpu_data]
pub struct Meshlet {
    pub mesh_index: u32,
    pub first_index: u32,
    pub tri_count: u32,
    pub first_primitive: u32,
    pub center: [f32; 3],
    pub radius: f32,
    pub cone_axis: [f32; 3],
    pub cone_cutoff: f32,
}

const _: () = assert!(core::mem::size_of::<Meshlet>() == 48);
const _: () = assert!(core::mem::align_of::<Meshlet>() == 4);
const _: () = assert!(core::mem::offset_of!(Meshlet, mesh_index) == 0);
const _: () = assert!(core::mem::offset_of!(Meshlet, first_index) == 4);
const _: () = assert!(core::mem::offset_of!(Meshlet, tri_count) == 8);
const _: () = assert!(core::mem::offset_of!(Meshlet, first_primitive) == 12);
const _: () = assert!(core::mem::offset_of!(Meshlet, center) == 16);
const _: () = assert!(core::mem::offset_of!(Meshlet, radius) == 28);
const _: () = assert!(core::mem::offset_of!(Meshlet, cone_axis) == 32);
const _: () = assert!(core::mem::offset_of!(Meshlet, cone_cutoff) == 44);

/// Contains only fields read by the mesh shader.
#[gpu_data]
pub struct MaterialEntry {
    pub base_color_map: u32,
    pub base_color_sampler: u32,
    pub metallic_roughness_map: u32,
    pub metallic_roughness_sampler: u32,
    pub normal_map: u32,
    pub normal_sampler: u32,
    pub emissive_map: u32,
    pub emissive_sampler: u32,
    pub alpha_mode: u32,
    pub alpha_cutoff: f32,
    pub occlusion_map: u32,
    pub occlusion_sampler: u32,
    pub base_color_factor: [f32; 4],
    pub emissive_factor: [f32; 3],
    pub occlusion_strength: f32,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub specular_ior: f32,
    pub material_type: u32,
    /// Bindless 2D ramp texture; 0 is the exact identity-ramp path.
    pub ramp_map: u32,
    /// Bindless sampler; 0 selects `MeshFrameData::ramp_default_sampler`.
    pub ramp_sampler: u32,
    /// Fresnel exponent for the additive ambient rim.
    pub rim_power: f32,
    /// Zero disables rim work.
    pub rim_boost: f32,
}

impl MaterialEntry {
    pub const fn standard() -> Self {
        // Texture map 0 means "factor only"; heap slot 0 remains the magenta fallback.
        Self {
            base_color_factor: [1.0, 1.0, 1.0, 1.0],
            occlusion_strength: 1.0,
            specular_ior: 1.5,
            alpha_cutoff: 0.5,
            ..Self::zeroed()
        }
    }

    const fn zeroed() -> Self {
        Self {
            base_color_map: 0,
            base_color_sampler: 0,
            metallic_roughness_map: 0,
            metallic_roughness_sampler: 0,
            normal_map: 0,
            normal_sampler: 0,
            emissive_map: 0,
            emissive_sampler: 0,
            alpha_mode: 0,
            alpha_cutoff: 0.0,
            occlusion_map: 0,
            occlusion_sampler: 0,
            base_color_factor: [0.0; 4],
            emissive_factor: [0.0; 3],
            occlusion_strength: 0.0,
            metallic_factor: 0.0,
            roughness_factor: 0.0,
            specular_ior: 0.0,
            material_type: 0,
            ramp_map: 0,
            ramp_sampler: 0,
            rim_power: 0.0,
            rim_boost: 0.0,
        }
    }
}

const _: () = assert!(core::mem::size_of::<MaterialEntry>() == 112);
const _: () = assert!(core::mem::align_of::<MaterialEntry>() == 4);
const _: () = assert!(core::mem::offset_of!(MaterialEntry, base_color_map) == 0);
const _: () = assert!(core::mem::offset_of!(MaterialEntry, base_color_sampler) == 4);
const _: () = assert!(core::mem::offset_of!(MaterialEntry, metallic_roughness_map) == 8);
const _: () = assert!(core::mem::offset_of!(MaterialEntry, metallic_roughness_sampler) == 12);
const _: () = assert!(core::mem::offset_of!(MaterialEntry, normal_map) == 16);
const _: () = assert!(core::mem::offset_of!(MaterialEntry, normal_sampler) == 20);
const _: () = assert!(core::mem::offset_of!(MaterialEntry, emissive_map) == 24);
const _: () = assert!(core::mem::offset_of!(MaterialEntry, emissive_sampler) == 28);
const _: () = assert!(core::mem::offset_of!(MaterialEntry, alpha_mode) == 32);
const _: () = assert!(core::mem::offset_of!(MaterialEntry, alpha_cutoff) == 36);
const _: () = assert!(core::mem::offset_of!(MaterialEntry, occlusion_map) == 40);
const _: () = assert!(core::mem::offset_of!(MaterialEntry, occlusion_sampler) == 44);
const _: () = assert!(core::mem::offset_of!(MaterialEntry, base_color_factor) == 48);
const _: () = assert!(core::mem::offset_of!(MaterialEntry, emissive_factor) == 64);
const _: () = assert!(core::mem::offset_of!(MaterialEntry, occlusion_strength) == 76);
const _: () = assert!(core::mem::offset_of!(MaterialEntry, metallic_factor) == 80);
const _: () = assert!(core::mem::offset_of!(MaterialEntry, roughness_factor) == 84);
const _: () = assert!(core::mem::offset_of!(MaterialEntry, specular_ior) == 88);
const _: () = assert!(core::mem::offset_of!(MaterialEntry, material_type) == 92);
const _: () = assert!(core::mem::offset_of!(MaterialEntry, ramp_map) == 96);
const _: () = assert!(core::mem::offset_of!(MaterialEntry, ramp_sampler) == 100);
const _: () = assert!(core::mem::offset_of!(MaterialEntry, rim_power) == 104);
const _: () = assert!(core::mem::offset_of!(MaterialEntry, rim_boost) == 108);
