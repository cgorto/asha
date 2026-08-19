//! Mesh scene data layer.
//!
//! One fixed-capacity scene owns the GPU-resident mesh pools, draw tables,
//! meshlets, and CPU mirrors that hosts use for direct draws and tests. A mesh
//! is visible to the renderer only after `add_mesh` has staged every stream,
//! index range, meshlet, bound, and table entry.

use std::collections::{HashMap, VecDeque};
use std::mem::{size_of, size_of_val};
use std::ops::Range;

use abi_core::GpuPtr;
use abi_mesh::DeformerStack;
use abi_mesh::world_transform;
pub use abi_mesh::{
    ClusterInstance, DrawTransform, IndirectData, JointWeights, MaterialEntry, MeshBatch,
    MeshBounds, MeshData, MeshFrameData, MeshInstance, MeshShadeLighting, MeshTableEntry, Meshlet,
};
use gpu::{CommandBuffer, Gpu, HazardFlags, Memory, Queue, Stage};

pub mod cull;
pub mod forward;
pub mod linework;
pub mod prepass;
pub mod primitives;
pub mod shadow;
pub mod silhouette;

pub use cull::ClusterCullPass;
pub use forward::{
    MeshForwardPass, MeshForwardSurfaceTargets, MeshForwardTargets, MeshLightField,
    ShaderCoatSlice, ShaderGroupKind, ShaderGroupSlice,
};
pub use linework::{LineworkDials, MeshLineworkPass};
pub use prepass::MeshDepthPrepass;
pub use primitives::MeshBuffers;
pub use shadow::{
    ShadowBlasDesc, ShadowBlasQueryPass, ShadowBlasStats, ShadowTlasBuilder, ShadowTlasStats,
    ShadowWorldQueryPass,
};
pub use silhouette::MeshSilhouettePass;

pub const MESHLET_MAX_VERTICES: usize = 64;
pub const MESHLET_MAX_TRIANGLES: usize = 124;
const MESHLET_CONE_WEIGHT: f32 = 0.5;

/// Growth headroom for rewritten mesh slots.
/// Initial reservations are exact; later growth may reserve up to 1.5×.
const SLOT_HEADROOM: f32 = 1.5;

/// One stream reservation for a mesh slot. `cap == 0` means unreserved.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ArenaSlot {
    pub(crate) base: u32,
    pub(crate) cap: u32,
}

/// Arena reservations for one mesh slot, reused across rewrites.
#[derive(Debug, Clone, Copy, Default)]
struct MeshSlot {
    positions: ArenaSlot,
    normals: ArenaSlot,
    uvs: ArenaSlot,
    tangents: ArenaSlot,
    colors: ArenaSlot,
    joint_weights: ArenaSlot,
    indices: ArenaSlot,
    meshlets: ArenaSlot,
    primitives: ArenaSlot,
    live: bool,
}

/// Rounds growth up when remaining capacity permits.
fn with_headroom(need: u32, remaining: u32) -> u32 {
    let want = (need as f64 * SLOT_HEADROOM as f64).ceil();
    let want = if want >= u32::MAX as f64 {
        u32::MAX
    } else {
        (want as u32).max(need)
    };
    if want <= remaining { want } else { need }
}

/// Reuses a reservation or appends a larger one to the arena.
pub(crate) fn reserve(
    slot: &mut ArenaSlot,
    offset: &mut u32,
    need: u32,
    capacity: u32,
    name: &str,
) {
    if need == 0 || need <= slot.cap {
        return;
    }
    let want = if slot.cap == 0 {
        need
    } else {
        with_headroom(need, capacity.saturating_sub(*offset))
    };
    assert_capacity(*offset, want, capacity, name);
    slot.base = *offset;
    slot.cap = want;
    *offset += want;
}

/// Indexed write into a per-slot CPU mirror, growing it to the slot bound.
pub(crate) fn set_row<T: Copy + Default>(dst: &mut Vec<T>, index: u32, value: T) {
    if dst.len() <= index as usize {
        dst.resize(index as usize + 1, T::default());
    }
    dst[index as usize] = value;
}

/// Writes a slice into a CPU arena mirror at `base`.
/// Consumers address live slots by base and count.
fn write_arena<T: Copy + Default>(dst: &mut Vec<T>, base: u32, src: &[T]) {
    let end = base as usize + src.len();
    if dst.len() < end {
        dst.resize(end, T::default());
    }
    dst[base as usize..end].copy_from_slice(src);
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MeshSceneDesc {
    pub max_meshes: u32,
    pub max_instances: u32,
    pub max_materials: u32,
    pub vertex_capacity: u32,
    /// Compact capacity consumed only by meshes that carry skin weights.
    pub joint_weight_capacity: u32,
    pub index_capacity: u32,
    pub max_meshlets: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct MeshDesc<'a> {
    pub positions: &'a [[f32; 3]],
    pub normals: &'a [[f32; 3]],
    pub uvs: &'a [[f32; 2]],
    pub indices: &'a [u32],
    pub tangents: Option<&'a [[f32; 4]]>,
    pub joint_weights: Option<&'a [JointWeights]>,
    /// Optional vertex-parallel RGBA tint. `None` means untinted and costs
    /// nothing: the table entry's pointer stays null.
    pub colors: Option<&'a [[f32; 4]]>,
}

// Shared frame-allocation contract for all mesh passes.
pub use gpu::pass::FrameAlloc;

/// Per-frame instance and batch streams, with CPU mirrors for validation.
#[derive(Clone, Copy)]
pub struct MeshInstances<'a> {
    pub instances: GpuPtr<MeshInstance>,
    pub batches: GpuPtr<MeshBatch>,
    pub transforms: GpuPtr<DrawTransform>,
    pub deformers: GpuPtr<DeformerStack>,
    pub instances_cpu: &'a [MeshInstance],
    pub batches_cpu: &'a [MeshBatch],
    pub transforms_cpu: &'a [DrawTransform],
}

impl MeshInstances<'_> {
    pub fn count(&self) -> u32 {
        self.instances_cpu.len() as u32
    }

    pub fn batch_count(&self) -> u32 {
        self.batches_cpu.len() as u32
    }
}

/// World-to-clip matrix consumed by mesh rasterization and culling.
#[derive(Clone, Copy)]
pub struct MeshRasterView {
    pub world_to_clip: abi_core::glam::Mat4,
}

/// Shared zeroed [`MeshFrameData`] base for mesh raster passes.
pub(crate) fn mesh_frame_data(
    scene: &MeshScene,
    instances: MeshInstances<'_>,
    clusters: GpuPtr<ClusterInstance>,
    view: MeshRasterView,
) -> MeshFrameData {
    MeshFrameData {
        world_to_clip: view.world_to_clip,
        mesh_table: scene.mesh_table_ptr(),
        transforms: instances.transforms,
        materials: scene.materials_ptr(),
        deformers: instances.deformers,
        batches: instances.batches,
        instances: instances.instances,
        clusters,
        meshlets: scene.meshlets_ptr(),
        index_data: scene.global_index_buffer_ptr(),
        lighting: MeshShadeLighting::zeroed(),
        light_field: GpuPtr::default(),
        light_field_dims: [0; 2],
        light_field_cell_size: 0.0,
        light_field_gate: 0.0,
        time: 0.0,
        _pad4: 0,
        eye: [0.0; 3],
        ramp_default_sampler: 0,
        _pad5: [0; 2],
    }
}

macro_rules! scene_handle {
    ($name:ident) => {
        #[repr(transparent)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
        pub struct $name(u64);

        impl $name {
            fn pack(index: u32, generation: u32) -> Self {
                assert!(generation != 0, "handle generation 0 is invalid");
                Self(((generation as u64) << 32) | index as u64)
            }

            pub fn is_null(self) -> bool {
                self.0 == 0
            }

            pub fn index(self) -> u32 {
                self.0 as u32
            }

            pub fn generation(self) -> u32 {
                (self.0 >> 32) as u32
            }

            pub fn raw(self) -> u64 {
                self.0
            }
        }
    };
}

scene_handle!(MeshHandle);
scene_handle!(MaterialHandle);
scene_handle!(InstanceHandle);

struct UploadCopy {
    staging: gpu::Ptr<u8>,
    dst: gpu::Ptr<u8>,
    bytes: u64,
}

#[derive(Default)]
struct UploadBatch {
    copies: Vec<UploadCopy>,
}

impl UploadBatch {
    fn stage_slice<T: Copy>(&mut self, gpu: &Gpu, dst: gpu::Ptr<T>, dst_offset: u32, data: &[T]) {
        if data.is_empty() {
            return;
        }
        assert!(
            size_of::<T>() != 0,
            "GPU uploads require non-zero-sized elements"
        );
        let bytes = size_of_val(data) as u64;
        let staging = gpu.alloc_slice::<T>(data.len() as u64, Memory::Default);
        // SAFETY: `staging` is a fresh host-visible allocation with `data.len()` elements.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), staging.cpu, data.len());
        }
        let dst_offset_bytes = dst_offset as i64 * size_of::<T>() as i64;
        let dst = gpu.mem_suballoc(
            dst.cast(),
            dst_offset_bytes,
            size_of::<T>() as u64,
            data.len() as u64,
        );
        self.copies.push(UploadCopy {
            staging: staging.cast(),
            dst,
            bytes,
        });
    }

    fn submit(self, gpu: &Gpu) {
        if self.copies.is_empty() {
            return;
        }

        let cb = gpu.commands_begin(Queue::Main);
        for copy in &self.copies {
            gpu.cmd_mem_copy_raw(cb, copy.dst, copy.staging, copy.bytes);
        }
        gpu.cmd_barrier(cb, Stage::Transfer, Stage::All, HazardFlags::empty());
        gpu.queue_submit(Queue::Main, &[cb]);
        gpu.queue_wait_idle(Queue::Main);

        for copy in self.copies {
            gpu.free(copy.staging);
        }
    }
}

struct PreparedMesh {
    positions: Vec<[f32; 4]>,
    normals: Vec<[f32; 4]>,
    uvs: Vec<[f32; 2]>,
    tangents: Option<Vec<[f32; 4]>>,
    colors: Option<Vec<[f32; 4]>>,
    indices: Vec<u32>,
    meshlets: Vec<Meshlet>,
    meshlet_vertex_counts: Vec<u32>,
    primitive_ids: Vec<u32>,
    meshlet_index_count: u32,
    bounds: MeshBounds,
}

/// High-water mark of every mesh arena, in elements.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MeshArenaUsage {
    pub positions: u32,
    pub normals: u32,
    pub uvs: u32,
    pub tangents: u32,
    pub colors: u32,
    pub joint_weights: u32,
    pub indices: u32,
    pub meshlets: u32,
    pub primitive_ids: u32,
    pub shadow_nodes: u32,
    pub shadow_primitives: u32,
}

pub struct MeshScene {
    desc: MeshSceneDesc,
    positions: gpu::Ptr<[f32; 4]>,
    normals: gpu::Ptr<[f32; 4]>,
    uvs: gpu::Ptr<[f32; 2]>,
    tangents: gpu::Ptr<[f32; 4]>,
    colors: gpu::Ptr<[f32; 4]>,
    joint_weights: gpu::Ptr<JointWeights>,
    index_buffer: gpu::Ptr<u32>,
    mesh_table: gpu::Ptr<MeshTableEntry>,
    mesh_data: gpu::Ptr<MeshData>,
    mesh_bounds: gpu::Ptr<MeshBounds>,
    meshlets: gpu::Ptr<Meshlet>,
    meshlet_primitive_ids: gpu::Ptr<u32>,
    materials: gpu::Ptr<MaterialEntry>,
    transforms: gpu::Ptr<DrawTransform>,
    instances: gpu::Ptr<MeshInstance>,
    batches: gpu::Ptr<MeshBatch>,
    shadow: Option<shadow::ShadowBlasPools>,
    meshlet_primitive_capacity: u32,
    position_offset: u32,
    normal_offset: u32,
    uv_offset: u32,
    tangent_offset: u32,
    color_offset: u32,
    joint_weight_offset: u32,
    index_offset: u32,
    meshlet_offset: u32,
    primitive_id_offset: u32,
    max_meshlets_per_mesh: u32,
    /// One reservation record per mesh slot; `len()` is the slot bound.
    mesh_slots: Vec<MeshSlot>,
    /// Slots `remove_mesh` retired, newest first. A popped slot keeps its
    /// arena reservation — the free list hands back the *reservation*, not
    /// just the index.
    mesh_free: Vec<u32>,
    mesh_live_count: u32,
    mesh_generations: Vec<u32>,
    material_generations: Vec<u32>,
    instance_generations: Vec<u32>,
    mesh_data_cpu: Vec<MeshData>,
    mesh_bounds_cpu: Vec<MeshBounds>,
    mesh_vertex_counts_cpu: Vec<u32>,
    meshlet_index_counts_cpu: Vec<u32>,
    indices_cpu: Vec<u32>,
    meshlets_cpu: Vec<Meshlet>,
    meshlet_vertex_counts_cpu: Vec<u32>,
    meshlet_primitive_ids_cpu: Vec<u32>,
    materials_cpu: Vec<MaterialEntry>,
    transforms_cpu: Vec<DrawTransform>,
    instances_cpu: Vec<MeshInstance>,
    batches_cpu: Vec<MeshBatch>,
}

impl MeshScene {
    pub fn new(gpu: &Gpu, desc: &MeshSceneDesc) -> Self {
        Self::new_inner(gpu, desc, None)
    }

    /// Allocates raster pools and immutable shadow-BLAS storage.
    pub fn new_with_shadows(gpu: &Gpu, desc: &MeshSceneDesc, shadow_desc: ShadowBlasDesc) -> Self {
        Self::new_inner(gpu, desc, Some(shadow_desc))
    }

    /// Allocates fixed GPU pools; capacities cannot grow safely.
    fn new_inner(gpu: &Gpu, desc: &MeshSceneDesc, shadow_desc: Option<ShadowBlasDesc>) -> Self {
        let meshlet_primitive_capacity = desc
            .max_meshlets
            .checked_mul(MESHLET_MAX_TRIANGLES as u32)
            .expect("max_meshlets * 124 exceeds u32");

        Self {
            desc: *desc,
            positions: gpu.alloc_slice::<[f32; 4]>(desc.vertex_capacity as u64, Memory::Gpu),
            normals: gpu.alloc_slice::<[f32; 4]>(desc.vertex_capacity as u64, Memory::Gpu),
            uvs: gpu.alloc_slice::<[f32; 2]>(desc.vertex_capacity as u64, Memory::Gpu),
            tangents: gpu.alloc_slice::<[f32; 4]>(desc.vertex_capacity as u64, Memory::Gpu),
            colors: gpu.alloc_slice::<[f32; 4]>(desc.vertex_capacity as u64, Memory::Gpu),
            joint_weights: gpu
                .alloc_slice::<JointWeights>(desc.joint_weight_capacity as u64, Memory::Gpu),
            index_buffer: gpu.alloc_slice::<u32>(desc.index_capacity as u64, Memory::Gpu),
            mesh_table: gpu.alloc_slice::<MeshTableEntry>(desc.max_meshes as u64, Memory::Gpu),
            mesh_data: gpu.alloc_slice::<MeshData>(desc.max_meshes as u64, Memory::Gpu),
            mesh_bounds: gpu.alloc_slice::<MeshBounds>(desc.max_meshes as u64, Memory::Gpu),
            meshlets: gpu.alloc_slice::<Meshlet>(desc.max_meshlets as u64, Memory::Gpu),
            meshlet_primitive_ids: gpu
                .alloc_slice::<u32>(meshlet_primitive_capacity as u64, Memory::Gpu),
            materials: gpu.alloc_slice::<MaterialEntry>(desc.max_materials as u64, Memory::Gpu),
            transforms: gpu.alloc_slice::<DrawTransform>(desc.max_instances as u64, Memory::Gpu),
            instances: gpu.alloc_slice::<MeshInstance>(desc.max_instances as u64, Memory::Gpu),
            // Each instance may require a distinct batch.
            batches: gpu.alloc_slice::<MeshBatch>(desc.max_instances as u64, Memory::Gpu),
            shadow: shadow_desc
                .map(|shadow_desc| shadow::ShadowBlasPools::new(gpu, desc.max_meshes, shadow_desc)),
            meshlet_primitive_capacity,
            position_offset: 0,
            normal_offset: 0,
            uv_offset: 0,
            tangent_offset: 0,
            color_offset: 0,
            joint_weight_offset: 0,
            index_offset: 0,
            meshlet_offset: 0,
            primitive_id_offset: 0,
            max_meshlets_per_mesh: 0,
            mesh_slots: Vec::with_capacity(desc.max_meshes as usize),
            mesh_free: Vec::new(),
            mesh_live_count: 0,
            mesh_generations: Vec::with_capacity(desc.max_meshes as usize),
            material_generations: Vec::with_capacity(desc.max_materials as usize),
            instance_generations: Vec::with_capacity(desc.max_instances as usize),
            mesh_data_cpu: Vec::with_capacity(desc.max_meshes as usize),
            mesh_bounds_cpu: Vec::with_capacity(desc.max_meshes as usize),
            mesh_vertex_counts_cpu: Vec::with_capacity(desc.max_meshes as usize),
            meshlet_index_counts_cpu: Vec::with_capacity(desc.max_meshes as usize),
            indices_cpu: Vec::with_capacity(desc.index_capacity as usize),
            meshlets_cpu: Vec::with_capacity(desc.max_meshlets as usize),
            meshlet_vertex_counts_cpu: Vec::with_capacity(desc.max_meshlets as usize),
            meshlet_primitive_ids_cpu: Vec::with_capacity(meshlet_primitive_capacity as usize),
            materials_cpu: Vec::with_capacity(desc.max_materials as usize),
            transforms_cpu: Vec::with_capacity(desc.max_instances as usize),
            instances_cpu: Vec::with_capacity(desc.max_instances as usize),
            batches_cpu: Vec::with_capacity(desc.max_instances as usize),
        }
    }

    /// Atomically registers mesh streams, ranges, bounds, and table rows.
    /// Reclaimed slots retain their arena reservations.
    pub fn add_mesh(&mut self, gpu: &Gpu, desc: MeshDesc<'_>) -> MeshHandle {
        let mesh_index = match self.mesh_free.pop() {
            Some(index) => index,
            None => {
                let index = self.mesh_slots.len() as u32;
                assert_capacity(index, 1, self.desc.max_meshes, "max_meshes");
                self.mesh_slots.push(MeshSlot::default());
                self.mesh_generations.push(1);
                index
            }
        };
        self.mesh_slots[mesh_index as usize].live = true;
        self.mesh_live_count += 1;
        self.write_mesh(gpu, mesh_index, desc);
        // Registration can change meshlet counts in existing batches.
        self.rebuild_batch_ranges_for(gpu, mesh_index);
        MeshHandle::pack(mesh_index, self.mesh_generations[mesh_index as usize])
    }

    /// Rewrites registered geometry without changing its handle generation.
    /// Fits reuse the reservation; larger meshes append a new reservation.
    pub fn update_mesh(&mut self, gpu: &Gpu, mesh: MeshHandle, desc: MeshDesc<'_>) {
        let mesh_index = self.validate_mesh(mesh) as u32;
        assert!(
            self.mesh_slots[mesh_index as usize].live,
            "update_mesh on removed mesh slot {mesh_index}"
        );
        self.write_mesh(gpu, mesh_index, desc);
        // Updates can change meshlet counts in existing batches.
        self.rebuild_batch_ranges_for(gpu, mesh_index);
    }

    /// Retires a mesh slot and increments its handle generation.
    /// Retired table bytes remain unused until the slot is reclaimed.
    pub fn remove_mesh(&mut self, mesh: MeshHandle) {
        let mesh_index = self.validate_mesh(mesh) as u32;
        assert!(
            self.mesh_slots[mesh_index as usize].live,
            "remove_mesh on already-removed mesh slot {mesh_index}"
        );
        self.mesh_slots[mesh_index as usize].live = false;
        self.mesh_live_count -= 1;
        let generation = &mut self.mesh_generations[mesh_index as usize];
        *generation = generation
            .checked_add(1)
            .expect("mesh generation exhausted");
        self.mesh_free.push(mesh_index);
    }

    /// Shared write path for mesh registration and updates.
    fn write_mesh(&mut self, gpu: &Gpu, mesh_index: u32, desc: MeshDesc<'_>) {
        validate_mesh_desc(desc);
        let prepared_shadow = self
            .shadow
            .as_ref()
            .map(|_| shadow::ShadowBlasPools::prepare(desc));

        let mut prepared = prepare_mesh(desc, mesh_index);
        let vertex_count = to_u32(prepared.positions.len(), "vertex_capacity");
        let index_count = to_u32(desc.indices.len(), "index_capacity");
        let total_index_count = to_u32(prepared.indices.len(), "index_capacity");
        let meshlet_count = to_u32(prepared.meshlets.len(), "max_meshlets");
        let cluster_vertex_count = prepared
            .meshlets
            .iter()
            .map(|meshlet| {
                meshlet
                    .tri_count
                    .checked_mul(3)
                    .expect("meshlet vertex count overflow")
            })
            .max()
            .expect("registered mesh has at least one meshlet");
        let primitive_id_count = to_u32(prepared.primitive_ids.len(), "max_meshlets");

        // Reserve each stream independently.
        let mut slot = self.mesh_slots[mesh_index as usize];
        let vertex_capacity = self.desc.vertex_capacity;
        reserve(
            &mut slot.positions,
            &mut self.position_offset,
            vertex_count,
            vertex_capacity,
            "vertex_capacity",
        );
        reserve(
            &mut slot.normals,
            &mut self.normal_offset,
            vertex_count,
            vertex_capacity,
            "vertex_capacity",
        );
        reserve(
            &mut slot.uvs,
            &mut self.uv_offset,
            vertex_count,
            vertex_capacity,
            "vertex_capacity",
        );
        if prepared.tangents.is_some() {
            reserve(
                &mut slot.tangents,
                &mut self.tangent_offset,
                vertex_count,
                vertex_capacity,
                "vertex_capacity",
            );
        }
        if prepared.colors.is_some() {
            reserve(
                &mut slot.colors,
                &mut self.color_offset,
                vertex_count,
                vertex_capacity,
                "vertex_capacity",
            );
        }
        if desc.joint_weights.is_some() {
            reserve(
                &mut slot.joint_weights,
                &mut self.joint_weight_offset,
                vertex_count,
                self.desc.joint_weight_capacity,
                "joint_weight_capacity",
            );
        }
        reserve(
            &mut slot.indices,
            &mut self.index_offset,
            total_index_count,
            self.desc.index_capacity,
            "index_capacity",
        );
        reserve(
            &mut slot.meshlets,
            &mut self.meshlet_offset,
            meshlet_count,
            self.desc.max_meshlets,
            "max_meshlets",
        );
        reserve(
            &mut slot.primitives,
            &mut self.primitive_id_offset,
            primitive_id_count,
            self.meshlet_primitive_capacity,
            "max_meshlets",
        );
        self.mesh_slots[mesh_index as usize] = slot;

        let first_index = slot.indices.base;
        let meshlet_offset = slot.meshlets.base;
        let first_primitive = slot.primitives.base;
        rebase_meshlets(&mut prepared.meshlets, first_index, first_primitive);

        let mesh_data = MeshData {
            idx_count: index_count,
            first_index,
            meshlet_offset,
            meshlet_count,
            cluster_vertex_count,
        };
        let table_entry = MeshTableEntry {
            positions: self.positions.gpu.offset(slot.positions.base as i64),
            normals: self.normals.gpu.offset(slot.normals.base as i64),
            uvs: self.uvs.gpu.offset(slot.uvs.base as i64),
            tangents: if prepared.tangents.is_some() {
                self.tangents.gpu.offset(slot.tangents.base as i64)
            } else {
                GpuPtr::null()
            },
            joint_weights: if desc.joint_weights.is_some() {
                self.joint_weights
                    .gpu
                    .offset(slot.joint_weights.base as i64)
            } else {
                GpuPtr::null()
            },
            colors: if prepared.colors.is_some() {
                self.colors.gpu.offset(slot.colors.base as i64)
            } else {
                GpuPtr::null()
            },
        };

        let mut upload = UploadBatch::default();
        upload.stage_slice(
            gpu,
            self.positions,
            slot.positions.base,
            &prepared.positions,
        );
        upload.stage_slice(gpu, self.normals, slot.normals.base, &prepared.normals);
        upload.stage_slice(gpu, self.uvs, slot.uvs.base, &prepared.uvs);
        if let Some(tangents) = prepared.tangents.as_ref() {
            upload.stage_slice(gpu, self.tangents, slot.tangents.base, tangents);
        }
        if let Some(colors) = prepared.colors.as_ref() {
            upload.stage_slice(gpu, self.colors, slot.colors.base, colors);
        }
        if let Some(joint_weights) = desc.joint_weights {
            upload.stage_slice(
                gpu,
                self.joint_weights,
                slot.joint_weights.base,
                joint_weights,
            );
        }
        upload.stage_slice(gpu, self.index_buffer, slot.indices.base, &prepared.indices);
        upload.stage_slice(gpu, self.meshlets, slot.meshlets.base, &prepared.meshlets);
        upload.stage_slice(
            gpu,
            self.meshlet_primitive_ids,
            slot.primitives.base,
            &prepared.primitive_ids,
        );
        upload.stage_slice(gpu, self.mesh_table, mesh_index, &[table_entry]);
        upload.stage_slice(gpu, self.mesh_data, mesh_index, &[mesh_data]);
        upload.stage_slice(gpu, self.mesh_bounds, mesh_index, &[prepared.bounds]);
        if let (Some(shadow), Some(prepared_shadow)) = (&mut self.shadow, prepared_shadow) {
            shadow.stage(
                gpu,
                &mut upload,
                mesh_index,
                table_entry.positions,
                self.index_buffer.gpu.offset(first_index as i64),
                prepared_shadow,
            );
        }
        upload.submit(gpu);

        set_row(&mut self.mesh_data_cpu, mesh_index, mesh_data);
        set_row(&mut self.mesh_bounds_cpu, mesh_index, prepared.bounds);
        set_row(&mut self.mesh_vertex_counts_cpu, mesh_index, vertex_count);
        set_row(
            &mut self.meshlet_index_counts_cpu,
            mesh_index,
            prepared.meshlet_index_count,
        );
        write_arena(&mut self.indices_cpu, slot.indices.base, &prepared.indices);
        write_arena(
            &mut self.meshlet_vertex_counts_cpu,
            slot.meshlets.base,
            &prepared.meshlet_vertex_counts,
        );
        write_arena(
            &mut self.meshlets_cpu,
            slot.meshlets.base,
            &prepared.meshlets,
        );
        write_arena(
            &mut self.meshlet_primitive_ids_cpu,
            slot.primitives.base,
            &prepared.primitive_ids,
        );

        self.max_meshlets_per_mesh = self.max_meshlets_per_mesh.max(meshlet_count);
    }

    pub fn add_material(&mut self, gpu: &Gpu, material: MaterialEntry) -> MaterialHandle {
        let material_index = self.material_generations.len() as u32;
        assert_capacity(material_index, 1, self.desc.max_materials, "max_materials");

        let mut upload = UploadBatch::default();
        upload.stage_slice(gpu, self.materials, material_index, &[material]);
        upload.submit(gpu);

        self.material_generations.push(1);
        self.materials_cpu.push(material);
        MaterialHandle::pack(material_index, 1)
    }

    /// Replaces one material row; call only for authored changes.
    pub fn update_material(&mut self, gpu: &Gpu, handle: MaterialHandle, material: MaterialEntry) {
        let material_index = self.validate_material(handle) as u32;
        let mut upload = UploadBatch::default();
        upload.stage_slice(gpu, self.materials, material_index, &[material]);
        upload.submit(gpu);
        self.materials_cpu[material_index as usize] = material;
    }

    pub fn add_instance(
        &mut self,
        gpu: &Gpu,
        mesh: MeshHandle,
        world: abi_core::glam::Mat4,
        material: MaterialHandle,
    ) -> InstanceHandle {
        let mesh_index = self.validate_mesh(mesh) as u32;
        let material_index = self.validate_material(material) as u32;
        let instance_index = self.instance_generations.len() as u32;
        assert_capacity(instance_index, 1, self.desc.max_instances, "max_instances");

        let transform = world_transform(world);
        let batch_index = self.find_or_add_batch(mesh_index, material_index);
        let instance = MeshInstance {
            batch_index,
            transform_index: instance_index,
            flags: 0,
            outline_group: 0,
            instance_color: [1.0; 4],
            joint_transforms: GpuPtr::null(),
            deformer_slot: 0,
            bounds_dilation: 0.0,
        };

        let mut upload = UploadBatch::default();
        upload.stage_slice(gpu, self.transforms, instance_index, &[transform]);
        upload.stage_slice(gpu, self.instances, instance_index, &[instance]);
        upload.submit(gpu);

        self.instance_generations.push(1);
        self.transforms_cpu.push(transform);
        self.instances_cpu.push(instance);
        self.rebuild_batch_ranges(gpu);
        InstanceHandle::pack(instance_index, 1)
    }

    /// Blocking cold-path world update; waits for the transfer queue to idle.
    pub fn set_world(&mut self, gpu: &Gpu, instance: InstanceHandle, world: abi_core::glam::Mat4) {
        let instance_index = self.validate_instance(instance) as u32;
        // Moving instances require dynamic shadow bounds.
        self.instances_cpu[instance_index as usize].flags |= abi_mesh::MESH_FLAG_DYNAMIC;
        let transform = world_transform(world);

        let mut upload = UploadBatch::default();
        upload.stage_slice(gpu, self.transforms, instance_index, &[transform]);
        upload.submit(gpu);

        self.transforms_cpu[instance_index as usize] = transform;
    }

    /// Records a non-waiting, frame-safe transform upload.
    ///
    /// `staging` is a host-visible ring slot; reuse it only after its prior
    /// copy completes, using the same frame-ownership gate as frame arenas.
    /// Bracket uploads with All→Transfer and Transfer→All barriers, and
    /// record before culling.
    pub fn stage_world(
        &mut self,
        gpu: &Gpu,
        cb: CommandBuffer,
        staging: gpu::Ptr<DrawTransform>,
        instance: InstanceHandle,
        world: abi_core::glam::Mat4,
    ) {
        let instance_index = self.validate_instance(instance) as u32;
        self.instances_cpu[instance_index as usize].flags |= abi_mesh::MESH_FLAG_DYNAMIC;
        let transform = world_transform(world);
        // SAFETY: caller contract — `staging` is host-visible and GPU-idle.
        unsafe { *staging.cpu = transform };
        let bytes = size_of::<DrawTransform>() as u64;
        let dst = gpu.mem_suballoc(
            self.transforms.cast(),
            instance_index as i64 * bytes as i64,
            bytes,
            1,
        );
        gpu.cmd_mem_copy_raw(cb, dst, staging.cast(), bytes);
        self.transforms_cpu[instance_index as usize] = transform;
    }

    /// Rewrites flags consumed by culling, including `MESH_FLAG_HIDDEN`.
    pub fn set_flags(&mut self, gpu: &Gpu, instance: InstanceHandle, flags: u32) {
        let instance_index = self.validate_instance(instance) as u32;
        let mut mesh_instance = self.instances_cpu[instance_index as usize];
        mesh_instance.flags = flags;

        let mut upload = UploadBatch::default();
        upload.stage_slice(gpu, self.instances, instance_index, &[mesh_instance]);
        upload.submit(gpu);

        self.instances_cpu[instance_index as usize] = mesh_instance;
    }

    /// Sets bounds dilation for retained scenes and direct deformation users.
    pub fn set_bounds_dilation_direct(
        &mut self,
        gpu: &Gpu,
        instance: InstanceHandle,
        dilation: f32,
    ) {
        assert!(
            dilation.is_finite() && dilation >= 0.0,
            "bounds dilation must be finite and nonnegative"
        );
        let instance_index = self.validate_instance(instance) as u32;
        let mut mesh_instance = self.instances_cpu[instance_index as usize];
        mesh_instance.bounds_dilation = dilation;

        let mut upload = UploadBatch::default();
        upload.stage_slice(gpu, self.instances, instance_index, &[mesh_instance]);
        upload.submit(gpu);

        self.instances_cpu[instance_index as usize] = mesh_instance;
    }

    /// Sets the silhouette group; zero disables outlining.
    pub fn set_outline_group(&mut self, gpu: &Gpu, instance: InstanceHandle, outline_group: u32) {
        assert!(outline_group <= u8::MAX as u32, "outline group must fit u8");
        let instance_index = self.validate_instance(instance) as u32;
        let mut mesh_instance = self.instances_cpu[instance_index as usize];
        mesh_instance.outline_group = outline_group;
        let mut upload = UploadBatch::default();
        upload.stage_slice(gpu, self.instances, instance_index, &[mesh_instance]);
        upload.submit(gpu);
        self.instances_cpu[instance_index as usize] = mesh_instance;
    }

    pub fn free(self, gpu: &Gpu) {
        if let Some(shadow) = self.shadow {
            shadow.free(gpu);
        }
        gpu.free(self.positions);
        gpu.free(self.normals);
        gpu.free(self.uvs);
        gpu.free(self.tangents);
        gpu.free(self.colors);
        gpu.free(self.joint_weights);
        gpu.free(self.index_buffer);
        gpu.free(self.mesh_table);
        gpu.free(self.mesh_data);
        gpu.free(self.mesh_bounds);
        gpu.free(self.meshlets);
        gpu.free(self.meshlet_primitive_ids);
        gpu.free(self.materials);
        gpu.free(self.transforms);
        gpu.free(self.instances);
        gpu.free(self.batches);
    }

    pub fn mesh_table_ptr(&self) -> GpuPtr<MeshTableEntry> {
        self.mesh_table.gpu
    }

    /// Raw immutable stream pools, exposed alongside the global index buffer
    /// for focused upload/readback tests and low-level consumers.
    pub fn mesh_table_buffer(&self) -> gpu::Ptr<MeshTableEntry> {
        self.mesh_table
    }

    pub fn joint_weights_buffer(&self) -> gpu::Ptr<JointWeights> {
        self.joint_weights
    }

    /// The mesh handle for a table index, for consumers iterating meshes
    /// by index (BLAS construction) rather than holding handles. Valid for
    /// indices below [`Self::mesh_slot_bound`] that are still
    /// [`live`](Self::mesh_slot_live) — the handle it synthesizes carries
    /// the slot's CURRENT generation, so it is never stale by construction.
    pub fn mesh_handle_at(&self, index: u32) -> MeshHandle {
        assert!(
            (index as usize) < self.mesh_slots.len(),
            "mesh index {index} out of range"
        );
        assert!(
            self.mesh_slots[index as usize].live,
            "mesh index {index} names a removed slot"
        );
        MeshHandle::pack(index, self.mesh_generations[index as usize])
    }

    /// Is this slot occupied? False for a slot [`Self::remove_mesh`]
    /// retired and no [`Self::add_mesh`] has yet reclaimed.
    pub fn mesh_slot_live(&self, index: u32) -> bool {
        self.mesh_slots
            .get(index as usize)
            .is_some_and(|slot| slot.live)
    }

    /// Every arena's high-water mark, in elements. Flat across a long run
    /// of edits is the whole point of reserved slots: the drain test
    /// asserts exactly this.
    pub fn arena_usage(&self) -> MeshArenaUsage {
        MeshArenaUsage {
            positions: self.position_offset,
            normals: self.normal_offset,
            uvs: self.uv_offset,
            tangents: self.tangent_offset,
            colors: self.color_offset,
            joint_weights: self.joint_weight_offset,
            indices: self.index_offset,
            meshlets: self.meshlet_offset,
            primitive_ids: self.primitive_id_offset,
            shadow_nodes: self
                .shadow
                .as_ref()
                .map_or(0, shadow::ShadowBlasPools::node_offset),
            shadow_primitives: self
                .shadow
                .as_ref()
                .map_or(0, shadow::ShadowBlasPools::primitive_offset),
        }
    }

    pub fn materials_ptr(&self) -> GpuPtr<MaterialEntry> {
        self.materials.gpu
    }
    pub fn shadow_blas(&self, mesh: MeshHandle) -> abi_light::ShadowBlas {
        let mesh_index = self.validate_mesh(mesh);
        self.shadow
            .as_ref()
            .expect("MeshScene was created without shadow BLAS storage")
            .entry(MeshHandle::pack(
                mesh_index as u32,
                self.mesh_generations[mesh_index],
            ))
    }

    pub fn shadow_blas_stats(&self, mesh: MeshHandle) -> ShadowBlasStats {
        let mesh_index = self.validate_mesh(mesh);
        self.shadow
            .as_ref()
            .expect("MeshScene was created without shadow BLAS storage")
            .stats(MeshHandle::pack(
                mesh_index as u32,
                self.mesh_generations[mesh_index],
            ))
    }

    pub fn shadow_blas_table_ptr(&self) -> GpuPtr<abi_light::ShadowBlas> {
        self.shadow
            .as_ref()
            .expect("MeshScene was created without shadow BLAS storage")
            .table_ptr()
    }

    pub fn shadow_allocated_bytes(&self) -> u64 {
        self.shadow
            .as_ref()
            .map_or(0, shadow::ShadowBlasPools::allocated_bytes)
    }
    pub fn shadow_payload_bytes(&self) -> u64 {
        self.shadow
            .as_ref()
            .map_or(0, shadow::ShadowBlasPools::payload_bytes)
    }

    /// Returns retained pools as this frame's instance stream.
    pub fn instances(&self) -> MeshInstances<'_> {
        MeshInstances {
            instances: self.instances.gpu,
            batches: self.batches.gpu,
            transforms: self.transforms.gpu,
            deformers: GpuPtr::null(),
            instances_cpu: &self.instances_cpu,
            batches_cpu: &self.batches_cpu,
            transforms_cpu: &self.transforms_cpu,
        }
    }

    pub fn mesh_data_ptr(&self) -> GpuPtr<MeshData> {
        self.mesh_data.gpu
    }

    pub fn mesh_bounds_ptr(&self) -> GpuPtr<MeshBounds> {
        self.mesh_bounds.gpu
    }

    pub fn meshlets_ptr(&self) -> GpuPtr<Meshlet> {
        self.meshlets.gpu
    }

    pub fn global_index_buffer_ptr(&self) -> GpuPtr<u32> {
        self.index_buffer.gpu
    }

    pub fn global_index_buffer(&self) -> gpu::Ptr<u32> {
        self.index_buffer
    }

    pub fn mesh_data(&self, mesh: MeshHandle) -> MeshData {
        self.mesh_data_cpu[self.validate_mesh(mesh)]
    }

    pub fn mesh_data_cpu(&self) -> &[MeshData] {
        &self.mesh_data_cpu
    }

    pub fn mesh_vertex_count(&self, mesh: MeshHandle) -> u32 {
        self.mesh_vertex_counts_cpu[self.validate_mesh(mesh)]
    }

    pub fn meshlet_index_count(&self, mesh: MeshHandle) -> u32 {
        self.meshlet_index_counts_cpu[self.validate_mesh(mesh)]
    }

    /// Returns the number of live meshes, not the slot-table length.
    pub fn mesh_count(&self) -> u32 {
        self.mesh_live_count
    }

    /// Returns the slot-table length, including holes.
    pub fn mesh_slot_bound(&self) -> u32 {
        self.mesh_slots.len() as u32
    }

    pub fn instance_count(&self) -> u32 {
        self.instance_generations.len() as u32
    }

    pub fn material_count(&self) -> u32 {
        self.material_generations.len() as u32
    }

    pub fn max_meshlets_per_mesh(&self) -> u32 {
        self.max_meshlets_per_mesh
    }

    pub fn meshlet_range(&self, mesh: MeshHandle) -> Range<usize> {
        let data = self.mesh_data(mesh);
        data.meshlet_offset as usize..(data.meshlet_offset + data.meshlet_count) as usize
    }

    pub fn primitive_id_range(&self, meshlet_index: usize) -> Range<usize> {
        let meshlet = self.meshlets_cpu[meshlet_index];
        meshlet.first_primitive as usize..(meshlet.first_primitive + meshlet.tri_count) as usize
    }

    pub fn indices_cpu(&self) -> &[u32] {
        &self.indices_cpu
    }

    pub fn meshlets_cpu(&self) -> &[Meshlet] {
        &self.meshlets_cpu
    }

    pub fn meshlet_vertex_counts_cpu(&self) -> &[u32] {
        &self.meshlet_vertex_counts_cpu
    }

    pub fn meshlet_primitive_ids_cpu(&self) -> &[u32] {
        &self.meshlet_primitive_ids_cpu
    }

    pub fn transforms_cpu(&self) -> &[DrawTransform] {
        &self.transforms_cpu
    }

    pub fn instances_cpu(&self) -> &[MeshInstance] {
        &self.instances_cpu
    }

    pub fn batches_cpu(&self) -> &[MeshBatch] {
        &self.batches_cpu
    }

    pub fn batch_count(&self) -> u32 {
        self.batches_cpu.len() as u32
    }

    pub fn materials_cpu(&self) -> &[MaterialEntry] {
        &self.materials_cpu
    }

    /// Maximum compacted-cluster table length for the retained stream.
    pub fn cluster_capacity(&self) -> u32 {
        self.batches_cpu
            .iter()
            .map(|batch| batch.cluster_capacity)
            .sum()
    }

    fn find_or_add_batch(&mut self, mesh_index: u32, material_index: u32) -> u32 {
        if let Some((index, _)) = self.batches_cpu.iter().enumerate().find(|(_, batch)| {
            batch.mesh_index == mesh_index && batch.material_index == material_index
        }) {
            return index as u32;
        }
        let index = self.batches_cpu.len() as u32;
        assert_capacity(index, 1, self.desc.max_instances, "max_instances (batches)");
        self.batches_cpu.push(MeshBatch {
            mesh_index,
            material_index,
            cluster_base: 0,
            cluster_capacity: 0,
        });
        index
    }

    /// Rebuilds affected retained ranges after registration or mesh updates.
    fn rebuild_batch_ranges_for(&mut self, gpu: &Gpu, mesh_index: u32) {
        if self
            .batches_cpu
            .iter()
            .any(|batch| batch.mesh_index == mesh_index)
        {
            self.rebuild_batch_ranges(gpu);
        }
    }

    fn rebuild_batch_ranges(&mut self, gpu: &Gpu) {
        let mut cluster_base = 0u32;
        for (batch_index, batch) in self.batches_cpu.iter_mut().enumerate() {
            let mut cluster_capacity = 0u32;
            for instance in &self.instances_cpu {
                if instance.batch_index == batch_index as u32 {
                    cluster_capacity = cluster_capacity
                        .checked_add(self.mesh_data_cpu[batch.mesh_index as usize].meshlet_count)
                        .expect("retained cluster capacity exceeds u32");
                }
            }
            batch.cluster_base = cluster_base;
            batch.cluster_capacity = cluster_capacity;
            cluster_base = cluster_base
                .checked_add(cluster_capacity)
                .expect("retained cluster base exceeds u32");
        }
        assert!(
            cluster_base
                <= self
                    .desc
                    .max_meshlets
                    .saturating_mul(self.desc.max_instances),
            "retained cluster capacity unexpectedly exceeds declared scene ceiling"
        );
        let mut upload = UploadBatch::default();
        upload.stage_slice(gpu, self.batches, 0, &self.batches_cpu);
        upload.submit(gpu);
    }

    fn validate_mesh(&self, handle: MeshHandle) -> usize {
        validate_handle(handle.raw(), &self.mesh_generations, "mesh")
    }

    fn validate_material(&self, handle: MaterialHandle) -> usize {
        validate_handle(handle.raw(), &self.material_generations, "material")
    }

    fn validate_instance(&self, handle: InstanceHandle) -> usize {
        validate_handle(handle.raw(), &self.instance_generations, "instance")
    }
}

fn validate_handle(raw: u64, generations: &[u32], name: &str) -> usize {
    assert!(raw != 0, "{name} handle 0 is invalid");
    let index = raw as u32 as usize;
    let generation = (raw >> 32) as u32;
    assert!(
        generations.get(index).copied() == Some(generation),
        "invalid {name} handle"
    );
    index
}

fn validate_mesh_desc(desc: MeshDesc<'_>) {
    assert!(
        !desc.positions.is_empty(),
        "mesh positions must be non-empty"
    );
    assert_eq!(
        desc.normals.len(),
        desc.positions.len(),
        "mesh normals length must match positions"
    );
    assert_eq!(
        desc.uvs.len(),
        desc.positions.len(),
        "mesh uvs length must match positions"
    );
    if let Some(tangents) = desc.tangents {
        assert_eq!(
            tangents.len(),
            desc.positions.len(),
            "mesh tangents length must match positions"
        );
    }
    if let Some(colors) = desc.colors {
        assert_eq!(
            colors.len(),
            desc.positions.len(),
            "mesh colors length must match positions"
        );
        for (i, color) in colors.iter().enumerate() {
            assert!(
                color.iter().all(|v| v.is_finite()),
                "mesh colors[{i}] must be finite"
            );
        }
    }
    if let Some(joint_weights) = desc.joint_weights {
        assert_eq!(
            joint_weights.len(),
            desc.positions.len(),
            "mesh joint weights length must match positions"
        );
        for (vertex, joint_weights) in joint_weights.iter().enumerate() {
            let mut sum = 0.0f32;
            for (influence, &weight) in joint_weights.weights.iter().enumerate() {
                assert!(
                    weight.is_finite(),
                    "mesh joint weights[{vertex}][{influence}] must be finite"
                );
                assert!(
                    (0.0..=1.0).contains(&weight),
                    "mesh joint weights[{vertex}][{influence}] must be in [0, 1]"
                );
                sum += weight;
            }
            assert!(
                (sum - 1.0).abs() <= 1.0e-4,
                "mesh joint weights[{vertex}] sum must be within 1e-4 of 1 (got {sum})"
            );
        }
    }
    assert!(!desc.indices.is_empty(), "mesh indices must be non-empty");
    assert!(
        desc.indices.len() % 3 == 0,
        "mesh indices length must be divisible by 3"
    );
    for (i, position) in desc.positions.iter().enumerate() {
        assert!(
            position.iter().all(|v| v.is_finite()),
            "mesh positions[{i}] must be finite"
        );
    }
    for (i, normal) in desc.normals.iter().enumerate() {
        assert!(
            normal.iter().all(|v| v.is_finite()),
            "mesh normals[{i}] must be finite"
        );
    }
    for (i, uv) in desc.uvs.iter().enumerate() {
        assert!(
            uv.iter().all(|v| v.is_finite()),
            "mesh uvs[{i}] must be finite"
        );
    }
    for (i, &index) in desc.indices.iter().enumerate() {
        assert!(
            (index as usize) < desc.positions.len(),
            "mesh indices[{i}] is out of range"
        );
    }
}

fn prepare_mesh(desc: MeshDesc<'_>, mesh_index: u32) -> PreparedMesh {
    let positions = desc
        .positions
        .iter()
        .map(|p| [p[0], p[1], p[2], 0.0])
        .collect::<Vec<_>>();
    let normals = desc
        .normals
        .iter()
        .map(|n| [n[0], n[1], n[2], 0.0])
        .collect::<Vec<_>>();
    let uvs = desc.uvs.to_vec();
    let tangents = desc.tangents.map(|tangents| tangents.to_vec());
    let colors = desc.colors.map(|colors| colors.to_vec());
    let bounds = mesh_bounds(desc.positions);

    let optimized_indices = meshopt::optimize_vertex_cache(desc.indices, desc.positions.len());
    let vertex_adapter = meshopt::VertexDataAdapter::new(
        meshopt::typed_to_bytes(desc.positions),
        size_of::<[f32; 3]>(),
        0,
    )
    .expect("tight vec3 positions satisfy meshopt VertexDataAdapter");
    let meshlets = meshopt::build_meshlets(
        &optimized_indices,
        &vertex_adapter,
        MESHLET_MAX_VERTICES,
        MESHLET_MAX_TRIANGLES,
        MESHLET_CONE_WEIGHT,
    );
    assert!(
        !meshlets.is_empty(),
        "registered meshes must produce at least one meshlet"
    );

    let mut primitive_lookup = primitive_lookup(desc.indices);
    let mut primitive_coverage = vec![0u8; desc.indices.len() / 3];
    let mut global_indices = desc.indices.to_vec();
    let mut out_meshlets = Vec::with_capacity(meshlets.len());
    let mut meshlet_vertex_counts = Vec::with_capacity(meshlets.len());
    let mut primitive_ids = Vec::with_capacity(desc.indices.len() / 3);
    let mut meshlet_index_count = 0u32;

    for meshlet in meshlets.iter() {
        let bounds = meshopt::compute_meshlet_bounds(meshlet, &vertex_adapter);
        let first_index = global_indices.len() as u32;
        let first_primitive = primitive_ids.len() as u32;
        let tri_count = (meshlet.triangles.len() / 3) as u32;

        assert!(
            (1..=MESHLET_MAX_TRIANGLES as u32).contains(&tri_count),
            "meshlet tri_count exceeds 124"
        );
        assert!(
            meshlet.vertices.len() <= MESHLET_MAX_VERTICES,
            "meshlet vertex count exceeds 64"
        );

        for triangle in meshlet.triangles.chunks_exact(3) {
            let source = [
                meshlet.vertices[triangle[0] as usize],
                meshlet.vertices[triangle[1] as usize],
                meshlet.vertices[triangle[2] as usize],
            ];
            let original = primitive_lookup
                .get_mut(&source)
                .and_then(VecDeque::pop_front)
                .unwrap_or_else(|| panic!("meshlet primitive {:?} has no source triangle", source));
            primitive_coverage[original as usize] += 1;
            primitive_ids.push(original);
            global_indices.extend_from_slice(&source);
        }

        out_meshlets.push(Meshlet {
            mesh_index,
            first_index,
            tri_count,
            first_primitive,
            center: bounds.center,
            radius: bounds.radius,
            cone_axis: bounds.cone_axis,
            cone_cutoff: bounds.cone_cutoff,
        });
        meshlet_vertex_counts.push(meshlet.vertices.len() as u32);
        meshlet_index_count += tri_count * 3;
    }

    for (tri, count) in primitive_coverage.iter().enumerate() {
        assert_eq!(
            *count, 1,
            "meshlet primitive remap must cover source triangle {tri} exactly once"
        );
    }
    assert!(
        primitive_lookup.values().all(VecDeque::is_empty),
        "meshlet primitive remap left source triangles uncovered"
    );

    PreparedMesh {
        positions,
        normals,
        uvs,
        tangents,
        colors,
        indices: global_indices,
        meshlets: out_meshlets,
        meshlet_vertex_counts,
        primitive_ids,
        meshlet_index_count,
        bounds,
    }
}

fn primitive_lookup(indices: &[u32]) -> HashMap<[u32; 3], VecDeque<u32>> {
    let mut lookup = HashMap::<[u32; 3], VecDeque<u32>>::new();
    for (tri, chunk) in indices.chunks_exact(3).enumerate() {
        lookup
            .entry([chunk[0], chunk[1], chunk[2]])
            .or_default()
            .push_back(tri as u32);
    }
    lookup
}

fn rebase_meshlets(meshlets: &mut [Meshlet], first_index: u32, first_primitive: u32) {
    for meshlet in meshlets {
        meshlet.first_index += first_index;
        meshlet.first_primitive += first_primitive;
    }
}

fn mesh_bounds(positions: &[[f32; 3]]) -> MeshBounds {
    let mut min = positions[0];
    let mut max = positions[0];
    for position in &positions[1..] {
        min[0] = min[0].min(position[0]);
        min[1] = min[1].min(position[1]);
        min[2] = min[2].min(position[2]);
        max[0] = max[0].max(position[0]);
        max[1] = max[1].max(position[1]);
        max[2] = max[2].max(position[2]);
    }
    MeshBounds {
        aabb_min: min,
        _pad0: 0.0,
        aabb_max: max,
        _pad1: 0.0,
    }
}

fn assert_capacity(used: u32, add: u32, capacity: u32, name: &str) {
    assert!(
        add <= capacity.saturating_sub(used),
        "{name} capacity exceeded"
    );
}

fn to_u32(value: usize, capacity_name: &str) -> u32 {
    assert!(
        value <= u32::MAX as usize,
        "{capacity_name} capacity exceeded"
    );
    value as u32
}
