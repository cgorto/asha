//! Immutable mesh CWBVH storage and shadow-query passes.
//!
//! Host-built nodes use the stable GPU ABI and original mesh positions.

use std::time::{Duration, Instant};

use abi_core::GpuPtr;
use abi_light::{
    CwbvhNode, ShadowBlas, ShadowBlasQueryData, ShadowQueryResult, ShadowSegment,
    ShadowTlasInstance, ShadowTlasNode, ShadowWorld, ShadowWorldQueryData, ShadowWorldQueryResult,
};
use abi_mesh::{
    MESH_FLAG_DYNAMIC, MESH_FLAG_HIDDEN, MESH_FLAG_NO_SHADOW, MESH_FLAG_SKINNED, cull_world_aabb,
};
use gpu::pass::{FrameAlloc, Pass};
use gpu::{CommandBuffer, Gpu, HazardFlags, Memory, Stage};
use obvhs::{
    BvhBuildParams, aabb::Aabb, bvh2::Bvh2, bvh2::builder::build_bvh2, cwbvh::builder::build_cwbvh,
};

use crate::{MeshDesc, MeshHandle, MeshInstances, MeshScene, UploadBatch};

pub const SHADOW_BLAS_STACK_CAPACITY: u32 = 32;
pub const SHADOW_TLAS_STACK_CAPACITY: u32 = 32;

#[derive(Debug, Clone, Copy, Default)]
pub struct ShadowBlasDesc {
    pub node_capacity: u32,
    pub primitive_capacity: u32,
}

impl ShadowBlasDesc {
    fn assert_valid(self) {
        assert!(
            self.node_capacity > 0 && self.primitive_capacity > 0,
            "shadow BLAS capacities must both be positive"
        );
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ShadowBlasStats {
    pub node_count: u32,
    pub primitive_count: u32,
    pub max_depth: u32,
    pub build_time: Duration,
}

pub(crate) struct PreparedShadowBlas {
    nodes: Vec<CwbvhNode>,
    primitive_ids: Vec<u32>,
    stats: ShadowBlasStats,
}

pub(crate) struct ShadowBlasPools {
    nodes: gpu::Ptr<CwbvhNode>,
    primitive_ids: gpu::Ptr<u32>,
    table: gpu::Ptr<ShadowBlas>,
    node_capacity: u32,
    primitive_capacity: u32,
    table_capacity: u32,
    node_offset: u32,
    primitive_offset: u32,
    /// Per-slot BLAS reservations reused across rewrites.
    slots: Vec<ShadowSlot>,
    entries_cpu: Vec<ShadowBlas>,
    stats_cpu: Vec<ShadowBlasStats>,
}

#[derive(Debug, Clone, Copy, Default)]
struct ShadowSlot {
    nodes: crate::ArenaSlot,
    primitives: crate::ArenaSlot,
}

impl ShadowBlasPools {
    pub(crate) fn new(gpu: &Gpu, max_meshes: u32, desc: ShadowBlasDesc) -> Self {
        desc.assert_valid();
        assert!(max_meshes > 0, "shadow BLAS table needs max_meshes > 0");
        Self {
            nodes: gpu.alloc_slice(desc.node_capacity as u64, Memory::Gpu),
            primitive_ids: gpu.alloc_slice(desc.primitive_capacity as u64, Memory::Gpu),
            table: gpu.alloc_slice(max_meshes as u64, Memory::Gpu),
            node_capacity: desc.node_capacity,
            primitive_capacity: desc.primitive_capacity,
            table_capacity: max_meshes,
            node_offset: 0,
            primitive_offset: 0,
            slots: Vec::with_capacity(max_meshes as usize),
            entries_cpu: Vec::with_capacity(max_meshes as usize),
            stats_cpu: Vec::with_capacity(max_meshes as usize),
        }
    }

    pub(crate) fn prepare(desc: MeshDesc<'_>) -> PreparedShadowBlas {
        let build_started = Instant::now();
        let mut aabbs = Vec::with_capacity(desc.indices.len() / 3);
        for triangle in desc.indices.chunks_exact(3) {
            let v0 = desc.positions[triangle[0] as usize];
            let v1 = desc.positions[triangle[1] as usize];
            let v2 = desc.positions[triangle[2] as usize];
            let min = [
                v0[0].min(v1[0]).min(v2[0]),
                v0[1].min(v1[1]).min(v2[1]),
                v0[2].min(v1[2]).min(v2[2]),
            ];
            let max = [
                v0[0].max(v1[0]).max(v2[0]),
                v0[1].max(v1[1]).max(v2[1]),
                v0[2].max(v1[2]).max(v2[2]),
            ];
            aabbs.push(Aabb::new(min.into(), max.into()));
        }

        let mut core_build_time = Duration::ZERO;
        let bvh = build_cwbvh(&aabbs, BvhBuildParams::medium_build(), &mut core_build_time);
        assert!(!bvh.uses_spatial_splits, "BLASes forbid spatial splits");
        let validation = bvh.validate(&aabbs, false);
        assert!(
            validation.max_depth < SHADOW_BLAS_STACK_CAPACITY,
            "CWBVH depth {} exceeds the {}-entry traversal stack",
            validation.max_depth,
            SHADOW_BLAS_STACK_CAPACITY
        );
        assert_eq!(bvh.primitive_indices.len(), aabbs.len());
        assert_eq!(validation.prim_count, aabbs.len());

        let node_count = u32::try_from(bvh.nodes.len()).expect("CWBVH node count exceeds u32");
        let primitive_count =
            u32::try_from(bvh.primitive_indices.len()).expect("CWBVH primitive count exceeds u32");
        PreparedShadowBlas {
            nodes: bvh.nodes.iter().map(pack_cwbvh_node).collect(),
            primitive_ids: bvh.primitive_indices,
            stats: ShadowBlasStats {
                node_count,
                primitive_count,
                max_depth: validation.max_depth,
                build_time: build_started.elapsed(),
            },
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn stage(
        &mut self,
        gpu: &Gpu,
        upload: &mut UploadBatch,
        mesh_index: u32,
        positions: abi_core::GpuPtr<[f32; 4]>,
        indices: abi_core::GpuPtr<u32>,
        prepared: PreparedShadowBlas,
    ) {
        assert!(
            (mesh_index as usize) < self.table_capacity as usize,
            "shadow BLAS table index {mesh_index} exceeds max_meshes"
        );
        if self.slots.len() <= mesh_index as usize {
            self.slots
                .resize(mesh_index as usize + 1, ShadowSlot::default());
        }
        let mut slot = self.slots[mesh_index as usize];
        crate::reserve(
            &mut slot.nodes,
            &mut self.node_offset,
            prepared.stats.node_count,
            self.node_capacity,
            "shadow node",
        );
        crate::reserve(
            &mut slot.primitives,
            &mut self.primitive_offset,
            prepared.stats.primitive_count,
            self.primitive_capacity,
            "shadow primitive",
        );
        self.slots[mesh_index as usize] = slot;

        let entry = ShadowBlas {
            nodes: self.nodes.gpu.offset(slot.nodes.base as i64),
            primitive_ids: self.primitive_ids.gpu.offset(slot.primitives.base as i64),
            positions,
            indices,
            node_count: prepared.stats.node_count,
            primitive_count: prepared.stats.primitive_count,
            _pad0: [0; 2],
        };
        upload.stage_slice(gpu, self.nodes, slot.nodes.base, &prepared.nodes);
        upload.stage_slice(
            gpu,
            self.primitive_ids,
            slot.primitives.base,
            &prepared.primitive_ids,
        );
        upload.stage_slice(gpu, self.table, mesh_index, &[entry]);

        crate::set_row(&mut self.entries_cpu, mesh_index, entry);
        crate::set_row(&mut self.stats_cpu, mesh_index, prepared.stats);
    }

    pub(crate) fn entry(&self, mesh: MeshHandle) -> ShadowBlas {
        self.entries_cpu[mesh.index() as usize]
    }

    pub(crate) fn stats(&self, mesh: MeshHandle) -> ShadowBlasStats {
        self.stats_cpu[mesh.index() as usize]
    }

    pub(crate) fn node_offset(&self) -> u32 {
        self.node_offset
    }

    pub(crate) fn primitive_offset(&self) -> u32 {
        self.primitive_offset
    }

    pub(crate) fn table_ptr(&self) -> abi_core::GpuPtr<ShadowBlas> {
        self.table.gpu
    }

    pub(crate) fn allocated_bytes(&self) -> u64 {
        u64::from(self.node_capacity) * core::mem::size_of::<CwbvhNode>() as u64
            + u64::from(self.primitive_capacity) * core::mem::size_of::<u32>() as u64
            + u64::from(self.table_capacity) * core::mem::size_of::<ShadowBlas>() as u64
    }

    pub(crate) fn payload_bytes(&self) -> u64 {
        u64::from(self.node_offset) * core::mem::size_of::<CwbvhNode>() as u64
            + u64::from(self.primitive_offset) * core::mem::size_of::<u32>() as u64
            + self.slots.len() as u64 * core::mem::size_of::<ShadowBlas>() as u64
    }

    pub(crate) fn free(self, gpu: &Gpu) {
        gpu.free(self.nodes);
        gpu.free(self.primitive_ids);
        gpu.free(self.table);
    }
}
#[derive(Debug, Clone, Copy, Default)]
pub struct ShadowTlasStats {
    pub node_count: u32,
    pub instance_count: u32,
    pub max_depth: u32,
    pub topology_rebuilt: bool,
    pub update_time: Duration,
}

/// Reusable host TLAS with topology rebuilds and in-place refits.
pub struct ShadowTlasBuilder {
    bvh: Bvh2,
    instance_ids: Vec<u32>,
    instance_ids_scratch: Vec<u32>,
    aabbs: Vec<Aabb>,
    nodes: Vec<ShadowTlasNode>,
    instances: Vec<ShadowTlasInstance>,
    max_depth: u32,
}

impl ShadowTlasBuilder {
    pub fn new(max_instances: u32) -> Self {
        let instance_capacity = max_instances as usize;
        let node_capacity = instance_capacity.saturating_mul(2).saturating_sub(1);
        Self {
            bvh: Bvh2::default(),
            instance_ids: Vec::with_capacity(instance_capacity),
            instance_ids_scratch: Vec::with_capacity(instance_capacity),
            aabbs: Vec::with_capacity(instance_capacity),
            nodes: Vec::with_capacity(node_capacity),
            instances: Vec::with_capacity(instance_capacity),
            max_depth: 0,
        }
    }

    pub fn build(
        &mut self,
        fa: &mut impl FrameAlloc,
        scene: &MeshScene,
    ) -> (ShadowWorld, ShadowTlasStats) {
        self.build_instances(fa, scene, scene.instances())
    }

    /// Builds or refits from the raster pass's instance stream.
    pub fn build_instances(
        &mut self,
        fa: &mut impl FrameAlloc,
        scene: &MeshScene,
        instances: MeshInstances<'_>,
    ) -> (ShadowWorld, ShadowTlasStats) {
        let started = Instant::now();
        self.collect_instances(scene, instances);
        let topology_rebuilt = self.instance_ids_scratch != self.instance_ids;
        if topology_rebuilt {
            core::mem::swap(&mut self.instance_ids, &mut self.instance_ids_scratch);
            self.rebuild_topology();
        } else {
            self.refit();
        }
        self.pack(instances);

        let world = ShadowWorld {
            nodes: fa.frame_alloc_slice(&self.nodes),
            instances: fa.frame_alloc_slice(&self.instances),
            blases: scene.shadow_blas_table_ptr(),
            node_count: self.nodes.len() as u32,
            instance_count: self.instances.len() as u32,
            _pad0: [0; 2],
        };
        let stats = ShadowTlasStats {
            node_count: world.node_count,
            instance_count: world.instance_count,
            max_depth: self.max_depth,
            topology_rebuilt,
            update_time: started.elapsed(),
        };
        (world, stats)
    }

    fn collect_instances(&mut self, scene: &MeshScene, instances: MeshInstances<'_>) {
        self.instance_ids_scratch.clear();
        self.aabbs.clear();
        for (instance_id, instance) in instances.instances_cpu.iter().enumerate() {
            if instance.flags & (MESH_FLAG_HIDDEN | MESH_FLAG_NO_SHADOW) != 0 {
                continue;
            }
            assert!(
                instance.flags & MESH_FLAG_SKINNED == 0 && instance.deformer_slot == 0,
                "instance {instance_id} has deformation but no authoritative shadow proxy"
            );
            let batch = instances.batches_cpu[instance.batch_index as usize];
            let transform = instances.transforms_cpu[instance.transform_index as usize];
            let bounds = scene.mesh_bounds_cpu[batch.mesh_index as usize];
            let (min, max) =
                cull_world_aabb(&bounds, &transform.model_to_world, instance.bounds_dilation);
            assert!(
                min.is_finite() && max.is_finite() && min.cmple(max).all(),
                "instance {instance_id} produced invalid shadow bounds"
            );
            self.instance_ids_scratch.push(instance_id as u32);
            self.aabbs
                .push(Aabb::new(min.to_array().into(), max.to_array().into()));
        }
    }

    fn rebuild_topology(&mut self) {
        if self.aabbs.is_empty() {
            self.bvh = Bvh2::default();
            self.max_depth = 0;
            return;
        }
        let mut params = BvhBuildParams::fastest_build();
        params.max_prims_per_leaf = 1;
        params.pre_split = false;
        let mut core_build_time = Duration::ZERO;
        self.bvh = build_bvh2(&self.aabbs, params, &mut core_build_time);
        self.bvh.reorder_in_stack_traversal_order();
        self.bvh.init_primitives_to_nodes_if_uninit();
        let validation = self.bvh.validate(&self.aabbs, false, true);
        assert_eq!(validation.prim_count, self.aabbs.len());
        assert!(
            validation.max_depth < SHADOW_TLAS_STACK_CAPACITY,
            "TLAS depth {} exceeds the {}-entry traversal stack",
            validation.max_depth,
            SHADOW_TLAS_STACK_CAPACITY
        );
        self.max_depth = validation.max_depth;
    }

    fn refit(&mut self) {
        if self.aabbs.is_empty() {
            return;
        }
        assert_eq!(self.bvh.primitives_to_nodes.len(), self.aabbs.len());
        for (primitive_id, aabb) in self.aabbs.iter().copied().enumerate() {
            let node_id = self.bvh.primitives_to_nodes[primitive_id] as usize;
            self.bvh.nodes[node_id].set_aabb(aabb);
        }
        self.bvh.refit_all();
    }

    fn pack(&mut self, stream: MeshInstances<'_>) {
        self.nodes.clear();
        self.nodes.extend(self.bvh.nodes.iter().map(|node| {
            let aabb = node.aabb();
            let (child_or_instance, leaf) = if node.is_leaf() {
                assert_eq!(node.prim_count, 1, "TLAS leaves contain one instance");
                let primitive_id = self.bvh.primitive_indices[node.first_index as usize];
                assert!((primitive_id as usize) < self.instance_ids.len());
                (primitive_id, 1)
            } else {
                assert!((node.first_index as usize + 1) < self.bvh.nodes.len());
                (node.first_index, 0)
            };
            ShadowTlasNode {
                min: aabb.min.to_array(),
                child_or_instance,
                max: aabb.max.to_array(),
                leaf,
            }
        }));

        self.instances.clear();
        self.instances
            .extend(self.instance_ids.iter().copied().map(|instance_id| {
                let instance = stream.instances_cpu[instance_id as usize];
                let batch = stream.batches_cpu[instance.batch_index as usize];
                let transform = stream.transforms_cpu[instance.transform_index as usize];
                ShadowTlasInstance {
                    world_to_local: transform.model_to_world.inverse(),
                    blas_index: batch.mesh_index,
                    instance_id,
                    flags: u32::from(instance.flags & MESH_FLAG_DYNAMIC != 0),
                    _pad0: 0,
                }
            }));
    }
}

/// GPU world-space queries through a refitted TLAS and immutable BLASes.
pub struct ShadowWorldQueryPass {
    shader: gpu::Shader,
}

impl ShadowWorldQueryPass {
    pub const GROUP_SIZE: u32 = 64;

    pub fn new(gpu: &Gpu) -> Self {
        Self {
            shader: gpu.shader_create_compute(
                &asha_assets::load_spv("shadow_world_queries"),
                Self::GROUP_SIZE,
                1,
                1,
                "shadow_world_queries",
            ),
        }
    }

    pub fn record(
        &self,
        gpu: &Gpu,
        cb: CommandBuffer,
        fa: &mut impl FrameAlloc,
        world: ShadowWorld,
        queries: GpuPtr<ShadowSegment>,
        results: GpuPtr<ShadowWorldQueryResult>,
        query_count: u32,
    ) {
        if query_count == 0 {
            return;
        }
        assert!(!queries.is_null(), "shadow world queries pointer is null");
        assert!(!results.is_null(), "shadow world results pointer is null");
        assert!(!world.blases.is_null(), "shadow world BLAS table is null");
        assert_eq!(
            world.nodes.is_null(),
            world.node_count == 0,
            "shadow world node pointer/count disagree"
        );
        assert_eq!(
            world.instances.is_null(),
            world.instance_count == 0,
            "shadow world instance pointer/count disagree"
        );

        let data = fa.frame_alloc(ShadowWorldQueryData {
            world,
            queries,
            results,
            query_count,
            _pad0: [0; 3],
        });
        gpu.cmd_set_compute_shader(cb, self.shader);
        gpu.cmd_dispatch(cb, data, query_count.div_ceil(Self::GROUP_SIZE), 1, 1);
        gpu.cmd_barrier(cb, Stage::Compute, Stage::All, HazardFlags::empty());
    }
}

impl Pass for ShadowWorldQueryPass {
    const NAME: &'static str = "shadow_world_queries";

    fn free(self, gpu: &Gpu) {
        gpu.shader_destroy(self.shader);
    }
}

/// GPU queries against one immutable mesh BLAS.
pub struct ShadowBlasQueryPass {
    shader: gpu::Shader,
}

impl ShadowBlasQueryPass {
    pub const GROUP_SIZE: u32 = 64;

    pub fn new(gpu: &Gpu) -> Self {
        Self {
            shader: gpu.shader_create_compute(
                &asha_assets::load_spv("shadow_blas_queries"),
                Self::GROUP_SIZE,
                1,
                1,
                "shadow_blas_queries",
            ),
        }
    }

    pub fn record(
        &self,
        gpu: &Gpu,
        cb: CommandBuffer,
        fa: &mut impl FrameAlloc,
        blas: ShadowBlas,
        queries: GpuPtr<ShadowSegment>,
        results: GpuPtr<ShadowQueryResult>,
        query_count: u32,
    ) {
        if query_count == 0 {
            return;
        }
        assert!(!queries.is_null(), "shadow queries pointer is null");
        assert!(!results.is_null(), "shadow results pointer is null");
        assert!(
            !blas.nodes.is_null()
                && !blas.primitive_ids.is_null()
                && !blas.positions.is_null()
                && !blas.indices.is_null(),
            "standalone shadow query needs a complete BLAS"
        );
        let data = fa.frame_alloc(ShadowBlasQueryData {
            blas,
            queries,
            results,
            query_count,
            _pad0: [0; 3],
        });
        gpu.cmd_set_compute_shader(cb, self.shader);
        gpu.cmd_dispatch(cb, data, query_count.div_ceil(Self::GROUP_SIZE), 1, 1);
        gpu.cmd_barrier(cb, Stage::Compute, Stage::All, HazardFlags::empty());
    }
}

impl Pass for ShadowBlasQueryPass {
    const NAME: &'static str = "shadow_blas_queries";

    fn free(self, gpu: &Gpu) {
        gpu.shader_destroy(self.shader);
    }
}

fn put_u32(bytes: &mut [u8; 80], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn pack_cwbvh_node(node: &obvhs::cwbvh::node::CwBvhNode) -> CwbvhNode {
    let mut bytes = [0u8; 80];
    for (axis, value) in node.p.to_array().into_iter().enumerate() {
        put_u32(&mut bytes, axis * 4, value.to_bits());
    }
    bytes[12..15].copy_from_slice(&node.e);
    bytes[15] = node.imask;
    put_u32(&mut bytes, 16, node.child_base_idx);
    put_u32(&mut bytes, 20, node.primitive_base_idx);
    bytes[24..32].copy_from_slice(&node.child_meta);
    bytes[32..40].copy_from_slice(&node.child_min_x);
    bytes[40..48].copy_from_slice(&node.child_max_x);
    bytes[48..56].copy_from_slice(&node.child_min_y);
    bytes[56..64].copy_from_slice(&node.child_max_y);
    bytes[64..72].copy_from_slice(&node.child_min_z);
    bytes[72..80].copy_from_slice(&node.child_max_z);

    CwbvhNode {
        words: core::array::from_fn(|i| {
            u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap())
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::cube;

    #[test]
    fn explicit_node_pack_matches_obvhs_032_layout() {
        let node = obvhs::cwbvh::node::CwBvhNode {
            p: [1.25, -2.5, 9.0].into(),
            e: [11, 22, 33],
            imask: 0xA5,
            child_base_idx: 0x1122_3344,
            primitive_base_idx: 0x5566_7788,
            child_meta: [1, 2, 3, 4, 5, 6, 7, 8],
            child_min_x: [9; 8],
            child_max_x: [10; 8],
            child_min_y: [11; 8],
            child_max_y: [12; 8],
            child_min_z: [13; 8],
            child_max_z: [14; 8],
        };
        let raw = bytemuck::bytes_of(&node);
        let expected =
            core::array::from_fn(|i| u32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap()));
        assert_eq!(pack_cwbvh_node(&node).words, expected);
    }

    #[test]
    fn cube_build_is_valid_and_remaps_every_source_triangle() {
        let cube = cube(1.0);
        let built = ShadowBlasPools::prepare(cube.desc());
        assert!(!built.nodes.is_empty());
        assert_eq!(built.primitive_ids.len(), 12);
        assert!(built.stats.max_depth < SHADOW_BLAS_STACK_CAPACITY);
        let mut ids = built.primitive_ids.clone();
        ids.sort_unstable();
        assert_eq!(ids, (0..12).collect::<Vec<_>>());
    }
}
