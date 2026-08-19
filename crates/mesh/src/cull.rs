//! GPU culling and indirect-command generation for mesh clusters.
//!
//! Counter slots rotate under the render-frame ownership contract.

use abi_core::GpuPtr;
use abi_core::glam::Vec3;
use abi_mesh::{
    ClusterCullData, ClusterInstance, IndirectData, MESH_FLAG_HIDDEN, MeshBatch,
    extract_frustum_planes,
};
use gpu::{CommandBuffer, Gpu, HazardFlags, Memory, Stage};

use crate::{FrameAlloc, MeshInstances, MeshRasterView, MeshScene};

/// Backface cone-cull tolerance, shared with CPU verification.
pub const CONE_CULL_EPSILON: f32 = 1.0e-4;

/// Cluster culling with persistent outputs and rotating count storage.
pub struct ClusterCullPass {
    cull_shader: gpu::Shader,
    args_shader: gpu::Shader,
    clusters: gpu::Ptr<ClusterInstance>,
    indirect: gpu::Ptr<IndirectData>,
    /// Per-frame-slot visible counts, laid out contiguously.
    visible_counts: gpu::Ptr<u32>,
    /// Candidate batches; empty batches use zero-instance commands.
    draw_counts: gpu::Ptr<u32>,
    in_flight: usize,
    max_clusters: u32,
    max_batches: u32,
}

impl ClusterCullPass {
    /// Retained constructor; snapshots the scene's current capacities.
    /// Populate the scene fully before construction; streams use `with_capacity`.
    pub fn new(gpu: &Gpu, scene: &MeshScene, in_flight: usize) -> Self {
        Self::with_capacity(
            gpu,
            scene.cluster_capacity(),
            scene.batch_count(),
            in_flight,
        )
    }

    /// Creates a culler for streamed frame data.
    /// Capacities must cover each frame's full population; `record` checks mirrors.
    pub fn with_capacity(gpu: &Gpu, max_clusters: u32, max_batches: u32, in_flight: usize) -> Self {
        assert!(in_flight >= 1);
        assert!(
            max_clusters > 0,
            "cluster cull needs nonzero cluster capacity"
        );
        assert!(
            max_clusters < (1 << 25),
            "cluster cull capacity {max_clusters} exceeds the 25-bit visibility-token range"
        );
        assert!(max_batches > 0, "cluster cull needs nonzero batch capacity");
        Self {
            cull_shader: gpu.shader_create_compute(
                &asha_assets::load_spv("cluster_cull"),
                64,
                4,
                1,
                "cluster_cull",
            ),
            args_shader: gpu.shader_create_compute(
                &asha_assets::load_spv("cluster_build_args"),
                64,
                1,
                1,
                "cluster_build_args",
            ),
            clusters: gpu.alloc_slice::<ClusterInstance>(max_clusters as u64, Memory::Gpu),
            indirect: gpu.alloc_slice::<IndirectData>(max_batches as u64, Memory::Gpu),
            visible_counts: gpu
                .alloc_slice::<u32>(max_batches as u64 * in_flight as u64, Memory::Default),
            draw_counts: gpu.alloc_slice::<u32>(in_flight as u64, Memory::Default),
            in_flight,
            max_clusters,
            max_batches,
        }
    }

    /// Clears counters, compacts clusters, and builds indirect commands.
    /// `first_instance` always names the batch's compacted-cluster base.
    #[allow(clippy::too_many_arguments)] // Every argument is a pass dependency.
    pub fn record(
        &self,
        gpu: &Gpu,
        cb: CommandBuffer,
        fa: &mut impl FrameAlloc,
        scene: &MeshScene,
        instances: MeshInstances<'_>,
        view: MeshRasterView,
        camera_pos: Vec3,
        counter_slot: usize,
    ) {
        assert!(counter_slot < self.in_flight);
        self.assert_partition(scene, instances);
        assert!(instances.batch_count() <= self.max_batches);
        let capacity = cluster_capacity(instances.batches_cpu);
        assert!(
            capacity < (1 << 25),
            "cluster partition capacity {capacity} exceeds the 25-bit visibility-token range"
        );
        assert!(capacity <= self.max_clusters);

        let count_offset = counter_slot
            .checked_mul(self.max_batches as usize)
            .expect("counter ring offset overflow");
        // SAFETY: the frame ownership gate makes this ring slot GPU-idle;
        // its range is exactly max_batches elements long.
        unsafe {
            for batch in 0..instances.batch_count() as usize {
                *self.visible_counts.cpu.add(count_offset + batch) = 0;
            }
            *self.draw_counts.cpu.add(counter_slot) = instances.batch_count();
        }

        if instances.batch_count() == 0 {
            return;
        }

        let data = fa.frame_alloc(ClusterCullData {
            instances: instances.instances,
            batches: instances.batches,
            transforms: instances.transforms,
            mesh_data: scene.mesh_data_ptr(),
            meshlets: scene.meshlets_ptr(),
            clusters: self.clusters.gpu,
            visible_counts: self.visible_counts.gpu.offset(count_offset as i64),
            output_indirect: self.indirect.gpu,
            instance_count: instances.count(),
            batch_count: instances.batch_count(),
            max_meshlets_per_mesh: scene.max_meshlets_per_mesh(),
            cull_mask: MESH_FLAG_HIDDEN,
            frustum_planes: extract_frustum_planes(&view.world_to_clip),
            camera_pos: camera_pos.to_array(),
            cone_cull_epsilon: CONE_CULL_EPSILON,
        });

        if instances.count() > 0 {
            gpu.cmd_set_compute_shader(cb, self.cull_shader);
            gpu.cmd_dispatch(
                cb,
                data,
                instances.count().div_ceil(64),
                scene.max_meshlets_per_mesh().max(1).div_ceil(4),
                1,
            );
            // Argument generation immediately consumes compute outputs.
            gpu.cmd_barrier(
                cb,
                Stage::Compute,
                Stage::Compute,
                HazardFlags::SHADER_BUFFER,
            );
        }

        gpu.cmd_set_compute_shader(cb, self.args_shader);
        gpu.cmd_dispatch(cb, data, instances.batch_count().div_ceil(64), 1, 1);
        // Graphics consumes indirect commands and compacted clusters.
        gpu.cmd_barrier(
            cb,
            Stage::Compute,
            Stage::All,
            HazardFlags::DRAW_ARGUMENTS | HazardFlags::SHADER_BUFFER,
        );
    }

    /// GPU-built commands for prepass and forward.
    pub fn output(&self) -> gpu::Ptr<IndirectData> {
        self.indirect
    }

    /// GPU compacted `(instance, meshlet)` records for raster shaders.
    pub fn clusters(&self) -> GpuPtr<ClusterInstance> {
        self.clusters.gpu
    }

    /// Allocation handle for test readback.
    pub fn clusters_output(&self) -> gpu::Ptr<ClusterInstance> {
        self.clusters
    }

    /// Pointer to this slot's candidate-batch count.
    pub fn draw_count_ptr(&self, gpu: &Gpu, slot: usize) -> gpu::Ptr<u32> {
        assert!(slot < self.in_flight);
        gpu.mem_suballoc(
            self.draw_counts.cast(),
            (slot * core::mem::size_of::<u32>()) as i64,
            core::mem::size_of::<u32>() as u64,
            1,
        )
        .cast()
    }

    /// Reads one completed frame-slot counter for verification.
    pub fn visible_count(&self, slot: usize, batch_index: u32) -> u32 {
        assert!(slot < self.in_flight);
        assert!(batch_index < self.max_batches);
        let offset = slot * self.max_batches as usize + batch_index as usize;
        // SAFETY: caller observes the frame-idle ownership contract.
        unsafe { *self.visible_counts.cpu.add(offset) }
    }

    /// Verifies that GPU counts stayed within host batch capacities.
    pub fn assert_counts(&self, slot: usize, batches: &[MeshBatch]) {
        assert!(slot < self.in_flight);
        assert!(batches.len() as u32 <= self.max_batches);
        for (batch_index, batch) in batches.iter().enumerate() {
            let count = self.visible_count(slot, batch_index as u32);
            assert!(
                count <= batch.cluster_capacity,
                "cluster cull exceeded batch {batch_index} capacity: {count} > {}; \
                 the host's exclusive cluster partition is invalid",
                batch.cluster_capacity,
            );
        }
    }

    pub fn free(self, gpu: &Gpu) {
        gpu.shader_destroy(self.cull_shader);
        gpu.shader_destroy(self.args_shader);
        gpu.free(self.clusters);
        gpu.free(self.indirect);
        gpu.free(self.visible_counts);
        gpu.free(self.draw_counts);
    }

    fn assert_partition(&self, scene: &MeshScene, instances: MeshInstances<'_>) {
        let mut expected_base = 0u32;
        for (batch_index, batch) in instances.batches_cpu.iter().enumerate() {
            assert!(
                batch.mesh_index < scene.mesh_slot_bound()
                    && scene.mesh_slot_live(batch.mesh_index)
                    && batch.material_index < scene.material_count(),
                "batch {batch_index} references an unregistered mesh or material"
            );
            assert_eq!(
                batch.cluster_base, expected_base,
                "batch {batch_index} does not begin at the prior exclusive range end"
            );
            expected_base = expected_base
                .checked_add(batch.cluster_capacity)
                .expect("cluster partition exceeds u32");
        }
        for (instance_id, instance) in instances.instances_cpu.iter().enumerate() {
            assert!(
                (instance.batch_index as usize) < instances.batches_cpu.len(),
                "instance {instance_id} has no batch"
            );
        }
    }
}

fn cluster_capacity(batches: &[MeshBatch]) -> u32 {
    batches.iter().fold(0u32, |sum, batch| {
        sum.checked_add(batch.cluster_capacity)
            .expect("cluster capacity exceeds u32")
    })
}
