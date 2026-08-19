//! Bridges Bevy extraction data to the Bevy-free mesh renderer.
//!
//! [`MeshBridge`] owns registration and streamed frame data; hosts retain
//! control of pass order and render targets.

use abi_core::GpuPtr;
use abi_mesh::DeformerStack;
use abi_mesh::{DrawTransform, MeshBatch, MeshInstance};
use mesh::{MeshInstances, MeshScene, MeshSceneDesc, ShadowBlasDesc};
use render::FrameCtx;

/// Adapts `FrameCtx` to `mesh::FrameAlloc`.
pub struct Alloc<'a, 'w>(pub &'a mut FrameCtx<'w>);

impl mesh::FrameAlloc for Alloc<'_, '_> {
    fn frame_alloc<T: bytemuck::Pod>(&mut self, value: T) -> GpuPtr<T> {
        self.0.frame_alloc(value)
    }

    fn frame_alloc_slice<T: bytemuck::Pod>(&mut self, values: &[T]) -> GpuPtr<T> {
        <FrameCtx<'_> as mesh::FrameAlloc>::frame_alloc_slice(self.0, values)
    }
}

/// Render-thread bridge owning scene registration and frame streams.
/// Call [`Self::ingest`] once before recording passes.
pub struct MeshBridge {
    scene: MeshScene,
    /// CPU mirror of this frame's streamed instances.
    instances_scratch: Vec<MeshInstance>,
    batches_scratch: Vec<MeshBatch>,
    transforms_scratch: Vec<DrawTransform>,
    instances: GpuPtr<MeshInstance>,
    batches: GpuPtr<MeshBatch>,
    transforms: GpuPtr<DrawTransform>,
    deformers: GpuPtr<DeformerStack>,
}

impl MeshBridge {
    pub fn new(gpu: &gpu::Gpu, desc: &MeshSceneDesc) -> Self {
        Self::from_scene(MeshScene::new(gpu, desc))
    }

    pub fn new_with_shadows(gpu: &gpu::Gpu, desc: &MeshSceneDesc, shadows: ShadowBlasDesc) -> Self {
        Self::from_scene(MeshScene::new_with_shadows(gpu, desc, shadows))
    }

    fn from_scene(scene: MeshScene) -> Self {
        Self {
            scene,
            instances_scratch: Vec::new(),
            batches_scratch: Vec::new(),
            transforms_scratch: Vec::new(),
            // Empty pointers and mirrors represent an empty stream.
            instances: GpuPtr::null(),
            batches: GpuPtr::null(),
            transforms: GpuPtr::null(),
            deformers: GpuPtr::null(),
        }
    }

    /// Applies materials, then mesh removals, updates, and additions.
    /// Removals run first so additions can safely reclaim freed slots.
    /// Call once before recording; extraction must be enabled.
    pub fn ingest(&mut self, ctx: &mut FrameCtx) {
        let gpu = ctx.gpu;
        for upload in ctx.material_uploads() {
            let handle = self.scene.add_material(gpu, upload.entry);
            assert!(
                handle.index() == upload.index,
                "material index authority violated"
            );
        }
        for removal in ctx.mesh_removals() {
            let handle = self.scene.mesh_handle_at(removal.index);
            self.scene.remove_mesh(handle);
        }
        for update in ctx.mesh_updates() {
            let upload = &update.0;
            let handle = self.scene.mesh_handle_at(upload.index);
            self.scene.update_mesh(
                gpu,
                handle,
                mesh::MeshDesc {
                    positions: &upload.positions,
                    normals: &upload.normals,
                    uvs: &upload.uvs,
                    indices: &upload.indices,
                    tangents: upload.tangents.as_deref(),
                    joint_weights: upload.joint_weights.as_deref(),
                    colors: upload.colors.as_deref(),
                },
            );
        }
        for upload in ctx.mesh_uploads() {
            let handle = self.scene.add_mesh(
                gpu,
                mesh::MeshDesc {
                    positions: &upload.positions,
                    normals: &upload.normals,
                    uvs: &upload.uvs,
                    indices: &upload.indices,
                    tangents: upload.tangents.as_deref(),
                    joint_weights: upload.joint_weights.as_deref(),
                    colors: upload.colors.as_deref(),
                },
            );
            assert!(
                handle.index() == upload.index,
                "mesh index authority violated"
            );
        }

        let (instances, count) = ctx.extracted::<MeshInstance>();
        let (batches, batch_count) = ctx.extracted::<MeshBatch>();
        let (transforms, transform_count) = ctx.extracted::<DrawTransform>();
        let (deformers, _) = ctx.extracted::<DeformerStack>();
        assert!(
            count == transform_count,
            "instance/transform streams diverged"
        );
        self.instances = instances;
        self.batches = batches;
        self.transforms = transforms;
        self.deformers = deformers;
        self.instances_scratch.clear();
        self.instances_scratch
            .extend_from_slice(ctx.extracted_host::<MeshInstance>());
        self.batches_scratch.clear();
        self.batches_scratch
            .extend_from_slice(ctx.extracted_host::<MeshBatch>());
        assert!(self.batches_scratch.len() as u32 == batch_count);
        self.transforms_scratch.clear();
        self.transforms_scratch
            .extend_from_slice(ctx.extracted_host::<DrawTransform>());
        assert!(self.transforms_scratch.len() as u32 == transform_count);
        self.rebuild_streamed_batch_ranges();
        ctx.extracted_host_mut::<MeshBatch>()
            .copy_from_slice(&self.batches_scratch);
    }

    /// Returns this frame's stream until the next [`Self::ingest`].
    pub fn instances(&self) -> MeshInstances<'_> {
        MeshInstances {
            instances: self.instances,
            batches: self.batches,
            transforms: self.transforms,
            deformers: self.deformers,
            instances_cpu: &self.instances_scratch,
            batches_cpu: &self.batches_scratch,
            transforms_cpu: &self.transforms_scratch,
        }
    }

    /// Returns the read-only scene used by mesh passes.
    pub fn scene(&self) -> &MeshScene {
        &self.scene
    }

    pub fn free(self, gpu: &gpu::Gpu) {
        self.scene.free(gpu);
    }

    /// Rebuilds streamed ranges after registrations or geometry updates.
    fn rebuild_streamed_batch_ranges(&mut self) {
        let mut cluster_base = 0u32;
        for (batch_index, batch) in self.batches_scratch.iter_mut().enumerate() {
            assert!(
                batch.mesh_index < self.scene.mesh_slot_bound()
                    && self.scene.mesh_slot_live(batch.mesh_index)
                    && batch.material_index < self.scene.material_count(),
                "streamed batch {batch_index} references unregistered mesh or material"
            );
            let meshlet_count = self.scene.mesh_data_cpu()[batch.mesh_index as usize].meshlet_count;
            let member_count = self
                .instances_scratch
                .iter()
                .filter(|instance| instance.batch_index == batch_index as u32)
                .count() as u32;
            batch.cluster_base = cluster_base;
            batch.cluster_capacity = member_count
                .checked_mul(meshlet_count)
                .expect("streamed cluster capacity exceeds u32");
            cluster_base = cluster_base
                .checked_add(batch.cluster_capacity)
                .expect("streamed cluster base exceeds u32");
        }
        for (instance_id, instance) in self.instances_scratch.iter().enumerate() {
            assert!(
                (instance.batch_index as usize) < self.batches_scratch.len(),
                "streamed instance {instance_id} has no batch"
            );
        }
    }
}
