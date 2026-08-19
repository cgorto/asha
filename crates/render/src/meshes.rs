//! Main-thread mesh extraction for the render-thread scene.
//!
//! Mesh assets are registered once and instances stream each frame. The
//! extract drains lifecycle events in this order: removals, updates, then
//! uploads. It mirrors the scene's free-list order, so both sides assign the
//! same reusable slots. Materials and shader groups use the same packet.
//!
//! Conversion accepts only the documented GPU layouts. Weighted meshes also
//! require a matching finite unit dual-quaternion palette; extraction validates
//! every referenced palette lane before publishing device pointers.

use abi_core::GpuPtr;
use abi_mesh::world_transform;
use abi_mesh::{
    DeformerStack, MAX_DEFORMERS, lattice_point_count, max_linear_scale, max_offset_magnitude,
};
use abi_mesh::{
    DrawTransform, DualQuat, JointWeights, MESH_FLAG_DYNAMIC, MESH_FLAG_NO_SHADOW,
    MESH_FLAG_SKINNED, MaterialEntry, MeshBatch, MeshInstance,
};
use bevy::asset::{AssetEvent, AssetId, Assets};
use bevy::ecs::message::{MessageCursor, Messages};
use bevy::mesh::{Indices, Mesh, Mesh3d, VertexAttributeValues};
use bevy::prelude::*;
use mesh::{ShaderCoatSlice, ShaderGroupSlice};

use crate::ExtractFn;

/// One validated Bevy mesh upload with its main-thread slot index.
pub struct MeshUpload {
    pub index: u32,
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub tangents: Option<Vec<[f32; 4]>>,
    /// Optional per-vertex multiplicative tint. `None` is untinted.
    pub colors: Option<Vec<[f32; 4]>>,
    pub joint_weights: Option<Vec<JointWeights>>,
    pub indices: Vec<u32>,
}

/// Replaces the payload in an already-registered mesh slot.
pub struct MeshUpdate(pub MeshUpload);

/// A mesh slot the main thread has released (its `Mesh` asset was removed).
/// The render thread retires the slot and increments its generation, invalidating
/// stale mesh handles.
pub struct MeshRemoval {
    pub index: u32,
}

/// One material registration with its main-thread index.
pub struct MaterialUpload {
    pub index: u32,
    pub entry: MaterialEntry,
}

/// A registered material handle used by extracted mesh instances.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshMaterial(u32);

impl MeshMaterial {
    pub fn index(self) -> u32 {
        self.0
    }
}
/// Authored instance flags; skinning and dynamism are derived.
///
/// Skinning flags are rejected here because extraction owns those bits.
#[repr(transparent)]
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MeshInstanceFlags(pub u32);

/// Retained palette of finite, unit dual-quaternion rigid transforms.
///
/// A palette is valid only for a weighted mesh and must cover every joint
/// index loaded by the shader, including zero-weight lanes.
#[derive(Component, Debug)]
pub struct MeshSkin {
    transforms: Box<[DualQuat]>,
}

impl MeshSkin {
    /// Creates a nonempty palette of rigid dual-quaternion transforms.
    ///
    /// Panics when the palette is empty; weighted meshes require a palette.
    pub fn new(transforms: Box<[DualQuat]>) -> Self {
        assert!(!transforms.is_empty(), "MeshSkin palette must not be empty");
        Self { transforms }
    }

    /// Returns the palette for association and extraction checks.
    pub fn transforms(&self) -> &[DualQuat] {
        &self.transforms
    }

    /// Mutably exposes palette entries for in-place pose updates.
    pub fn transforms_mut(&mut self) -> &mut [DualQuat] {
        &mut self.transforms
    }
}

/// Optional per-entity parameter vector; absence means neutral white.
#[repr(transparent)]
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct MeshInstanceColor(pub [f32; 4]);

impl Default for MeshInstanceColor {
    fn default() -> Self {
        Self([1.0; 4])
    }
}

/// Registered bespoke forward group; absence selects standard shading.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshShader(u32);

impl MeshShader {
    pub fn group(self) -> u32 {
        self.0
    }
}

/// How a shader group participates in the frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ShaderGroupMode {
    /// Draw the forward run with this pair instead of `mesh_vert`/`mesh_frag`.
    #[default]
    ReplaceForward,
    /// Draw after forward with additive color blending.
    Coat,
}

/// Additive coat group, independent of the forward group.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshCoat(u32);

impl MeshCoat {
    pub fn group(self) -> u32 {
        self.0
    }
}

/// One shader group registration and its `.spv` entry names.
///
/// A replacement vertex shader must preserve standard vertex positions so
/// the depth-prepass `Equal` test accepts its fragments.
#[derive(Clone, Debug)]
pub struct ShaderGroupDesc {
    pub vert: Option<String>,
    pub frag: String,
    pub mode: ShaderGroupMode,
}

/// One registered shader group bound for the render thread.
pub struct ShaderGroupUpload {
    pub index: u32,
    pub desc: ShaderGroupDesc,
}

/// Main-thread shader-group registry; registration assigns dense indices.
#[derive(Resource, Default)]
pub struct ShaderGroups {
    count: u32,
    pub(crate) pending: Vec<ShaderGroupUpload>,
}

impl ShaderGroups {
    pub fn register(&mut self, desc: ShaderGroupDesc) -> MeshShader {
        assert!(
            desc.mode == ShaderGroupMode::ReplaceForward,
            "register is the forward path; coat groups use register_coat"
        );
        MeshShader(self.push(desc))
    }

    pub fn register_coat(&mut self, desc: ShaderGroupDesc) -> MeshCoat {
        assert!(
            desc.mode == ShaderGroupMode::Coat,
            "register_coat requires ShaderGroupMode::Coat"
        );
        MeshCoat(self.push(desc))
    }

    fn push(&mut self, desc: ShaderGroupDesc) -> u32 {
        let index = self.count;
        self.count += 1;
        self.pending.push(ShaderGroupUpload { index, desc });
        index
    }

    pub fn count(&self) -> u32 {
        self.count
    }
}

/// Main-thread material registry; `add` assigns the next index and queues it.
#[derive(Resource, Default)]
pub struct MeshMaterials {
    count: u32,
    pending: Vec<MaterialUpload>,
}

impl MeshMaterials {
    pub fn add(&mut self, entry: MaterialEntry) -> MeshMaterial {
        let index = self.count;
        self.count += 1;
        self.pending.push(MaterialUpload { index, entry });
        MeshMaterial(index)
    }

    pub fn count(&self) -> u32 {
        self.count
    }
}

struct JointInfluenceBounds {
    joint_index: u32,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
}

/// Cold culling metadata for positively influenced joints.
struct SkinCullMetadata {
    joints: Box<[JointInfluenceBounds]>,
    /// Deduplicated pivot/co-influence pairs used by blend bounds.
    pivot_pairs: Box<[(u32, u32)]>,
    max_rest_position_magnitude: f64,
}

struct MeshRegistration {
    index: u32,
    drain: u64,
    /// Maximum index across all four shader-loaded weight lanes.
    max_joint_index: Option<u32>,
    skin_cull: Option<SkinCullMetadata>,
}

fn maximum_joint_index(joint_weights: Option<&[JointWeights]>) -> Option<u32> {
    joint_weights.map(|joint_weights| {
        joint_weights
            .iter()
            .flat_map(|vertex| vertex.joint_indices)
            .max()
            .expect("weighted meshes have one JointWeights row per nonempty vertex stream")
    })
}

fn derive_skin_cull_metadata(
    positions: &[[f32; 3]],
    joint_weights: Option<&[JointWeights]>,
) -> Option<SkinCullMetadata> {
    let joint_weights = joint_weights?;
    assert_eq!(
        joint_weights.len(),
        positions.len(),
        "skin-cull weights must be vertex-parallel"
    );

    let mut joint_lookup: bevy::platform::collections::HashMap<u32, usize> = Default::default();
    let mut pair_lookup: bevy::platform::collections::HashSet<(u32, u32)> = Default::default();
    let mut joints = Vec::<JointInfluenceBounds>::new();
    let mut max_rest_position_magnitude = 0.0f64;

    for (vertex, (&position, weights)) in positions.iter().zip(joint_weights).enumerate() {
        assert!(
            position.iter().all(|value| value.is_finite()),
            "skin-cull position[{vertex}] must be finite"
        );
        let px = f64::from(position[0]);
        let py = f64::from(position[1]);
        let pz = f64::from(position[2]);
        max_rest_position_magnitude =
            max_rest_position_magnitude.max((px * px + py * py + pz * pz).sqrt());

        assert!(
            weights.weights[0] > 0.0,
            "skin-cull weights[{vertex}] slot 0 must carry the vertex's leading influence \
             (it is the dual-quaternion hemisphere pivot)"
        );

        let mut accepted_sum = 0.0f32;
        for influence in 0..4 {
            let weight = weights.weights[influence];
            assert!(
                weight.is_finite() && weight >= 0.0,
                "skin-cull weight[{vertex}][{influence}] must be finite and nonnegative"
            );
            accepted_sum += weight;
            if weight == 0.0 {
                continue;
            }

            let joint_index = weights.joint_indices[influence];
            let bounds_index = if let Some(&index) = joint_lookup.get(&joint_index) {
                index
            } else {
                let index = joints.len();
                joint_lookup.insert(joint_index, index);
                joints.push(JointInfluenceBounds {
                    joint_index,
                    aabb_min: position,
                    aabb_max: position,
                });
                index
            };
            let bounds = &mut joints[bounds_index];
            for axis in 0..3 {
                bounds.aabb_min[axis] = bounds.aabb_min[axis].min(position[axis]);
                bounds.aabb_max[axis] = bounds.aabb_max[axis].max(position[axis]);
            }
        }
        assert!(
            (accepted_sum - 1.0).abs() <= 1.0e-4,
            "skin-cull weights[{vertex}] sum must be within 1e-4 of 1 (got {accepted_sum})"
        );
        let pivot = joint_lookup[&weights.joint_indices[0]] as u32;
        for influence in 1..4 {
            if weights.weights[influence] == 0.0 {
                continue;
            }
            let other = joint_lookup[&weights.joint_indices[influence]] as u32;
            if other != pivot {
                pair_lookup.insert((pivot.min(other), pivot.max(other)));
            }
        }
    }
    assert!(
        !joints.is_empty(),
        "an accepted weighted mesh must have a positive influence"
    );

    let mut pivot_pairs: Vec<(u32, u32)> = pair_lookup.into_iter().collect();
    pivot_pairs.sort_unstable();
    Some(SkinCullMetadata {
        joints: joints.into_boxed_slice(),
        pivot_pairs: pivot_pairs.into_boxed_slice(),
        max_rest_position_magnitude,
    })
}

fn assert_mesh_skin_association(registration: &MeshRegistration, skin: Option<&MeshSkin>) {
    match (registration.max_joint_index, skin) {
        (None, None) => {
            assert!(registration.skin_cull.is_none());
        }
        (None, Some(_)) => panic!(
            "static mesh {} rejects MeshSkin; only meshes with joint weights may carry a palette",
            registration.index
        ),
        (Some(_), None) => panic!("weighted mesh {} requires MeshSkin", registration.index),
        (Some(max_joint_index), Some(skin)) => {
            assert!(registration.skin_cull.is_some());
            assert!(
                (max_joint_index as usize) < skin.transforms().len(),
                "weighted mesh {} references joint {max_joint_index}, but MeshSkin palette has {} transforms",
                registration.index,
                skin.transforms().len()
            );
            for (joint, transform) in skin.transforms().iter().enumerate() {
                assert!(
                    transform
                        .real
                        .iter()
                        .chain(&transform.dual)
                        .all(|value| value.is_finite()),
                    "weighted mesh {} MeshSkin transform {joint} must be finite",
                    registration.index
                );
                let real_norm_squared: f32 = transform.real.iter().map(|value| value * value).sum();
                assert!(
                    (real_norm_squared - 1.0).abs() <= 1.0e-3,
                    "weighted mesh {} MeshSkin transform {joint} must have a unit rotation \
                     (|real|^2 = {real_norm_squared})",
                    registration.index
                );
            }
        }
    }
}

/// A palette entry lifted to f64 with shader-equivalent normalization.
struct JointRigidF64 {
    real: [f64; 4],
    dual: [f64; 4],
}

impl JointRigidF64 {
    fn from_entry(entry: &DualQuat) -> Self {
        let real = entry.real.map(f64::from);
        let dual = entry.dual.map(f64::from);
        let norm = real.iter().map(|value| value * value).sum::<f64>().sqrt();
        assert!(
            norm.is_finite() && norm > 0.5,
            "skin-cull palette rotation must be near-unit (|real| = {norm})"
        );
        Self {
            real: real.map(|value| value / norm),
            dual: dual.map(|value| value / norm),
        }
    }

    fn translation(&self) -> [f64; 3] {
        let [rx, ry, rz, rw] = self.real;
        let [dx, dy, dz, dw] = self.dual;
        [
            2.0 * (rw * dx - dw * rx + (ry * dz - rz * dy)),
            2.0 * (rw * dy - dw * ry + (rz * dx - rx * dz)),
            2.0 * (rw * dz - dw * rz + (rx * dy - ry * dx)),
        ]
    }

    fn transform_point(&self, p: [f64; 3]) -> [f64; 3] {
        let [rx, ry, rz, rw] = self.real;
        let axis = [rx, ry, rz];
        let inner = [
            axis[1] * p[2] - axis[2] * p[1] + rw * p[0],
            axis[2] * p[0] - axis[0] * p[2] + rw * p[1],
            axis[0] * p[1] - axis[1] * p[0] + rw * p[2],
        ];
        let rotated = [
            p[0] + 2.0 * (axis[1] * inner[2] - axis[2] * inner[1]),
            p[1] + 2.0 * (axis[2] * inner[0] - axis[0] * inner[2]),
            p[2] + 2.0 * (axis[0] * inner[1] - axis[1] * inner[0]),
        ];
        let t = self.translation();
        [rotated[0] + t[0], rotated[1] + t[1], rotated[2] + t[2]]
    }
}

fn norm3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Conservative displacement bound for dual-quaternion skinning.
///
/// Rigid joint corner bounds and normalized-blend slack cover every weighted
/// vertex. Near-antipodal pivot pairs panic because the blend is singular.
fn skin_bounds_dilation(
    metadata: &SkinCullMetadata,
    skin: &MeshSkin,
    model_to_world: &abi_core::glam::Mat4,
) -> f32 {
    let mut max_joint_displacement = 0.0f64;
    let mut max_dual_magnitude = 0.0f64;
    let lifted: Vec<JointRigidF64> = metadata
        .joints
        .iter()
        .map(|bounds| {
            let entry = skin
                .transforms()
                .get(bounds.joint_index as usize)
                .unwrap_or_else(|| {
                    panic!(
                        "positive skin-cull joint {} is outside the MeshSkin palette",
                        bounds.joint_index
                    )
                });
            JointRigidF64::from_entry(entry)
        })
        .collect();
    for (bounds, joint) in metadata.joints.iter().zip(&lifted) {
        max_dual_magnitude = max_dual_magnitude.max(
            joint
                .dual
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt(),
        );
        for corner in 0..8 {
            let p = [
                f64::from(if corner & 1 == 0 {
                    bounds.aabb_min[0]
                } else {
                    bounds.aabb_max[0]
                }),
                f64::from(if corner & 2 == 0 {
                    bounds.aabb_min[1]
                } else {
                    bounds.aabb_max[1]
                }),
                f64::from(if corner & 4 == 0 {
                    bounds.aabb_min[2]
                } else {
                    bounds.aabb_max[2]
                }),
            ];
            let moved = joint.transform_point(p);
            let displacement = norm3([moved[0] - p[0], moved[1] - p[1], moved[2] - p[2]]);
            assert!(
                displacement.is_finite(),
                "skin-cull corner displacement must be finite"
            );
            max_joint_displacement = max_joint_displacement.max(displacement);
        }
    }

    let mut min_pair_dot = 1.0f64;
    let mut max_real_chord = 0.0f64;
    let mut max_dual_chord = 0.0f64;
    for &(a, b) in &metadata.pivot_pairs {
        let a = &lifted[a as usize];
        let b = &lifted[b as usize];
        let dot: f64 = a.real.iter().zip(&b.real).map(|(a, b)| a * b).sum();
        let sign = if dot < 0.0 { -1.0 } else { 1.0 };
        min_pair_dot = min_pair_dot.min(dot.abs());
        let mut real_chord_squared = 0.0f64;
        let mut dual_chord_squared = 0.0f64;
        for axis in 0..4 {
            let real_delta = sign * a.real[axis] - b.real[axis];
            let dual_delta = sign * a.dual[axis] - b.dual[axis];
            real_chord_squared += real_delta * real_delta;
            dual_chord_squared += dual_delta * dual_delta;
        }
        max_real_chord = max_real_chord.max(real_chord_squared.sqrt());
        max_dual_chord = max_dual_chord.max(dual_chord_squared.sqrt());
    }
    assert!(
        min_pair_dot > 0.01,
        "co-influencing joints are near-antipodal (pivot-pair dot {min_pair_dot}); \
         the dual-quaternion blend is singular for this palette"
    );

    let norm_slack = 1.0 / min_pair_dot - 1.0;
    let blend_real_error = max_real_chord / min_pair_dot + norm_slack;
    let blend_dual_error = max_dual_chord / min_pair_dot + norm_slack * max_dual_magnitude;
    let blend_slack = 2.0 * blend_real_error * metadata.max_rest_position_magnitude
        + 2.0 * (blend_dual_error + max_dual_magnitude * blend_real_error);

    let model_dilation = max_joint_displacement + blend_slack;
    let world_scale = max_linear_scale(model_to_world);
    assert!(
        world_scale.is_finite() && world_scale >= 0.0,
        "skin-cull world scale must be finite and nonnegative"
    );
    finite_f32_upper_bound(
        model_dilation * f64::from(world_scale),
        "skin bounds dilation",
    )
}

fn finite_f32_upper_bound(value: f64, name: &str) -> f32 {
    assert!(
        value.is_finite() && value >= 0.0 && value <= f64::from(f32::MAX),
        "{name} overflowed or became non-finite ({value})"
    );
    if value == 0.0 {
        return 0.0;
    }
    let rounded = value as f32;
    let upper = if f64::from(rounded) < value {
        f32::from_bits(rounded.to_bits() + 1)
    } else {
        rounded
    };
    assert!(upper.is_finite(), "{name} overflowed f32");
    upper
}

fn compose_bounds_dilation(skin: f32, deformer: f32) -> f32 {
    assert!(
        skin.is_finite() && skin >= 0.0 && deformer.is_finite() && deformer >= 0.0,
        "bounds dilation inputs must be finite and nonnegative"
    );
    finite_f32_upper_bound(f64::from(skin) + f64::from(deformer), "bounds dilation")
}

fn derive_mesh_instance_flags(authored: u32, dynamic: bool, skinned: bool) -> u32 {
    assert!(
        authored & (MESH_FLAG_SKINNED | MESH_FLAG_DYNAMIC) == 0,
        "MeshInstanceFlags may not author derived MESH_FLAG_SKINNED or MESH_FLAG_DYNAMIC"
    );
    authored
        | if dynamic || skinned {
            MESH_FLAG_DYNAMIC
        } else {
            0
        }
        | if skinned {
            MESH_FLAG_SKINNED | MESH_FLAG_NO_SHADOW
        } else {
            0
        }
}

/// Copies one palette into a disjoint aligned frame-arena subrun.
///
/// # Safety
/// The caller must allocate the exact total palette length and pass the next
/// unwritten destination; each subrun is copied once and remains disjoint.
unsafe fn write_mesh_skin(
    palette_base: GpuPtr<DualQuat>,
    palette_dst: &mut *mut DualQuat,
    palette_written: &mut usize,
    skin: Option<&MeshSkin>,
) -> GpuPtr<DualQuat> {
    let Some(skin) = skin else {
        return GpuPtr::null();
    };
    let entry_offset = i64::try_from(*palette_written).expect("frame palette offset exceeds i64");
    let joint_transforms = palette_base.offset(entry_offset);
    assert!(
        joint_transforms.addr() % 16 == 0 && (*palette_dst as usize) % 16 == 0,
        "every MeshSkin palette subrun must start at an actual 16-byte-aligned address"
    );
    // SAFETY: caller allocated the frame-wide run for the exact palette sum;
    // this destination is its next disjoint subrun.
    unsafe {
        std::ptr::copy_nonoverlapping(
            skin.transforms().as_ptr(),
            *palette_dst,
            skin.transforms().len(),
        );
        *palette_dst = (*palette_dst).add(skin.transforms().len());
    }
    *palette_written = palette_written
        .checked_add(skin.transforms().len())
        .expect("frame palette count exceeds usize");
    joint_transforms
}

/// Builds the mesh extract and its retained lifecycle registry.
///
/// Each frame drains removals, then updates, then new uploads. It then validates
/// mesh/skin associations, streams instances and palettes, and sorts batches by
/// group before publishing host and arena lanes.
pub(crate) fn make_mesh_extract() -> ExtractFn {
    let mut registry: bevy::platform::collections::HashMap<AssetId<Mesh>, MeshRegistration> =
        Default::default();
    let mut free_slots: Vec<u32> = Vec::new();
    let mut slot_bound: u32 = 0;
    let mut query: Option<
        QueryState<(
            bevy::prelude::Ref<'static, Mesh3d>,
            bevy::prelude::Ref<'static, GlobalTransform>,
            &'static MeshMaterial,
            Option<&'static MeshSkin>,
            Option<&'static DeformerStack>,
            Option<&'static MeshInstanceFlags>,
            Option<&'static MeshShader>,
            Option<&'static MeshCoat>,
            Option<&'static MeshInstanceColor>,
        )>,
    > = None;
    let mut asset_cursor = MessageCursor::<AssetEvent<Mesh>>::default();
    let mut batch_lookup: bevy::platform::collections::HashMap<(u32, u32, u64), u32> =
        Default::default();
    let mut batch_keys: Vec<u64> = Vec::new();
    let mut batch_order: Vec<u32> = Vec::new();
    let mut batch_remap: Vec<u32> = Vec::new();
    let mut batches_sorted: Vec<MeshBatch> = Vec::new();
    let mut drain: u64 = 0;

    Box::new(move |world, frame| {
        drain += 1;

        let messages = world.resource::<Messages<AssetEvent<Mesh>>>();
        let mut removed: Vec<AssetId<Mesh>> = Vec::new();
        let mut modified: Vec<AssetId<Mesh>> = Vec::new();
        for event in asset_cursor.read(messages) {
            match event {
                AssetEvent::Removed { id } => removed.push(*id),
                AssetEvent::Modified { id } => modified.push(*id),
                _ => {}
            }
        }
        for id in removed {
            if let Some(registration) = registry.remove(&id) {
                frame.mesh_removals.push(MeshRemoval {
                    index: registration.index,
                });
                free_slots.push(registration.index);
            }
        }
        let assets = world.resource::<Assets<Mesh>>();
        for id in modified {
            let Some(registration) = registry.get_mut(&id) else {
                continue;
            };
            if drain <= registration.drain + 1 {
                continue;
            }
            let mesh = assets
                .get(id)
                .expect("AssetEvent::Modified names an asset Assets<Mesh> still holds");
            let upload = convert_mesh(id, mesh, registration.index);
            registration.max_joint_index = maximum_joint_index(upload.joint_weights.as_deref());
            registration.skin_cull =
                derive_skin_cull_metadata(&upload.positions, upload.joint_weights.as_deref());
            registration.drain = drain;
            frame.mesh_updates.push(MeshUpdate(upload));
        }

        for (id, mesh) in assets.iter() {
            if registry.contains_key(&id) {
                continue;
            }
            let index = free_slots.pop().unwrap_or_else(|| {
                let index = slot_bound;
                slot_bound += 1;
                index
            });
            let upload = convert_mesh(id, mesh, index);
            let max_joint_index = maximum_joint_index(upload.joint_weights.as_deref());
            let skin_cull =
                derive_skin_cull_metadata(&upload.positions, upload.joint_weights.as_deref());
            frame.mesh_uploads.push(upload);
            registry.insert(
                id,
                MeshRegistration {
                    index,
                    drain,
                    max_joint_index,
                    skin_cull,
                },
            );
        }

        let mut materials = world.resource_mut::<MeshMaterials>();
        frame.material_uploads.append(&mut materials.pending);
        let mut groups = world.resource_mut::<ShaderGroups>();
        frame.shader_group_uploads.append(&mut groups.pending);

        let qs = query.get_or_insert_with(|| {
            world.query::<(
                bevy::prelude::Ref<Mesh3d>,
                bevy::prelude::Ref<GlobalTransform>,
                &MeshMaterial,
                Option<&MeshSkin>,
                Option<&DeformerStack>,
                Option<&MeshInstanceFlags>,
                Option<&MeshShader>,
                Option<&MeshCoat>,
                Option<&MeshInstanceColor>,
            )>()
        });
        let count = qs.query(&*world).iter().len();
        let deformer_count = qs
            .query(&*world)
            .iter()
            .filter(|(_, _, _, _, deformer, ..)| deformer.is_some_and(|stack| stack.count > 0))
            .count();
        let mut palette_entry_count = 0usize;
        for (mesh3d, _, _, skin, _, flags, ..) in qs.query(&*world).iter() {
            let registration = registry.get(&mesh3d.id()).unwrap_or_else(|| {
                panic!(
                    "Mesh3d references mesh asset {} that Assets<Mesh> has never contained",
                    mesh3d.id()
                )
            });
            assert_mesh_skin_association(registration, skin);
            let _ =
                derive_mesh_instance_flags(flags.map_or(0, |flags| flags.0), false, skin.is_some());
            if let Some(skin) = skin {
                palette_entry_count = palette_entry_count
                    .checked_add(skin.transforms().len())
                    .expect("frame palette entry count exceeds usize");
            }
        }
        let mut host_instances: Vec<MeshInstance> = take_host(&mut frame.extracted);
        let mut host_batches: Vec<MeshBatch> = take_host(&mut frame.extracted);
        let mut host_transforms: Vec<DrawTransform> = take_host(&mut frame.extracted);
        host_instances.clear();
        host_batches.clear();
        host_transforms.clear();
        let (instances_base, instances_dst) = frame.arena.alloc_bytes(
            (count * size_of::<MeshInstance>()) as u64,
            align_of::<MeshInstance>().max(4) as u64,
        );
        let (batches_base, batches_dst) = frame.arena.alloc_bytes(
            (count * size_of::<MeshBatch>()) as u64,
            align_of::<MeshBatch>().max(4) as u64,
        );
        let (transforms_base, mut transforms_dst) = frame.arena.alloc_bytes(
            (count * size_of::<DrawTransform>()) as u64,
            align_of::<DrawTransform>().max(4) as u64,
        );
        let (deformers_base, mut deformers_dst) = if deformer_count > 0 {
            frame.arena.alloc_bytes(
                (deformer_count * size_of::<DeformerStack>()) as u64,
                align_of::<DeformerStack>().max(4) as u64,
            )
        } else {
            (GpuPtr::<u8>::null(), std::ptr::null_mut())
        };
        let (palette_base, palette_cpu) = if palette_entry_count > 0 {
            let palette_bytes = u64::try_from(palette_entry_count)
                .expect("frame palette entry count exceeds u64")
                .checked_mul(size_of::<DualQuat>() as u64)
                .expect("frame palette byte count exceeds u64");
            let (base, cpu) = frame.arena.alloc_bytes(palette_bytes, 16);
            assert!(
                base.addr() % 16 == 0 && (cpu as usize) % 16 == 0,
                "frame palette run must have actual 16-byte-aligned GPU and CPU addresses"
            );
            (base.cast::<DualQuat>(), cpu.cast::<DualQuat>())
        } else {
            (GpuPtr::null(), std::ptr::null_mut())
        };
        let mut palette_dst = palette_cpu;
        let mut palette_written = 0usize;
        let mut i = 0u32;
        let mut deformer_written = 0u32;
        let mut batch_count = 0u32;
        batch_lookup.clear();
        batch_keys.clear();
        for (mesh3d, global, material, skin, deformer, flags, shader, coat, color) in
            qs.query(&mut *world).iter()
        {
            let registration = registry
                .get(&mesh3d.id())
                .expect("mesh/skin association pass resolved every Mesh3d asset");
            let mesh_index = registration.index;
            let transform = world_transform(global.to_matrix());
            host_transforms.push(transform);
            let (deformer_slot, deformer_dilation) = if let Some(stack) = deformer
                && stack.count > 0
            {
                let slot = deformer_written + 1;
                let dilation = deformer_bounds_dilation(stack, &transform.model_to_world);
                // SAFETY: `deformers_dst` walks a run sized for exactly
                // `deformer_count` stacks; this branch runs once per counted stack.
                unsafe {
                    deformers_dst
                        .cast::<DeformerStack>()
                        .write_unaligned(*stack);
                    deformers_dst = deformers_dst.add(size_of::<DeformerStack>());
                }
                deformer_written += 1;
                (slot, dilation)
            } else {
                (0, 0.0)
            };
            let skin_dilation = skin.map_or(0.0, |skin| {
                skin_bounds_dilation(
                    registration
                        .skin_cull
                        .as_ref()
                        .expect("associated weighted mesh must retain skin-cull metadata"),
                    skin,
                    &transform.model_to_world,
                )
            });
            let bounds_dilation = compose_bounds_dilation(skin_dilation, deformer_dilation);
            let group_key = (u64::from(shader.map_or(0, |shader| shader.0 + 1)) << 32)
                | u64::from(coat.map_or(0, |coat| coat.0 + 1));
            let batch_index = if let Some(&batch_index) =
                batch_lookup.get(&(mesh_index, material.0, group_key))
            {
                batch_index
            } else {
                let batch_index = batch_count;
                batch_lookup.insert((mesh_index, material.0, group_key), batch_index);
                let batch = MeshBatch {
                    mesh_index,
                    material_index: material.0,
                    cluster_base: 0,
                    cluster_capacity: 0,
                };
                host_batches.push(batch);
                batch_keys.push(group_key);
                batch_count += 1;
                batch_index
            };
            let dynamic = global.is_changed()
                || mesh3d.is_changed()
                || deformer.is_some_and(|stack| stack.count > 0);
            let instance_flags = derive_mesh_instance_flags(
                flags.map_or(0, |flags| flags.0),
                dynamic,
                skin.is_some(),
            );
            // SAFETY: the palette run was sized from this exact query after
            // every mesh/skin association passed; each call advances through
            // one disjoint, 16-byte-aligned (32-byte stride) subrun.
            let joint_transforms = unsafe {
                write_mesh_skin(palette_base, &mut palette_dst, &mut palette_written, skin)
            };
            let instance = MeshInstance {
                batch_index,
                transform_index: i,
                flags: instance_flags,
                outline_group: 0,
                instance_color: color.map_or([1.0; 4], |color| color.0),
                joint_transforms,
                deformer_slot,
                bounds_dilation,
            };
            host_instances.push(instance);
            // SAFETY: the run was sized for `count` elements above and the
            // query is iterated once; dst walks element-by-element.
            unsafe {
                transforms_dst
                    .cast::<DrawTransform>()
                    .write_unaligned(transform);
                transforms_dst = transforms_dst.add(size_of::<DrawTransform>());
            }
            i += 1;
        }
        assert!(
            i as usize == count,
            "instance query length changed mid-extract"
        );
        assert!(
            deformer_written as usize == deformer_count,
            "deformer query length changed mid-extract"
        );
        assert!(
            palette_written == palette_entry_count,
            "skin palette query length changed mid-extract"
        );
        let mut host_slices: Vec<ShaderGroupSlice> = take_host(&mut frame.extracted);
        let mut host_coats: Vec<ShaderCoatSlice> = take_host(&mut frame.extracted);
        host_slices.clear();
        host_coats.clear();
        batch_order.clear();
        batch_order.extend(0..batch_count);
        batch_order.sort_by_key(|&index| batch_keys[index as usize]);
        batch_remap.clear();
        batch_remap.resize(batch_count as usize, 0);
        batches_sorted.clear();
        for (new_index, &old_index) in batch_order.iter().enumerate() {
            batch_remap[old_index as usize] = new_index as u32;
            batches_sorted.push(host_batches[old_index as usize]);
        }
        std::mem::swap(&mut host_batches, &mut batches_sorted);
        for instance in &mut host_instances {
            instance.batch_index = batch_remap[instance.batch_index as usize];
        }
        let key_at = |i: u32| batch_keys[batch_order[i as usize] as usize];
        let mut run = 0u32;
        while run < batch_count {
            let group = key_at(run) >> 32;
            let base = run;
            while run < batch_count && key_at(run) >> 32 == group {
                run += 1;
            }
            if group != 0 {
                host_slices.push(ShaderGroupSlice {
                    group: group as u32 - 1,
                    batch_base: base,
                    batch_count: run - base,
                });
            }
        }
        run = 0;
        while run < batch_count {
            let key = key_at(run);
            let base = run;
            while run < batch_count && key_at(run) == key {
                run += 1;
            }
            let coat = key & 0xffff_ffff;
            if coat != 0 {
                host_coats.push(ShaderCoatSlice {
                    group: coat as u32 - 1,
                    batch_base: base,
                    batch_count: run - base,
                });
            }
        }
        // SAFETY: both runs were sized for `count`/`count` elements above;
        // the Vecs hold `count` and `batch_count` (asserted <= count).
        unsafe {
            std::ptr::copy_nonoverlapping(
                host_instances.as_ptr().cast::<u8>(),
                instances_dst,
                host_instances.len() * size_of::<MeshInstance>(),
            );
            std::ptr::copy_nonoverlapping(
                host_batches.as_ptr().cast::<u8>(),
                batches_dst,
                host_batches.len() * size_of::<MeshBatch>(),
            );
        }
        frame.extracted.host.insert(
            std::any::TypeId::of::<MeshInstance>(),
            Box::new(host_instances),
        );
        frame
            .extracted
            .host
            .insert(std::any::TypeId::of::<MeshBatch>(), Box::new(host_batches));
        frame.extracted.host.insert(
            std::any::TypeId::of::<DrawTransform>(),
            Box::new(host_transforms),
        );
        frame.extracted.host.insert(
            std::any::TypeId::of::<ShaderGroupSlice>(),
            Box::new(host_slices),
        );
        frame.extracted.host.insert(
            std::any::TypeId::of::<ShaderCoatSlice>(),
            Box::new(host_coats),
        );
        frame.extracted.map.insert(
            std::any::TypeId::of::<MeshInstance>(),
            (instances_base.addr(), count as u32),
        );
        frame.extracted.map.insert(
            std::any::TypeId::of::<MeshBatch>(),
            (batches_base.addr(), batch_count),
        );
        frame.extracted.map.insert(
            std::any::TypeId::of::<DrawTransform>(),
            (transforms_base.addr(), count as u32),
        );
        frame.extracted.map.insert(
            std::any::TypeId::of::<DeformerStack>(),
            (
                if deformer_count > 0 {
                    deformers_base.addr()
                } else {
                    0
                },
                deformer_count as u32,
            ),
        );
    })
}

fn deformer_bounds_dilation(stack: &DeformerStack, model_to_world: &abi_core::glam::Mat4) -> f32 {
    let count = (stack.count as usize).min(MAX_DEFORMERS);
    let world_scale = max_linear_scale(model_to_world);
    let mut dilation = 0.0;
    for lat in &stack.lattices[..count] {
        let point_count = lattice_point_count(lat.resolution);
        let offset = max_offset_magnitude(&lat.offsets, point_count);
        dilation += offset * max_linear_scale(&lat.lattice_to_model) * world_scale;
    }
    dilation
}

/// Converts only the accepted Bevy mesh layouts into [`MeshUpload`].
///
/// Unsupported topology, attributes, or mismatched skin streams panic with
/// the asset identifier; deeper finite and index checks run in the scene.
pub(crate) fn convert_mesh(id: AssetId<Mesh>, mesh: &Mesh, index: u32) -> MeshUpload {
    use bevy::mesh::PrimitiveTopology;
    assert!(
        mesh.primitive_topology() == PrimitiveTopology::TriangleList,
        "mesh asset {id}: only TriangleList topology is supported (got {:?})",
        mesh.primitive_topology(),
    );

    let positions = match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
        Some(VertexAttributeValues::Float32x3(v)) => v.clone(),
        Some(_) => panic!("mesh asset {id}: positions must be Float32x3"),
        None => panic!("mesh asset {id}: missing positions"),
    };
    let normals = match mesh.attribute(Mesh::ATTRIBUTE_NORMAL) {
        Some(VertexAttributeValues::Float32x3(v)) => v.clone(),
        Some(_) => panic!("mesh asset {id}: normals must be Float32x3"),
        None => panic!("mesh asset {id}: missing normals"),
    };
    let uvs = match mesh.attribute(Mesh::ATTRIBUTE_UV_0) {
        Some(VertexAttributeValues::Float32x2(v)) => v.clone(),
        Some(_) => panic!("mesh asset {id}: UV0 must be Float32x2"),
        None => panic!("mesh asset {id}: missing UV0"),
    };
    let tangents = match mesh.attribute(Mesh::ATTRIBUTE_TANGENT) {
        Some(VertexAttributeValues::Float32x4(v)) => Some(v.clone()),
        Some(_) => panic!("mesh asset {id}: tangents must be Float32x4"),
        None => None,
    };
    let colors = match mesh.attribute(Mesh::ATTRIBUTE_COLOR) {
        Some(VertexAttributeValues::Float32x4(v)) => Some(v.clone()),
        Some(VertexAttributeValues::Float32x3(v)) => {
            Some(v.iter().map(|c| [c[0], c[1], c[2], 1.0]).collect())
        }
        Some(_) => panic!("mesh asset {id}: colors must be Float32x4 or Float32x3"),
        None => None,
    };
    if let Some(colors) = colors.as_ref() {
        assert_eq!(
            colors.len(),
            positions.len(),
            "mesh asset {id}: color count must match positions"
        );
    }
    let joint_indices = match mesh.attribute(Mesh::ATTRIBUTE_JOINT_INDEX) {
        Some(VertexAttributeValues::Uint16x4(v)) => Some(v),
        Some(_) => panic!("mesh asset {id}: joint indices must be Uint16x4"),
        None => None,
    };
    let weights = match mesh.attribute(Mesh::ATTRIBUTE_JOINT_WEIGHT) {
        Some(VertexAttributeValues::Float32x4(v)) => Some(v),
        Some(_) => panic!("mesh asset {id}: joint weights must be Float32x4"),
        None => None,
    };
    assert!(
        joint_indices.is_some() == weights.is_some(),
        "mesh asset {id}: joint indices and joint weights must be present together"
    );
    let joint_weights = joint_indices.zip(weights).map(|(indices, weights)| {
        assert_eq!(
            indices.len(),
            positions.len(),
            "mesh asset {id}: joint index count must match positions"
        );
        assert_eq!(
            weights.len(),
            positions.len(),
            "mesh asset {id}: joint weight count must match positions"
        );
        indices
            .iter()
            .zip(weights)
            .enumerate()
            .map(|(vertex, (joint_indices, weights))| {
                let mut sum = 0.0f32;
                for (influence, &weight) in weights.iter().enumerate() {
                    assert!(
                        weight.is_finite(),
                        "mesh asset {id}: joint weight[{vertex}][{influence}] must be finite"
                    );
                    assert!(
                        (0.0..=1.0).contains(&weight),
                        "mesh asset {id}: joint weight[{vertex}][{influence}] must be in [0, 1]"
                    );
                    sum += weight;
                }
                assert!(
                    (sum - 1.0).abs() <= 1.0e-4,
                    "mesh asset {id}: joint weights[{vertex}] sum must be within 1e-4 of 1 (got {sum})"
                );
                JointWeights::canonical(joint_indices.map(u32::from), *weights)
            })
            .collect()
    });
    let indices = match mesh.indices() {
        Some(Indices::U32(v)) => v.clone(),
        Some(Indices::U16(v)) => v.iter().map(|&i| u32::from(i)).collect(),
        None => panic!("mesh asset {id}: non-indexed meshes are not supported"),
    };

    MeshUpload {
        index,
        positions,
        normals,
        uvs,
        tangents,
        colors,
        joint_weights,
        indices,
    }
}

/// Removes a host-lane `Vec` for refilling while retaining capacity.
fn take_host<T: 'static>(extracted: &mut crate::Extracted) -> Vec<T> {
    extracted
        .host
        .remove(&std::any::TypeId::of::<T>())
        .and_then(|any| any.downcast::<Vec<T>>().ok())
        .map(|boxed| *boxed)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;
    use std::sync::Mutex;

    use abi_core::GpuPtr;
    use abi_core::glam::{Mat4, Quat, Vec3};
    use abi_mesh::DeformerStack;
    use abi_mesh::{
        DualQuat, JointWeights, MESH_FLAG_DYNAMIC, MESH_FLAG_NO_LINEWORK, MESH_FLAG_NO_SHADOW,
        MESH_FLAG_SKINNED, MaterialEntry, MeshInstance,
    };
    use bevy::asset::{AssetEvent, AssetId, Assets, RenderAssetUsages};
    use bevy::ecs::message::Messages;
    use bevy::mesh::{Indices, Mesh, Mesh3d, PrimitiveTopology, VertexAttributeValues};
    use bevy::prelude::{GlobalTransform, World};
    use gpu::{Gpu, Memory};

    use super::{
        MeshMaterials, MeshRegistration, MeshSkin, ShaderGroups, assert_mesh_skin_association,
        convert_mesh, derive_mesh_instance_flags, derive_skin_cull_metadata, make_mesh_extract,
        maximum_joint_index, skin_bounds_dilation, write_mesh_skin,
    };
    use crate::{Arena, Extracted, Frame};

    static GPU_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn triangle() -> Mesh {
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_POSITION,
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 0.0, 1.0]; 3]);
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0, 0.0]; 3]);
        mesh.insert_indices(Indices::U32(vec![0, 1, 2]));
        mesh
    }

    #[test]
    fn mesh_skin_is_fixed_length_and_mutable_in_place() {
        let transforms = vec![DualQuat::IDENTITY, DualQuat::IDENTITY].into_boxed_slice();
        let mut skin = MeshSkin::new(transforms);
        assert_eq!(skin.transforms().len(), 2);
        let allocation = skin.transforms().as_ptr();
        skin.transforms_mut()[1].dual[2] = 17.0;
        assert_eq!(skin.transforms()[1].dual[2], 17.0);
        assert_eq!(skin.transforms().as_ptr(), allocation);

        let panic = std::panic::catch_unwind(|| MeshSkin::new(Box::new([])))
            .expect_err("an empty palette must panic");
        assert!(panic_message(panic).contains("must not be empty"));
    }

    #[test]
    fn mesh_skin_association_uses_all_four_slots() {
        let static_mesh = MeshRegistration {
            index: 2,
            drain: 1,
            max_joint_index: None,
            skin_cull: None,
        };
        let weighted_rows = [JointWeights {
            joint_indices: [0, 9, 1, 2],
            weights: [1.0, 0.0, 0.0, 0.0],
        }];
        let weighted_mesh = MeshRegistration {
            index: 3,
            drain: 1,
            max_joint_index: maximum_joint_index(Some(&weighted_rows)),
            skin_cull: derive_skin_cull_metadata(&[[0.0; 3]], Some(&weighted_rows)),
        };
        assert_eq!(weighted_mesh.max_joint_index, Some(9));
        assert_mesh_skin_association(&static_mesh, None);
        let valid_skin = MeshSkin::new(vec![DualQuat::IDENTITY; 10].into_boxed_slice());
        assert_mesh_skin_association(&weighted_mesh, Some(&valid_skin));

        let panic = std::panic::catch_unwind(|| {
            assert_mesh_skin_association(&weighted_mesh, None);
        })
        .expect_err("weighted mesh without MeshSkin must panic");
        assert!(panic_message(panic).contains("requires MeshSkin"));

        let panic = std::panic::catch_unwind(|| {
            assert_mesh_skin_association(&static_mesh, Some(&valid_skin));
        })
        .expect_err("static mesh with MeshSkin must panic");
        assert!(panic_message(panic).contains("static mesh 2 rejects MeshSkin"));

        let short_skin = MeshSkin::new(vec![DualQuat::IDENTITY; 9].into_boxed_slice());
        let panic = std::panic::catch_unwind(|| {
            assert_mesh_skin_association(&weighted_mesh, Some(&short_skin));
        })
        .expect_err("zero-weight slot 9 must still require palette row 9");
        assert!(panic_message(panic).contains("references joint 9"));

        let mut non_finite = MeshSkin::new(vec![DualQuat::IDENTITY; 10].into_boxed_slice());
        non_finite.transforms_mut()[4].dual[2] = f32::NAN;
        let panic = std::panic::catch_unwind(|| {
            assert_mesh_skin_association(&weighted_mesh, Some(&non_finite));
        })
        .expect_err("a non-finite palette entry must panic before extraction");
        assert!(panic_message(panic).contains("transform 4 must be finite"));

        let zeroed = MeshSkin::new(vec![DualQuat::default(); 10].into_boxed_slice());
        let panic = std::panic::catch_unwind(|| {
            assert_mesh_skin_association(&weighted_mesh, Some(&zeroed));
        })
        .expect_err("an all-zero palette must panic before extraction");
        assert!(panic_message(panic).contains("must have a unit rotation"));
    }

    #[test]
    fn skin_cull_requires_slot_zero_to_be_the_hemisphere_pivot() {
        let weights = [JointWeights {
            joint_indices: [4, 1, 0, 0],
            weights: [0.0, 1.0, 0.0, 0.0],
        }];
        let panic = std::panic::catch_unwind(|| {
            derive_skin_cull_metadata(&[[0.0; 3]], Some(&weights));
        })
        .expect_err("a zero-weight slot 0 must panic: it is the blend pivot");
        assert!(panic_message(panic).contains("hemisphere pivot"));
    }

    #[test]
    fn skin_cull_metadata_retains_only_positive_joint_influences() {
        let positions = [[-2.0, 1.0, 3.0], [4.0, -1.0, 0.5], [7.0, 8.0, 9.0]];
        let weights = [
            JointWeights {
                joint_indices: [2, 50_000, 9, 12],
                weights: [0.75, 0.0, 0.25, 0.0],
            },
            JointWeights {
                joint_indices: [2, 60_000, 9, 12],
                weights: [0.5, 0.0, 0.5, 0.0],
            },
            JointWeights {
                joint_indices: [9, 65_000, 2, 12],
                weights: [1.0, 0.0, 0.0, 0.0],
            },
        ];
        let metadata = derive_skin_cull_metadata(&positions, Some(&weights)).unwrap();

        assert_eq!(metadata.joints.len(), 2);
        assert_eq!(metadata.joints[0].joint_index, 2);
        assert_eq!(metadata.joints[0].aabb_min, [-2.0, -1.0, 0.5]);
        assert_eq!(metadata.joints[0].aabb_max, [4.0, 1.0, 3.0]);
        assert_eq!(metadata.joints[1].joint_index, 9);
        assert_eq!(metadata.joints[1].aabb_min, [-2.0, -1.0, 0.5]);
        assert_eq!(metadata.joints[1].aabb_max, [7.0, 8.0, 9.0]);
        assert!(
            metadata
                .joints
                .iter()
                .all(|bounds| bounds.joint_index < 50_000),
            "zero-weight high indices must not consume displacement metadata"
        );
        assert_eq!(metadata.pivot_pairs.as_ref(), &[(0, 1)]);
    }

    #[test]
    fn skin_cull_bound_covers_exact_dual_quaternion_blends_and_world_scale() {
        let positions = [
            [-1.5, 0.2, 0.7],
            [2.0, -0.5, 1.2],
            [0.3, 1.8, -0.9],
            [-0.7, -1.4, 2.1],
        ];
        let weights = [
            JointWeights {
                joint_indices: [0, 1, 2, 0],
                weights: [0.65, 0.35, 0.0, 0.0],
            },
            JointWeights {
                joint_indices: [1, 2, 0, 1],
                weights: [0.2, 0.800_05, 0.0, 0.0],
            },
            JointWeights {
                joint_indices: [0, 2, 1, 0],
                weights: [0.4, 0.6, 0.0, 0.0],
            },
            JointWeights {
                joint_indices: [0, 1, 2, 0],
                weights: [0.2, 0.3, 0.5, 0.0],
            },
        ];
        let palette = [
            DualQuat::from_mat4(Mat4::from_rotation_translation(
                Quat::from_rotation_z(0.65),
                Vec3::new(1.1, -0.4, 0.7),
            )),
            DualQuat::from_mat4(Mat4::from_rotation_translation(
                Quat::from_rotation_x(-0.4),
                Vec3::new(-0.3, 0.9, -0.6),
            )),
            DualQuat::from_mat4(Mat4::from_rotation_translation(
                Quat::from_euler(abi_core::glam::EulerRot::XYZ, 0.2, -0.5, 0.8),
                Vec3::new(-0.6, 0.8, -0.2),
            )),
        ];
        let skin = MeshSkin::new(palette.into());
        let model_to_world = Mat4::from_scale_rotation_translation(
            Vec3::new(2.5, 0.75, 1.4),
            Quat::from_rotation_y(0.3),
            Vec3::new(4.0, -2.0, 7.0),
        );
        let metadata = derive_skin_cull_metadata(&positions, Some(&weights)).unwrap();
        let dilation = skin_bounds_dilation(&metadata, &skin, &model_to_world);
        assert_ne!(weights[1].weights.iter().sum::<f32>(), 1.0);

        for (vertex, (&position, weights)) in positions.iter().zip(&weights).enumerate() {
            let p = Vec3::from_array(position);
            let skinned = abi_mesh::evaluate_vertex_position(
                skin.transforms(),
                std::slice::from_ref(weights),
                0,
                GpuPtr::<DeformerStack>::null(),
                0,
                p,
            );
            let exact_world_displacement = model_to_world.transform_vector3(skinned - p).length();
            assert!(
                exact_world_displacement <= dilation + 2.0e-5,
                "vertex {vertex} displacement {exact_world_displacement} exceeds dilation {dilation}"
            );
        }
    }

    #[test]
    fn identity_skin_palette_has_exact_zero_dilation() {
        let positions = [[-2.0, 0.5, 1.0], [3.0, -1.0, 0.25]];
        let weights = [
            JointWeights {
                joint_indices: [0, 1, 0, 1],
                weights: [0.25, 0.75, 0.0, 0.0],
            },
            JointWeights {
                joint_indices: [1, 0, 1, 0],
                weights: [0.625, 0.375, 0.0, 0.0],
            },
        ];
        let metadata = derive_skin_cull_metadata(&positions, Some(&weights)).unwrap();
        let skin = MeshSkin::new(vec![DualQuat::IDENTITY; 2].into_boxed_slice());
        let model_to_world = Mat4::from_scale(Vec3::new(3.0, 0.5, 1.25));

        assert_eq!(skin_bounds_dilation(&metadata, &skin, &model_to_world), 0.0);
    }

    /// Rejects a near-antipodal co-influencing joint pair.
    #[test]
    fn near_antipodal_pivot_pair_refuses_to_bound() {
        let positions = [[0.5, 0.25, -0.75]];
        let weights = [JointWeights {
            joint_indices: [0, 1, 0, 0],
            weights: [0.5, 0.5, 0.0, 0.0],
        }];
        let metadata = derive_skin_cull_metadata(&positions, Some(&weights)).unwrap();
        let skin = MeshSkin::new(
            vec![
                DualQuat::IDENTITY,
                DualQuat::from_rotation_translation(
                    Quat::from_rotation_z(std::f32::consts::PI * 0.997),
                    Vec3::ZERO,
                ),
            ]
            .into_boxed_slice(),
        );
        let panic = std::panic::catch_unwind(|| {
            skin_bounds_dilation(&metadata, &skin, &Mat4::IDENTITY);
        })
        .expect_err("a near-antipodal pivot pair must panic");
        assert!(panic_message(panic).contains("near-antipodal"));
    }

    #[test]
    fn skinned_flags_are_derived_and_cannot_be_authored() {
        let flags = derive_mesh_instance_flags(MESH_FLAG_NO_LINEWORK, false, true);
        assert_eq!(
            flags,
            MESH_FLAG_NO_LINEWORK | MESH_FLAG_SKINNED | MESH_FLAG_DYNAMIC | MESH_FLAG_NO_SHADOW
        );
        assert_eq!(derive_mesh_instance_flags(0, false, false), 0);

        for derived in [MESH_FLAG_SKINNED, MESH_FLAG_DYNAMIC] {
            let panic = std::panic::catch_unwind(|| {
                derive_mesh_instance_flags(derived, false, false);
            })
            .expect_err("derived MeshInstanceFlags bit must panic");
            assert!(panic_message(panic).contains("may not author derived"));
        }
    }

    #[test]
    fn palettes_copy_to_aligned_disjoint_subruns_and_static_stays_null() {
        #[repr(align(16))]
        struct AlignedPalette([DualQuat; 5]);

        let entry = |seed: f32| DualQuat {
            real: [0.0, 0.0, 0.0, 1.0],
            dual: [seed, seed + 1.0, seed + 2.0, 0.0],
        };
        let first = MeshSkin::new(vec![entry(1.0), entry(4.0)].into_boxed_slice());
        let second = MeshSkin::new(vec![entry(7.0), entry(10.0), entry(13.0)].into_boxed_slice());
        let mut storage = AlignedPalette([DualQuat::IDENTITY; 5]);
        let mut dst = storage.0.as_mut_ptr();
        let mut written = 0usize;
        let base = GpuPtr::from_addr(0x1000);

        // SAFETY: storage has room for exactly the two palettes (2 + 3).
        let static_ptr = unsafe { write_mesh_skin(base, &mut dst, &mut written, None) };
        assert!(static_ptr.is_null());
        assert_eq!(written, 0);
        // SAFETY: first occupies entries 0..2 of the five-entry storage.
        let first_ptr = unsafe { write_mesh_skin(base, &mut dst, &mut written, Some(&first)) };
        // SAFETY: second occupies the remaining entries 2..5.
        let second_ptr = unsafe { write_mesh_skin(base, &mut dst, &mut written, Some(&second)) };

        assert_eq!(first_ptr.addr(), 0x1000);
        assert_eq!(
            second_ptr.addr(),
            0x1000 + (2 * size_of::<DualQuat>()) as u64
        );
        assert_eq!(first_ptr.addr() % 16, 0);
        assert_eq!(second_ptr.addr() % 16, 0);
        assert_eq!((storage.0.as_ptr() as usize) % 16, 0);
        assert_eq!(written, 5);
        for (copied, expected) in storage.0[..2].iter().zip(first.transforms()) {
            assert_eq!((copied.real, copied.dual), (expected.real, expected.dual));
        }
        for (copied, expected) in storage.0[2..].iter().zip(second.transforms()) {
            assert_eq!((copied.real, copied.dual), (expected.real, expected.dual));
        }
    }

    #[test]
    fn mesh_extract_composes_skin_and_deformer_dilation_with_palette_state() {
        let _guard = GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let gpu = Gpu::new(true).expect("vulkan init");
        let arena_buf = gpu.alloc_slice::<u8>(64 << 10, Memory::Default);
        let mut frame = Frame {
            arena: Arena {
                buf: arena_buf,
                cap: 64 << 10,
                offset: 0,
            },
            extracted: Extracted::default(),
            frame: 1,
            time: 0.0,
            extent: [1, 1],
            mesh_uploads: Vec::new(),
            mesh_updates: Vec::new(),
            mesh_removals: Vec::new(),
            material_uploads: Vec::new(),
            shader_group_uploads: Vec::new(),
            proc_texture_uploads: Vec::new(),
        };
        let mut world = World::new();
        world.insert_resource(Assets::<Mesh>::default());
        world.insert_resource(Messages::<AssetEvent<Mesh>>::default());
        world.insert_resource(MeshMaterials::default());
        world.insert_resource(ShaderGroups::default());

        let static_mesh = world.resource_mut::<Assets<Mesh>>().add(triangle());
        let mut weighted = triangle();
        weighted.insert_attribute(
            Mesh::ATTRIBUTE_JOINT_INDEX,
            VertexAttributeValues::Uint16x4(vec![[0, 3, 2, 1]; 3]),
        );
        weighted.insert_attribute(
            Mesh::ATTRIBUTE_JOINT_WEIGHT,
            VertexAttributeValues::Float32x4(vec![[1.0, 0.0, 0.0, 0.0]; 3]),
        );
        let weighted_mesh = world.resource_mut::<Assets<Mesh>>().add(weighted);
        let material = world
            .resource_mut::<MeshMaterials>()
            .add(MaterialEntry::standard());
        let identity = DualQuat::IDENTITY;
        let palette = [
            DualQuat::from_mat4(Mat4::from_translation(Vec3::new(2.0, 0.0, 0.0))),
            identity,
            identity,
            identity,
        ];
        let mut deformer = DeformerStack::zeroed();
        deformer.count = 1;
        deformer.lattices[0].model_to_lattice = Mat4::IDENTITY;
        deformer.lattices[0].lattice_to_model = Mat4::IDENTITY;
        deformer.lattices[0].resolution = [2; 3];
        for offset in &mut deformer.lattices[0].offsets[..8] {
            *offset = [0.25, 0.0, 0.0, 0.0];
        }
        world.spawn((
            Mesh3d(weighted_mesh),
            GlobalTransform::default(),
            material,
            MeshSkin::new(palette.into()),
            deformer,
        ));
        world.spawn((Mesh3d(static_mesh), GlobalTransform::default(), material));
        world.clear_trackers();

        let mut extract = make_mesh_extract();
        extract(&mut world, &mut frame);
        let instances = frame.extracted.get_host::<MeshInstance>();
        assert_eq!(instances.len(), 2);
        let skinned = instances
            .iter()
            .find(|instance| instance.flags & MESH_FLAG_SKINNED != 0)
            .expect("weighted entity must publish one skinned instance");
        let static_instance = instances
            .iter()
            .find(|instance| instance.flags & MESH_FLAG_SKINNED == 0)
            .expect("static entity must publish one static instance");
        assert_eq!(
            skinned.flags,
            MESH_FLAG_SKINNED | MESH_FLAG_DYNAMIC | MESH_FLAG_NO_SHADOW
        );
        assert!(static_instance.joint_transforms.is_null());
        assert_eq!(static_instance.flags, 0);
        assert_eq!(skinned.joint_transforms.addr() % 16, 0);
        assert_eq!(skinned.deformer_slot, 1);
        assert_eq!(
            skinned.bounds_dilation, 2.25,
            "2 units of skin translation and 0.25 units of later deformation must add"
        );
        assert_eq!(static_instance.bounds_dilation, 0.0);
        let palette_offset =
            usize::try_from(skinned.joint_transforms.addr() - arena_buf.gpu.addr())
                .expect("frame palette offset fits usize");
        // SAFETY: the extracted pointer names the four-matrix palette subrun
        // in this still-owned mapped frame arena.
        let copied = unsafe {
            std::slice::from_raw_parts(
                arena_buf.cpu.add(palette_offset).cast::<DualQuat>(),
                palette.len(),
            )
        };
        assert_eq!((copied.as_ptr() as usize) % 16, 0);
        for (copied, expected) in copied.iter().zip(palette) {
            assert_eq!((copied.real, copied.dual), (expected.real, expected.dual));
        }

        drop(extract);
        drop(frame);
        gpu.free(arena_buf);
    }

    #[test]
    fn static_mesh_has_no_joint_weight_allocation() {
        let upload = convert_mesh(AssetId::invalid(), &triangle(), 0);
        assert!(upload.joint_weights.is_none());
    }

    #[test]
    fn exact_joint_attribute_pair_converts_to_shared_payload() {
        let mut mesh = triangle();
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_JOINT_INDEX,
            VertexAttributeValues::Uint16x4(vec![[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]),
        );
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_JOINT_WEIGHT,
            VertexAttributeValues::Float32x4(vec![
                [1.0, 0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0, 0.0],
                [0.25; 4],
            ]),
        );

        let upload = convert_mesh(AssetId::invalid(), &mesh, 0);
        let weights = upload.joint_weights.expect("valid pair must convert");
        assert_eq!(weights.len(), 3);
        assert_eq!(weights[1].joint_indices, [4, 5, 6, 7]);
        assert_eq!(weights[1].weights, [0.5, 0.5, 0.0, 0.0]);
    }

    #[test]
    fn generic_bevy_joint_weights_are_strictly_validated() {
        for (bad_weights, expected) in [
            ([f32::NAN, 0.0, 0.0, 0.0], "must be finite"),
            ([1.1, -0.1, 0.0, 0.0], "must be in [0, 1]"),
            ([0.5, 0.25, 0.0, 0.0], "sum must be within 1e-4"),
        ] {
            let mut mesh = triangle();
            mesh.insert_attribute(
                Mesh::ATTRIBUTE_JOINT_INDEX,
                VertexAttributeValues::Uint16x4(vec![[0; 4]; 3]),
            );
            let mut weights = vec![[1.0, 0.0, 0.0, 0.0]; 3];
            weights[0] = bad_weights;
            mesh.insert_attribute(
                Mesh::ATTRIBUTE_JOINT_WEIGHT,
                VertexAttributeValues::Float32x4(weights),
            );
            let panic = std::panic::catch_unwind(|| {
                convert_mesh(AssetId::invalid(), &mesh, 0);
            })
            .expect_err("malformed generic bevy weights must panic");
            assert!(panic_message(panic).contains(expected));
        }
    }

    #[test]
    fn joint_attributes_must_be_paired() {
        let mut joints_only = triangle();
        joints_only.insert_attribute(
            Mesh::ATTRIBUTE_JOINT_INDEX,
            VertexAttributeValues::Uint16x4(vec![[0; 4]; 3]),
        );
        let panic = std::panic::catch_unwind(|| {
            convert_mesh(AssetId::invalid(), &joints_only, 0);
        })
        .expect_err("JOINT_INDEX without JOINT_WEIGHT must panic");
        assert!(panic_message(panic).contains("must be present together"));

        let mut weights_only = triangle();
        weights_only.insert_attribute(
            Mesh::ATTRIBUTE_JOINT_WEIGHT,
            VertexAttributeValues::Float32x4(vec![[0.25; 4]; 3]),
        );
        let panic = std::panic::catch_unwind(|| {
            convert_mesh(AssetId::invalid(), &weights_only, 0);
        })
        .expect_err("JOINT_WEIGHT without JOINT_INDEX must panic");
        assert!(panic_message(panic).contains("must be present together"));
    }

    #[test]
    fn joint_attributes_require_exact_types_and_vertex_counts() {
        let mut wrong_type = triangle();
        wrong_type.insert_attribute(
            bevy::mesh::MeshVertexAttribute {
                format: bevy::mesh::VertexFormat::Uint32x4,
                ..Mesh::ATTRIBUTE_JOINT_INDEX
            },
            VertexAttributeValues::Uint32x4(vec![[0; 4]; 3]),
        );
        wrong_type.insert_attribute(
            Mesh::ATTRIBUTE_JOINT_WEIGHT,
            VertexAttributeValues::Float32x4(vec![[0.25; 4]; 3]),
        );
        let panic = std::panic::catch_unwind(|| {
            convert_mesh(AssetId::invalid(), &wrong_type, 0);
        })
        .expect_err("Uint32x4 joint indices must panic");
        assert!(panic_message(panic).contains("joint indices must be Uint16x4"));

        let mut wrong_weight_type = triangle();
        wrong_weight_type.insert_attribute(
            Mesh::ATTRIBUTE_JOINT_INDEX,
            VertexAttributeValues::Uint16x4(vec![[0; 4]; 3]),
        );
        wrong_weight_type.insert_attribute(
            bevy::mesh::MeshVertexAttribute {
                format: bevy::mesh::VertexFormat::Float32x3,
                ..Mesh::ATTRIBUTE_JOINT_WEIGHT
            },
            VertexAttributeValues::Float32x3(vec![[0.0; 3]; 3]),
        );
        let panic = std::panic::catch_unwind(|| {
            convert_mesh(AssetId::invalid(), &wrong_weight_type, 0);
        })
        .expect_err("Float32x3 joint weights must panic");
        assert!(panic_message(panic).contains("joint weights must be Float32x4"));

        let mut wrong_index_count = triangle();
        wrong_index_count.insert_attribute(
            Mesh::ATTRIBUTE_JOINT_INDEX,
            VertexAttributeValues::Uint16x4(vec![[0; 4]; 2]),
        );
        wrong_index_count.insert_attribute(
            Mesh::ATTRIBUTE_JOINT_WEIGHT,
            VertexAttributeValues::Float32x4(vec![[0.25; 4]; 3]),
        );
        let panic = std::panic::catch_unwind(|| {
            convert_mesh(AssetId::invalid(), &wrong_index_count, 0);
        })
        .expect_err("non-vertex-parallel joint indices must panic");
        assert!(panic_message(panic).contains("joint index count must match positions"));

        let mut wrong_weight_count = triangle();
        wrong_weight_count.insert_attribute(
            Mesh::ATTRIBUTE_JOINT_INDEX,
            VertexAttributeValues::Uint16x4(vec![[0; 4]; 3]),
        );
        wrong_weight_count.insert_attribute(
            Mesh::ATTRIBUTE_JOINT_WEIGHT,
            VertexAttributeValues::Float32x4(vec![[0.25; 4]; 2]),
        );
        let panic = std::panic::catch_unwind(|| {
            convert_mesh(AssetId::invalid(), &wrong_weight_count, 0);
        })
        .expect_err("non-vertex-parallel joint weights must panic");
        assert!(panic_message(panic).contains("joint weight count must match positions"));
    }

    /// Verifies removal, slot reuse, and in-place modification lanes.
    #[test]
    fn mesh_lifecycle_frees_reuses_and_updates_slots() {
        let _guard = GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let gpu = Gpu::new(true).expect("vulkan init");
        let arena_buf = gpu.alloc_slice::<u8>(64 << 10, Memory::Default);
        let mut frame = Frame {
            arena: Arena {
                buf: arena_buf,
                cap: 64 << 10,
                offset: 0,
            },
            extracted: Extracted::default(),
            frame: 1,
            time: 0.0,
            extent: [1, 1],
            mesh_uploads: Vec::new(),
            mesh_updates: Vec::new(),
            mesh_removals: Vec::new(),
            material_uploads: Vec::new(),
            shader_group_uploads: Vec::new(),
            proc_texture_uploads: Vec::new(),
        };
        let mut world = World::new();
        world.insert_resource(Assets::<Mesh>::default());
        world.insert_resource(Messages::<AssetEvent<Mesh>>::default());
        world.insert_resource(MeshMaterials::default());
        world.insert_resource(ShaderGroups::default());
        let mut extract = make_mesh_extract();

        let first = world.resource_mut::<Assets<Mesh>>().add(triangle());
        let second = world.resource_mut::<Assets<Mesh>>().add(triangle());
        world.clear_trackers();
        extract(&mut world, &mut frame);
        let mut assigned: Vec<u32> = frame.mesh_uploads.iter().map(|u| u.index).collect();
        assigned.sort_unstable();
        assert_eq!(assigned, vec![0, 1], "two adds take the two fresh slots");
        frame.mesh_uploads.clear();
        assert!(frame.mesh_updates.is_empty() && frame.mesh_removals.is_empty());

        world.clear_trackers();
        extract(&mut world, &mut frame);
        assert!(
            frame.mesh_uploads.is_empty()
                && frame.mesh_updates.is_empty()
                && frame.mesh_removals.is_empty(),
            "a frame that changed nothing must produce no lifecycle traffic"
        );

        {
            let mut assets = world.resource_mut::<Assets<Mesh>>();
            let mut mesh = assets.get_mut(&first).expect("first asset is live");
            mesh.insert_attribute(
                Mesh::ATTRIBUTE_POSITION,
                vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            );
        }
        world
            .resource_mut::<Messages<AssetEvent<Mesh>>>()
            .write(AssetEvent::Modified { id: first.id() });
        world.clear_trackers();
        extract(&mut world, &mut frame);
        assert_eq!(frame.mesh_updates.len(), 1, "Modified must reach the seam");
        assert_eq!(frame.mesh_updates[0].0.positions[1], [2.0, 0.0, 0.0]);
        let first_slot = frame.mesh_updates[0].0.index;
        assert!(frame.mesh_uploads.is_empty() && frame.mesh_removals.is_empty());
        frame.mesh_updates.clear();

        let first_id = first.id();
        world.resource_mut::<Assets<Mesh>>().remove(&first);
        world
            .resource_mut::<Messages<AssetEvent<Mesh>>>()
            .write(AssetEvent::Removed { id: first_id });
        world.clear_trackers();
        extract(&mut world, &mut frame);
        assert_eq!(frame.mesh_removals.len(), 1);
        assert_eq!(frame.mesh_removals[0].index, first_slot);
        assert!(frame.mesh_uploads.is_empty() && frame.mesh_updates.is_empty());
        frame.mesh_removals.clear();

        let third = world.resource_mut::<Assets<Mesh>>().add(triangle());
        world.clear_trackers();
        extract(&mut world, &mut frame);
        assert_eq!(frame.mesh_uploads.len(), 1);
        assert_eq!(
            frame.mesh_uploads[0].index, first_slot,
            "a new asset must reclaim the freed slot, not grow the table"
        );
        frame.mesh_uploads.clear();

        let ephemeral = world.resource_mut::<Assets<Mesh>>().add(triangle());
        let ephemeral_id = ephemeral.id();
        world.resource_mut::<Assets<Mesh>>().remove(&ephemeral);
        world
            .resource_mut::<Messages<AssetEvent<Mesh>>>()
            .write(AssetEvent::Removed { id: ephemeral_id });
        world.clear_trackers();
        extract(&mut world, &mut frame);
        assert!(
            frame.mesh_uploads.is_empty() && frame.mesh_removals.is_empty(),
            "an asset that never reached the scan owns no slot"
        );

        drop(second);
        drop(third);
        drop(extract);
        drop(frame);
        gpu.free(arena_buf);
    }

    fn panic_message(panic: Box<dyn std::any::Any + Send>) -> String {
        panic
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| {
                panic
                    .downcast_ref::<&str>()
                    .map(|message| (*message).to_owned())
            })
            .unwrap_or_default()
    }
}
