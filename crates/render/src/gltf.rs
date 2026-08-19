//! glTF loaders that produce the mesh layouts accepted by this crate.
//!
//! Static imports produce TriangleList data with Float32x3 positions and
//! normals, Float32x2 UV0, u32 indices, and supported optional attributes.
//! They walk the default scene, bake node transforms, rotate normals and
//! tangents, and reject skinned nodes. Skinned imports accept exactly one
//! identity-transformed mesh node, one
//! primitive, one skin, four influences, embedded buffers, explicit inverse
//! binds, and an ascending parent-before-child palette. All parsing and
//! contract failures return [`GltfMeshError`]; they never panic.

use std::fmt;

use abi_mesh::JointWeights;
use bevy::{
    asset::{AssetApp, AssetLoader, LoadContext, RenderAssetUsages, io::Reader},
    math::{Mat3, Mat4, Quat, Vec3},
    mesh::{Indices, Mesh, PrimitiveTopology, VertexAttributeValues},
    prelude::App,
    reflect::TypePath,
};

/// One joint in palette order with column-major glTF matrices.
#[derive(Clone, Debug, PartialEq)]
pub struct GltfSkinJoint {
    pub name: String,
    pub parent: Option<u16>,
    pub bind_matrix: [[f32; 4]; 4],
    pub inverse_bind_matrix: [[f32; 4]; 4],
}

/// Cold-path skinned mesh plus vertex-parallel weights and joint palette.
pub struct SkinnedGltf {
    pub mesh: Mesh,
    pub joint_weights: Vec<JointWeights>,
    pub joints: Vec<GltfSkinJoint>,
}

/// Registers the `.gltf` and `.glb` mesh asset loader.
pub fn plugin(app: &mut App) {
    app.init_asset::<Mesh>()
        .init_asset_loader::<GltfMeshLoader>();
}

#[derive(Default, TypePath)]
struct GltfMeshLoader;

impl AssetLoader for GltfMeshLoader {
    type Asset = Mesh;
    type Settings = ();
    type Error = GltfMeshError;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &(),
        _load_context: &mut LoadContext<'_>,
    ) -> Result<Mesh, GltfMeshError> {
        let mut bytes = Vec::new();
        reader
            .read_to_end(&mut bytes)
            .await
            .map_err(|error| GltfMeshError::new(format!("read glTF: {error}")))?;
        gltf_to_mesh(&bytes)
    }

    fn extensions(&self) -> &[&str] {
        &["gltf", "glb"]
    }
}

#[derive(Debug)]
pub struct GltfMeshError(String);

impl GltfMeshError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for GltfMeshError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for GltfMeshError {}

/// Produces one indexed TriangleList with accepted static attributes.
///
/// The default scene hierarchy is flattened; malformed input returns an error.
pub fn gltf_to_mesh(bytes: &[u8]) -> Result<Mesh, GltfMeshError> {
    let gltf::Gltf { document, blob } = gltf::Gltf::from_slice(bytes)
        .map_err(|error| GltfMeshError::new(format!("parse glTF: {error}")))?;
    let buffers = gltf::import_buffers(&document, None, blob)
        .map_err(|error| GltfMeshError::new(format!("load glTF buffers: {error}")))?;
    let scene = document
        .default_scene()
        .ok_or_else(|| GltfMeshError::new("glTF has no default scene"))?;
    let mut output = FlattenedMesh::default();
    let mut ancestors = Vec::new();

    for node in scene.nodes() {
        append_node(
            node,
            Mat4::IDENTITY,
            Quat::IDENTITY,
            &buffers,
            &mut output,
            &mut ancestors,
        )?;
    }

    output.into_mesh()
}

fn append_node(
    node: gltf::Node<'_>,
    parent_transform: Mat4,
    parent_rotation: Quat,
    buffers: &[gltf::buffer::Data],
    output: &mut FlattenedMesh,
    ancestors: &mut Vec<usize>,
) -> Result<(), GltfMeshError> {
    let node_index = node.index();
    if ancestors.contains(&node_index) {
        return Err(GltfMeshError::new(format!(
            "node {node_index} forms a cycle in the scene hierarchy"
        )));
    }
    ancestors.push(node_index);

    let local_transform = Mat4::from_cols_array_2d(&node.transform().matrix());
    if !local_transform.is_finite() {
        return Err(GltfMeshError::new(format!(
            "node {node_index} has a non-finite transform"
        )));
    }
    let world_transform = parent_transform * local_transform;
    if !world_transform.is_finite() {
        return Err(GltfMeshError::new(format!(
            "node {node_index} has a non-finite world transform"
        )));
    }
    let world_rotation = normalize_rotation(
        parent_rotation * node_rotation(&node)?,
        format!("node {node_index} has an invalid world rotation"),
    )?;

    if node.skin().is_some() {
        return Err(GltfMeshError::new(format!(
            "node {node_index} is skinned; use gltf_to_skinned_mesh instead of the static loader"
        )));
    }
    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            append_primitive(
                primitive,
                node_index,
                world_transform,
                world_rotation,
                buffers,
                output,
            )?;
        }
    }
    for child in node.children() {
        append_node(
            child,
            world_transform,
            world_rotation,
            buffers,
            output,
            ancestors,
        )?;
    }

    ancestors.pop();
    Ok(())
}

/// Imports one supported skinned mesh and its pose palette.
///
/// This is pure cold-path parsing; frame palette extraction is separate.
pub fn gltf_to_skinned_mesh(bytes: &[u8]) -> Result<SkinnedGltf, GltfMeshError> {
    let gltf::Gltf { document, blob } = gltf::Gltf::from_slice(bytes)
        .map_err(|error| GltfMeshError::new(format!("parse glTF: {error}")))?;
    let buffers = gltf::import_buffers(&document, None, blob)
        .map_err(|error| GltfMeshError::new(format!("load glTF buffers: {error}")))?;
    let scene = document
        .default_scene()
        .ok_or_else(|| GltfMeshError::new("glTF has no default scene"))?;
    let hierarchy = SceneHierarchy::build(&document, scene)?;

    let mut skinned_nodes = document
        .nodes()
        .filter(|node| hierarchy.world[node.index()].is_some() && node.skin().is_some());
    let mesh_node = skinned_nodes
        .next()
        .ok_or_else(|| GltfMeshError::new("default scene has no skinned mesh node"))?;
    if let Some(extra) = skinned_nodes.next() {
        return Err(GltfMeshError::new(format!(
            "default scene has more than one skinned mesh node (including node {})",
            extra.index()
        )));
    }
    if document.skins().len() != 1 {
        return Err(GltfMeshError::new(format!(
            "skinned import requires exactly one skin, found {}",
            document.skins().len()
        )));
    }
    let mesh_world = hierarchy.world[mesh_node.index()].expect("filtered reachable node");
    if max_identity_error(mesh_world) > 1.0e-6 {
        return Err(GltfMeshError::new(format!(
            "skinned mesh node {} must have an identity world transform",
            mesh_node.index()
        )));
    }

    let skin = mesh_node
        .skin()
        .expect("selected node is known to reference a skin");
    let source_mesh = mesh_node.mesh().ok_or_else(|| {
        GltfMeshError::new(format!(
            "skinned node {} does not reference a mesh",
            mesh_node.index()
        ))
    })?;
    let mut primitives = source_mesh.primitives();
    let primitive = primitives.next().ok_or_else(|| {
        GltfMeshError::new(format!("mesh {} has no primitives", source_mesh.index()))
    })?;
    if primitives.next().is_some() {
        return Err(GltfMeshError::new(format!(
            "skinned mesh {} must contain exactly one primitive",
            source_mesh.index()
        )));
    }
    if primitive.morph_targets().next().is_some() {
        return Err(GltfMeshError::new(
            "skinned mesh morph targets are not supported",
        ));
    }

    let reader =
        primitive.reader(|buffer| buffers.get(buffer.index()).map(|data| data.0.as_slice()));
    if reader.read_joints(1).is_some() || reader.read_weights(1).is_some() {
        return Err(GltfMeshError::new(
            "skinned mesh supports exactly JOINTS_0/WEIGHTS_0 (four influences)",
        ));
    }
    let joint_indices = reader
        .read_joints(0)
        .ok_or_else(|| GltfMeshError::new("skinned mesh is missing JOINTS_0"))?
        .into_u16()
        .collect::<Vec<_>>();
    let weights = reader
        .read_weights(0)
        .ok_or_else(|| GltfMeshError::new("skinned mesh is missing WEIGHTS_0"))?
        .into_f32()
        .collect::<Vec<_>>();

    let skin_nodes = skin.joints().collect::<Vec<_>>();
    if skin_nodes.is_empty() {
        return Err(GltfMeshError::new("skin joint palette is empty"));
    }
    if skin_nodes.len() > u16::MAX as usize {
        return Err(GltfMeshError::new("skin joint palette exceeds u16"));
    }
    let inverse_binds = skin
        .reader(|buffer| buffers.get(buffer.index()).map(|data| data.0.as_slice()))
        .read_inverse_bind_matrices()
        .ok_or_else(|| GltfMeshError::new("skin is missing inverseBindMatrices"))?
        .collect::<Vec<_>>();
    if inverse_binds.len() != skin_nodes.len() {
        return Err(GltfMeshError::new(format!(
            "inverse-bind count {} does not match joint count {}",
            inverse_binds.len(),
            skin_nodes.len()
        )));
    }

    let mut node_to_palette = vec![None; document.nodes().len()];
    for (palette_index, node) in skin_nodes.iter().enumerate() {
        if hierarchy.world[node.index()].is_none() {
            return Err(GltfMeshError::new(format!(
                "joint node {} is not reachable from the default scene",
                node.index()
            )));
        }
        if node_to_palette[node.index()]
            .replace(palette_index as u16)
            .is_some()
        {
            return Err(GltfMeshError::new(format!(
                "joint node {} appears more than once in the skin palette",
                node.index()
            )));
        }
    }

    let mut names = std::collections::HashSet::with_capacity(skin_nodes.len());
    let mut joints = Vec::with_capacity(skin_nodes.len());
    let mut roots = 0usize;
    for (palette_index, (node, inverse_bind)) in skin_nodes.iter().zip(inverse_binds).enumerate() {
        let name = node
            .name()
            .filter(|name| !name.is_empty())
            .ok_or_else(|| GltfMeshError::new(format!("joint node {} has no name", node.index())))?
            .to_owned();
        if !names.insert(name.clone()) {
            return Err(GltfMeshError::new(format!(
                "joint name {name:?} appears more than once"
            )));
        }
        let mut ancestor = hierarchy.parent[node.index()];
        let mut parent = None;
        while let Some(node_index) = ancestor {
            if let Some(index) = node_to_palette[node_index] {
                parent = Some(index);
                break;
            }
            ancestor = hierarchy.parent[node_index];
        }
        if let Some(parent) = parent {
            if parent as usize >= palette_index {
                return Err(GltfMeshError::new(format!(
                    "joint {name:?} precedes its parent in the skin palette"
                )));
            }
        } else {
            roots += 1;
        }

        let bind = hierarchy.world[node.index()].expect("joint reachability checked");
        let inverse_bind_matrix = Mat4::from_cols_array_2d(&inverse_bind);
        if !inverse_bind_matrix.is_finite() {
            return Err(GltfMeshError::new(format!(
                "joint {name:?} has a non-finite inverse bind"
            )));
        }
        let bind_error = max_identity_error(bind * inverse_bind_matrix);
        if bind_error > 1.0e-4 {
            return Err(GltfMeshError::new(format!(
                "joint {name:?} inverse bind disagrees with its bind transform ({bind_error})"
            )));
        }
        joints.push(GltfSkinJoint {
            name,
            parent,
            bind_matrix: bind.to_cols_array_2d(),
            inverse_bind_matrix: inverse_bind,
        });
    }
    if roots != 1 {
        return Err(GltfMeshError::new(format!(
            "skin palette must have exactly one root, found {roots}"
        )));
    }

    if joint_indices.len() != weights.len() {
        return Err(GltfMeshError::new(format!(
            "JOINTS_0 count {} does not match WEIGHTS_0 count {}",
            joint_indices.len(),
            weights.len()
        )));
    }
    validate_joint_weights(&joint_indices, &weights, joints.len())?;
    let joint_weights = joint_indices
        .iter()
        .zip(&weights)
        .map(|(indices, weights)| JointWeights::canonical(indices.map(u32::from), *weights))
        .collect::<Vec<_>>();

    let mut output = FlattenedMesh::default();
    append_primitive(
        primitive,
        mesh_node.index(),
        Mat4::IDENTITY,
        Quat::IDENTITY,
        &buffers,
        &mut output,
    )?;
    if output.positions.len() != joint_weights.len() {
        return Err(GltfMeshError::new(format!(
            "skin attribute count {} does not match vertex count {}",
            joint_weights.len(),
            output.positions.len()
        )));
    }
    let mut mesh = output.into_mesh()?;
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_JOINT_INDEX,
        VertexAttributeValues::Uint16x4(joint_indices),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_JOINT_WEIGHT, weights);

    Ok(SkinnedGltf {
        mesh,
        joint_weights,
        joints,
    })
}

fn validate_joint_weights(
    joint_indices: &[[u16; 4]],
    weights: &[[f32; 4]],
    joint_count: usize,
) -> Result<(), GltfMeshError> {
    let mut used = vec![false; joint_count];
    for (vertex, (indices, weights)) in joint_indices.iter().zip(weights).enumerate() {
        let mut sum = 0.0f32;
        for influence in 0..4 {
            let index = indices[influence] as usize;
            let weight = weights[influence];
            if index >= joint_count {
                return Err(GltfMeshError::new(format!(
                    "vertex {vertex} influence {influence} references joint {index}, palette has {joint_count}"
                )));
            }
            if !weight.is_finite() || !(0.0..=1.0).contains(&weight) {
                return Err(GltfMeshError::new(format!(
                    "vertex {vertex} influence {influence} has invalid weight {weight}"
                )));
            }
            if weight > 0.0 {
                used[index] = true;
            }
            sum += weight;
        }
        if (sum - 1.0).abs() > 1.0e-4 {
            return Err(GltfMeshError::new(format!(
                "vertex {vertex} weights sum to {sum}, expected 1"
            )));
        }
    }
    if let Some(unused) = used.iter().position(|used| !used) {
        return Err(GltfMeshError::new(format!(
            "skin palette joint {unused} has no positive vertex weight"
        )));
    }
    Ok(())
}

struct SceneHierarchy {
    world: Vec<Option<Mat4>>,
    parent: Vec<Option<usize>>,
}

impl SceneHierarchy {
    fn build(document: &gltf::Document, scene: gltf::Scene<'_>) -> Result<Self, GltfMeshError> {
        let node_count = document.nodes().len();
        let mut hierarchy = Self {
            world: vec![None; node_count],
            parent: vec![None; node_count],
        };
        let mut ancestors = Vec::new();
        for node in scene.nodes() {
            hierarchy.visit(node, None, Mat4::IDENTITY, &mut ancestors)?;
        }
        Ok(hierarchy)
    }

    fn visit(
        &mut self,
        node: gltf::Node<'_>,
        parent: Option<usize>,
        parent_world: Mat4,
        ancestors: &mut Vec<usize>,
    ) -> Result<(), GltfMeshError> {
        let index = node.index();
        if ancestors.contains(&index) {
            return Err(GltfMeshError::new(format!(
                "node {index} forms a cycle in the scene hierarchy"
            )));
        }
        if self.world[index].is_some() {
            return Err(GltfMeshError::new(format!(
                "node {index} is referenced more than once by the default scene"
            )));
        }
        let local = Mat4::from_cols_array_2d(&node.transform().matrix());
        let world = parent_world * local;
        if !world.is_finite() {
            return Err(GltfMeshError::new(format!(
                "node {index} has a non-finite world transform"
            )));
        }
        self.world[index] = Some(world);
        self.parent[index] = parent;
        ancestors.push(index);
        for child in node.children() {
            self.visit(child, Some(index), world, ancestors)?;
        }
        ancestors.pop();
        Ok(())
    }
}

fn max_identity_error(matrix: Mat4) -> f32 {
    matrix
        .to_cols_array()
        .into_iter()
        .zip(Mat4::IDENTITY.to_cols_array())
        .fold(0.0f32, |error, (actual, expected)| {
            error.max((actual - expected).abs())
        })
}

fn node_rotation(node: &gltf::Node<'_>) -> Result<Quat, GltfMeshError> {
    let node_index = node.index();
    let rotation = match node.transform() {
        gltf::scene::Transform::Decomposed { rotation, .. } => Quat::from_array(rotation),
        gltf::scene::Transform::Matrix { matrix } => rotation_from_matrix(matrix, node_index)?,
    };
    normalize_rotation(
        rotation,
        format!("node {node_index} has an invalid rotation"),
    )
}

fn rotation_from_matrix(matrix: [[f32; 4]; 4], node_index: usize) -> Result<Quat, GltfMeshError> {
    let x_axis = Vec3::from_array([matrix[0][0], matrix[0][1], matrix[0][2]])
        .try_normalize()
        .ok_or_else(|| GltfMeshError::new(format!("node {node_index} has a degenerate matrix")))?;
    let y_source = Vec3::from_array([matrix[1][0], matrix[1][1], matrix[1][2]]);
    let y_axis = (y_source - x_axis * y_source.dot(x_axis))
        .try_normalize()
        .ok_or_else(|| GltfMeshError::new(format!("node {node_index} has a degenerate matrix")))?;
    let z_axis = x_axis.cross(y_axis);
    let rotation = Quat::from_mat3(&Mat3::from_cols(x_axis, y_axis, z_axis));

    normalize_rotation(
        rotation,
        format!("node {node_index} has an invalid matrix rotation"),
    )
}

fn normalize_rotation(rotation: Quat, message: String) -> Result<Quat, GltfMeshError> {
    let length_squared = rotation.length_squared();
    if !rotation.is_finite() || !length_squared.is_finite() || length_squared <= 0.0 {
        return Err(GltfMeshError::new(message));
    }
    let rotation = rotation.normalize();
    if rotation.is_finite() {
        Ok(rotation)
    } else {
        Err(GltfMeshError::new(message))
    }
}

fn append_primitive(
    primitive: gltf::Primitive<'_>,
    node_index: usize,
    world_transform: Mat4,
    world_rotation: Quat,
    buffers: &[gltf::buffer::Data],
    output: &mut FlattenedMesh,
) -> Result<(), GltfMeshError> {
    let label = format!("node {node_index}, primitive {}", primitive.index());
    if primitive.mode() != gltf::mesh::Mode::Triangles {
        return Err(GltfMeshError::new(format!(
            "{label}: only TriangleList primitives are supported"
        )));
    }

    let reader =
        primitive.reader(|buffer| buffers.get(buffer.index()).map(|data| data.0.as_slice()));
    let positions = reader
        .read_positions()
        .ok_or_else(|| GltfMeshError::new(format!("{label}: missing POSITION Float32x3")))?
        .collect::<Vec<_>>();
    let normals = reader
        .read_normals()
        .ok_or_else(|| GltfMeshError::new(format!("{label}: missing NORMAL Float32x3")))?
        .collect::<Vec<_>>();
    let uvs = reader
        .read_tex_coords(0)
        .ok_or_else(|| GltfMeshError::new(format!("{label}: missing TEXCOORD_0 Float32x2")))?
        .into_f32()
        .collect::<Vec<_>>();
    let indices = reader
        .read_indices()
        .ok_or_else(|| GltfMeshError::new(format!("{label}: missing indices")))?
        .into_u32()
        .collect::<Vec<_>>();
    let tangents = reader
        .read_tangents()
        .map(|values| values.collect::<Vec<_>>());

    if positions.is_empty() {
        return Err(GltfMeshError::new(format!("{label}: positions are empty")));
    }
    if positions.len() != normals.len() || positions.len() != uvs.len() {
        return Err(GltfMeshError::new(format!(
            "{label}: POSITION, NORMAL, and TEXCOORD_0 counts must match"
        )));
    }
    if tangents
        .as_ref()
        .is_some_and(|values| values.len() != positions.len())
    {
        return Err(GltfMeshError::new(format!(
            "{label}: TANGENT count must match POSITION"
        )));
    }
    if indices.is_empty() || indices.len() % 3 != 0 {
        return Err(GltfMeshError::new(format!(
            "{label}: indices must be a non-empty triangle list"
        )));
    }
    if let Some(has_tangents) = output.has_tangents {
        if has_tangents != tangents.is_some() {
            return Err(GltfMeshError::new(format!(
                "{label}: tangent presence must match every merged primitive"
            )));
        }
    } else {
        output.has_tangents = Some(tangents.is_some());
        if tangents.is_some() {
            output.tangents = Some(Vec::new());
        }
    }

    let vertex_offset = u32::try_from(output.positions.len())
        .map_err(|_| GltfMeshError::new(format!("{label}: too many vertices")))?;
    for &index in &indices {
        if index as usize >= positions.len() {
            return Err(GltfMeshError::new(format!(
                "{label}: index {index} exceeds {} vertices",
                positions.len()
            )));
        }
        output.indices.push(
            index
                .checked_add(vertex_offset)
                .ok_or_else(|| GltfMeshError::new(format!("{label}: index offset overflow")))?,
        );
    }

    output.positions.extend(
        positions
            .into_iter()
            .map(|position| {
                let position = world_transform.transform_point3(Vec3::from_array(position));
                if position.is_finite() {
                    Ok(position.to_array())
                } else {
                    Err(GltfMeshError::new(format!(
                        "{label}: transformed POSITION is non-finite"
                    )))
                }
            })
            .collect::<Result<Vec<_>, _>>()?,
    );
    output.normals.extend(
        normals
            .into_iter()
            .map(|normal| rotate_direction(normal, world_rotation, &label, "NORMAL"))
            .collect::<Result<Vec<_>, _>>()?,
    );
    output.uvs.extend(uvs);
    if let Some(tangents) = tangents {
        let output_tangents = output.tangents.as_mut().ok_or_else(|| {
            GltfMeshError::new(format!("{label}: tangent storage was not initialized"))
        })?;
        output_tangents.extend(
            tangents
                .into_iter()
                .map(|tangent| {
                    if !tangent[3].is_finite() {
                        return Err(GltfMeshError::new(format!(
                            "{label}: TANGENT handedness is non-finite"
                        )));
                    }
                    let direction = rotate_direction(
                        [tangent[0], tangent[1], tangent[2]],
                        world_rotation,
                        &label,
                        "TANGENT",
                    )?;
                    Ok([direction[0], direction[1], direction[2], tangent[3]])
                })
                .collect::<Result<Vec<_>, _>>()?,
        );
    }

    Ok(())
}

fn rotate_direction(
    direction: [f32; 3],
    rotation: Quat,
    primitive: &str,
    attribute: &str,
) -> Result<[f32; 3], GltfMeshError> {
    (rotation * Vec3::from_array(direction))
        .try_normalize()
        .filter(|direction| direction.is_finite())
        .map(|direction| direction.to_array())
        .ok_or_else(|| GltfMeshError::new(format!("{primitive}: {attribute} is not normalizable")))
}

#[derive(Default)]
struct FlattenedMesh {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    tangents: Option<Vec<[f32; 4]>>,
    indices: Vec<u32>,
    has_tangents: Option<bool>,
}

impl FlattenedMesh {
    fn into_mesh(self) -> Result<Mesh, GltfMeshError> {
        if self.positions.is_empty() || self.indices.is_empty() {
            return Err(GltfMeshError::new("default scene has no indexed geometry"));
        }

        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, self.positions);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, self.normals);
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, self.uvs);
        if let Some(tangents) = self.tangents {
            mesh.insert_attribute(Mesh::ATTRIBUTE_TANGENT, tangents);
        }
        mesh.insert_indices(Indices::U32(self.indices));
        Ok(mesh)
    }
}

#[cfg(test)]
mod tests {
    use bevy::asset::AssetId;

    use super::*;
    use crate::meshes::convert_mesh;

    /// Two nodes sharing one quad mesh: node 0 at identity, node 1
    const FIXTURE: &str = r#"{
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0, 1]}],
        "nodes": [
            {"mesh": 0},
            {"mesh": 0, "translation": [0, 1, 0], "rotation": [0, 0.7071068, 0, 0.7071068]}
        ],
        "meshes": [{"primitives": [{
            "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
            "indices": 3
        }]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3",
             "min": [0, 0, 0], "max": [1, 1, 0]},
            {"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": 4, "type": "VEC2"},
            {"bufferView": 3, "componentType": 5123, "count": 6, "type": "SCALAR"}
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 48},
            {"buffer": 0, "byteOffset": 48, "byteLength": 48},
            {"buffer": 0, "byteOffset": 96, "byteLength": 32},
            {"buffer": 0, "byteOffset": 128, "byteLength": 12}
        ],
        "buffers": [{"byteLength": 142, "uri": "data:application/octet-stream;base64,AAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAACAPwAAgD8AAAAAAAAAAAAAgD8AAAAAAAAAAAAAAAAAAIA/AAAAAAAAAAAAAIA/AAAAAAAAAAAAAIA/AAAAAAAAAAAAAIA/AAAAAAAAAAAAAIA/AAAAAAAAgD8AAIA/AAAAAAAAgD8AAAEAAgAAAAIAAwAAAA=="}]
    }"#;

    #[test]
    fn fixture_round_trips_through_convert_mesh() {
        let mesh = gltf_to_mesh(FIXTURE.as_bytes()).unwrap();
        let upload = convert_mesh(AssetId::invalid(), &mesh, 0);

        assert!(upload.positions.len() == 8);
        assert!(upload.normals.len() == 8);
        assert!(upload.uvs.len() == 8);
        assert!(upload.indices.len() == 12);
        assert!(upload.tangents.is_none());
        assert!(upload.joint_weights.is_none());
        assert!(upload.indices.iter().all(|&i| (i as usize) < 8));
    }

    #[test]
    fn node_transforms_bake_into_geometry() {
        let mesh = gltf_to_mesh(FIXTURE.as_bytes()).unwrap();
        let upload = convert_mesh(AssetId::invalid(), &mesh, 0);

        let max_y = upload.positions.iter().fold(0.0f32, |m, p| m.max(p[1]));
        assert!((max_y - 2.0).abs() < 1e-5, "translation not baked: {max_y}");
        let rotated = upload.normals[4];
        assert!(
            (rotated[0] - 1.0).abs() < 1e-5 && rotated[2].abs() < 1e-5,
            "rotation not applied to normals: {rotated:?}"
        );
        assert!(upload.positions[..4].iter().all(|p| p[1] <= 1.0));
        assert!((upload.normals[0][2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn joint_weight_validation_is_strict() {
        let valid_indices = [[0, 1, 0, 0], [1, 0, 0, 0]];
        let valid_weights = [[0.75, 0.25, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]];
        validate_joint_weights(&valid_indices, &valid_weights, 2).unwrap();

        let bad_sum = [[0.5, 0.25, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]];
        assert!(
            validate_joint_weights(&valid_indices, &bad_sum, 2)
                .unwrap_err()
                .to_string()
                .contains("weights sum")
        );
        let bad_index = [[0, 2, 0, 0], [1, 0, 0, 0]];
        assert!(
            validate_joint_weights(&bad_index, &valid_weights, 2)
                .unwrap_err()
                .to_string()
                .contains("references joint")
        );
        let unused_indices = [[0, 0, 0, 0], [0, 0, 0, 0]];
        assert!(
            validate_joint_weights(&unused_indices, &valid_weights, 2)
                .unwrap_err()
                .to_string()
                .contains("has no positive vertex weight")
        );
    }

    #[test]
    fn malformed_gltf_is_an_error() {
        assert!(gltf_to_mesh(b"not gltf").is_err());
        assert!(gltf_to_skinned_mesh(b"not gltf").is_err());
    }
}
