//! Primitive meshes and additive binned particle rasterization.

use abi_particles::{ParticleDrawFragData, ParticleDrawVertData};
use gpu::pass::{FrameAlloc, Pass};
use gpu::{
    BlendFactor, BlendOp, BlendState, CommandBuffer, CompareOp, DepthFlags, DepthState, Gpu,
    HazardFlags, LoadOp, Memory, RenderAttachment, RenderPassDesc, ShaderTypeGraphics, Stage,
    StoreOp, Texture,
};
use mesh::primitives::icosphere;

use crate::sim::{ParticleSimPass, upload_slice};
use crate::spec::{PRIMITIVE_COUNT, ParticleView};

#[derive(Clone, Copy)]
struct PrimitiveMesh {
    positions: gpu::Ptr<[f32; 4]>,
    normals: gpu::Ptr<[f32; 4]>,
    uvs: gpu::Ptr<[f32; 2]>,
    indices: gpu::Ptr<u32>,
}

#[derive(Clone, Default)]
struct CpuMesh {
    positions: Vec<[f32; 4]>,
    normals: Vec<[f32; 4]>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<u32>,
}

impl CpuMesh {
    fn push(&mut self, position: [f32; 3], normal: [f32; 3], uv: [f32; 2]) {
        self.positions
            .push([position[0], position[1], position[2], 0.0]);
        self.normals.push([normal[0], normal[1], normal[2], 0.0]);
        self.uvs.push(uv);
    }
}

fn quad_mesh() -> CpuMesh {
    CpuMesh {
        positions: vec![
            [-0.5, -0.5, 0.0, 0.0],
            [-0.5, 0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
            [0.5, -0.5, 0.0, 0.0],
        ],
        normals: vec![[0.0, 0.0, 1.0, 0.0]; 4],
        uvs: vec![[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
        indices: vec![0, 1, 2, 0, 2, 3],
    }
}

fn disc_mesh() -> CpuMesh {
    const SEGMENTS: u32 = 12;
    let mut mesh = CpuMesh::default();
    mesh.push([0.0; 3], [0.0, 0.0, 1.0], [0.5, 0.5]);
    for segment in 0..SEGMENTS {
        let angle = segment as f32 * core::f32::consts::TAU / SEGMENTS as f32;
        mesh.push(
            [angle.cos() * 0.5, angle.sin() * 0.5, 0.0],
            [0.0, 0.0, 1.0],
            [(angle.cos() + 1.0) * 0.5, (angle.sin() + 1.0) * 0.5],
        );
    }
    for segment in 0..SEGMENTS {
        mesh.indices
            .extend_from_slice(&[0, segment + 1, (segment + 1) % SEGMENTS + 1]);
    }
    mesh
}

fn cube_mesh() -> CpuMesh {
    const FACES: [([f32; 3], [[f32; 3]; 4]); 6] = [
        (
            [-1.0, 0.0, 0.0],
            [
                [-0.5, -0.5, 0.5],
                [-0.5, -0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, 0.5, 0.5],
            ],
        ),
        (
            [1.0, 0.0, 0.0],
            [
                [0.5, -0.5, -0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, -0.5],
            ],
        ),
        (
            [0.0, -1.0, 0.0],
            [
                [-0.5, -0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.5, -0.5, -0.5],
            ],
        ),
        (
            [0.0, 1.0, 0.0],
            [
                [-0.5, 0.5, 0.5],
                [-0.5, 0.5, -0.5],
                [0.5, 0.5, -0.5],
                [0.5, 0.5, 0.5],
            ],
        ),
        (
            [0.0, 0.0, -1.0],
            [
                [0.5, -0.5, -0.5],
                [-0.5, -0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [0.5, 0.5, -0.5],
            ],
        ),
        (
            [0.0, 0.0, 1.0],
            [
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5],
            ],
        ),
    ];
    let mut mesh = CpuMesh::default();
    for (normal, corners) in FACES {
        let base = mesh.positions.len() as u32;
        for (corner, position) in corners.into_iter().enumerate() {
            mesh.push(
                position,
                normal,
                match corner {
                    0 => [0.0, 0.0],
                    1 => [1.0, 0.0],
                    2 => [1.0, 1.0],
                    _ => [0.0, 1.0],
                },
            );
        }
        mesh.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    mesh
}

fn icosphere_mesh() -> CpuMesh {
    let source = icosphere(0.5, 1);
    CpuMesh {
        positions: source
            .positions
            .iter()
            .map(|p| [p[0], p[1], p[2], 0.0])
            .collect(),
        normals: source
            .normals
            .iter()
            .map(|n| [n[0], n[1], n[2], 0.0])
            .collect(),
        uvs: source.uvs,
        indices: source.indices,
    }
}

/// Low-poly cone with its broad end at +Y.
fn cone_mesh() -> CpuMesh {
    const SEGMENTS: u32 = 8;
    let mut mesh = CpuMesh::default();
    let slant = (1.25_f32).sqrt();
    for segment in 0..SEGMENTS {
        let angle = segment as f32 * core::f32::consts::TAU / SEGMENTS as f32;
        mesh.push(
            [0.5 * angle.cos(), 0.5, 0.5 * angle.sin()],
            [angle.cos() / slant, -0.5 / slant, angle.sin() / slant],
            [segment as f32 / SEGMENTS as f32, 0.0],
        );
    }
    let apex = mesh.positions.len() as u32;
    mesh.push([0.0, -0.5, 0.0], [0.0, -1.0, 0.0], [0.5, 1.0]);
    for segment in 0..SEGMENTS {
        mesh.indices
            .extend_from_slice(&[(segment + 1) % SEGMENTS, segment, apex]);
    }
    let center = mesh.positions.len() as u32;
    mesh.push([0.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5]);
    let ring = mesh.positions.len() as u32;
    for segment in 0..SEGMENTS {
        let angle = segment as f32 * core::f32::consts::TAU / SEGMENTS as f32;
        mesh.push(
            [0.5 * angle.cos(), 0.5, 0.5 * angle.sin()],
            [0.0, 1.0, 0.0],
            [(angle.cos() + 1.0) * 0.5, (angle.sin() + 1.0) * 0.5],
        );
    }
    for segment in 0..SEGMENTS {
        mesh.indices
            .extend_from_slice(&[center, ring + segment, ring + (segment + 1) % SEGMENTS]);
    }
    mesh
}

/// Wedge primitive with per-face normals.
fn prism_mesh() -> CpuMesh {
    let mut mesh = CpuMesh::default();
    let tri = |mesh: &mut CpuMesh, normal, vertices: [[f32; 3]; 3]| {
        let base = mesh.positions.len() as u32;
        for (index, point) in vertices.into_iter().enumerate() {
            mesh.push(point, normal, [[0.5, 1.0], [0.0, 0.0], [1.0, 0.0]][index]);
        }
        mesh.indices.extend_from_slice(&[base, base + 1, base + 2]);
    };
    tri(
        &mut mesh,
        [0.0, 0.0, 1.0],
        [[0.0, 0.5, 0.5], [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5]],
    );
    tri(
        &mut mesh,
        [0.0, 0.0, -1.0],
        [[0.0, 0.5, -0.5], [0.5, -0.5, -0.5], [-0.5, -0.5, -0.5]],
    );
    let quad = |mesh: &mut CpuMesh, normal, points: [[f32; 3]; 4], indices: [u32; 6]| {
        let base = mesh.positions.len() as u32;
        for (index, point) in points.into_iter().enumerate() {
            mesh.push(
                point,
                normal,
                [[0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [1.0, 0.0]][index],
            );
        }
        mesh.indices
            .extend(indices.into_iter().map(|index| base + index));
    };
    quad(
        &mut mesh,
        [0.894427, 0.447214, 0.0],
        [
            [0.0, 0.5, 0.5],
            [0.0, 0.5, -0.5],
            [0.5, -0.5, 0.5],
            [0.5, -0.5, -0.5],
        ],
        [0, 2, 1, 2, 3, 1],
    );
    quad(
        &mut mesh,
        [-0.894427, 0.447214, 0.0],
        [
            [0.0, 0.5, -0.5],
            [0.0, 0.5, 0.5],
            [-0.5, -0.5, -0.5],
            [-0.5, -0.5, 0.5],
        ],
        [0, 2, 1, 2, 3, 1],
    );
    quad(
        &mut mesh,
        [0.0, -1.0, 0.0],
        [
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, -0.5, -0.5],
            [-0.5, -0.5, -0.5],
        ],
        [0, 2, 1, 0, 3, 2],
    );
    mesh
}

/// Primitive storage and additive binned raster pass.
pub struct ParticleDrawPass {
    vert_shader: gpu::Shader,
    frag_shader: gpu::Shader,
    meshes: [PrimitiveMesh; PRIMITIVE_COUNT as usize],
}

impl ParticleDrawPass {
    pub fn new(gpu: &Gpu) -> Self {
        Self {
            vert_shader: gpu.shader_create(
                &asha_assets::load_spv("particle_vert"),
                ShaderTypeGraphics::Vertex,
                "particle_vert",
            ),
            frag_shader: gpu.shader_create(
                &asha_assets::load_spv("particle_frag"),
                ShaderTypeGraphics::Fragment,
                "particle_frag",
            ),
            meshes: Self::upload_meshes(
                gpu,
                [
                    quad_mesh(),
                    disc_mesh(),
                    cube_mesh(),
                    icosphere_mesh(),
                    cone_mesh(),
                    prism_mesh(),
                ],
            ),
        }
    }

    fn upload_meshes(
        gpu: &Gpu,
        cpu_meshes: [CpuMesh; PRIMITIVE_COUNT as usize],
    ) -> [PrimitiveMesh; PRIMITIVE_COUNT as usize] {
        let mut meshes = Vec::with_capacity(PRIMITIVE_COUNT as usize);
        let cb = gpu.commands_begin(gpu::Queue::Main);
        let mut staging = Vec::with_capacity((PRIMITIVE_COUNT * 4) as usize);
        for cpu in cpu_meshes {
            assert!(
                !cpu.positions.is_empty()
                    && cpu.positions.len() == cpu.normals.len()
                    && cpu.positions.len() == cpu.uvs.len()
            );
            assert!(
                cpu.indices
                    .iter()
                    .all(|&index| index < cpu.positions.len() as u32)
            );
            let positions = gpu.alloc_slice::<[f32; 4]>(cpu.positions.len() as u64, Memory::Gpu);
            let normals = gpu.alloc_slice::<[f32; 4]>(cpu.normals.len() as u64, Memory::Gpu);
            let uvs = gpu.alloc_slice::<[f32; 2]>(cpu.uvs.len() as u64, Memory::Gpu);
            let indices = gpu.alloc_slice::<u32>(cpu.indices.len() as u64, Memory::Gpu);
            upload_slice(gpu, cb, positions, &cpu.positions, &mut staging);
            upload_slice(gpu, cb, normals, &cpu.normals, &mut staging);
            upload_slice(gpu, cb, uvs, &cpu.uvs, &mut staging);
            upload_slice(gpu, cb, indices, &cpu.indices, &mut staging);
            meshes.push(PrimitiveMesh {
                positions,
                normals,
                uvs,
                indices,
            });
        }
        gpu.cmd_barrier(cb, Stage::Transfer, Stage::All, HazardFlags::empty());
        gpu.queue_submit(gpu::Queue::Main, &[cb]);
        gpu.queue_wait_idle(gpu::Queue::Main);
        for src in staging {
            gpu.free(src);
        }
        match meshes.try_into() {
            Ok(meshes) => meshes,
            Err(_) => panic!("fixed primitive mesh count"),
        }
    }

    pub fn record(
        &self,
        gpu: &Gpu,
        cb: CommandBuffer,
        fa: &mut impl FrameAlloc,
        sim: &ParticleSimPass,
        target: Texture,
        depth: Texture,
        view: ParticleView,
    ) {
        gpu.cmd_begin_render_pass(
            cb,
            RenderPassDesc {
                color_attachments: &[RenderAttachment {
                    texture: target,
                    load_op: LoadOp::Load,
                    store_op: StoreOp::Store,
                    ..Default::default()
                }],
                depth_attachment: Some(RenderAttachment {
                    texture: depth,
                    load_op: LoadOp::Load,
                    store_op: StoreOp::Store,
                    ..Default::default()
                }),
                render_area_size: view.screen_size.to_array(),
                ..Default::default()
            },
        );
        gpu.cmd_set_shaders(cb, self.vert_shader, self.frag_shader);
        gpu.cmd_set_depth_state(
            cb,
            DepthState {
                mode: DepthFlags::READ,
                compare: CompareOp::Greater,
                ..Default::default()
            },
        );
        gpu.cmd_set_cull_mode(cb, false);
        gpu.cmd_set_blend_state(
            cb,
            BlendState {
                enable: true,
                color_op: BlendOp::Add,
                src_color_factor: BlendFactor::SrcAlpha,
                dst_color_factor: BlendFactor::One,
                alpha_op: BlendOp::Add,
                src_alpha_factor: BlendFactor::SrcAlpha,
                dst_alpha_factor: BlendFactor::One,
                color_write_mask: 0x0f,
            },
        );
        let frag = fa.frame_alloc(ParticleDrawFragData {
            materials: sim.materials.gpu,
        });
        for primitive in 0..PRIMITIVE_COUNT {
            let mesh = self.meshes[primitive as usize];
            let vert = fa.frame_alloc(ParticleDrawVertData {
                particles: sim.particles.gpu,
                visible: sim.visible_ptr(primitive),
                emitters: sim.emitters.gpu,
                materials: sim.materials.gpu,
                positions: mesh.positions.gpu,
                normals: mesh.normals.gpu,
                uvs: mesh.uvs.gpu,
                view_proj: view.view_proj.to_cols_array_2d(),
                camera_right: view.camera_right.to_array(),
                _pad0: 0,
                camera_up: view.camera_up.to_array(),
                _pad1: 0,
                camera_forward: view.camera_forward.to_array(),
                _pad_forward: 0,
                primitive,
                _pad2: [0; 3],
            });
            gpu.cmd_draw_indexed_instanced_indirect(
                cb,
                vert.cast(),
                frag.cast(),
                mesh.indices.cast(),
                sim.draw_args_ptr(gpu, primitive),
            );
        }
        gpu.cmd_end_render_pass(cb);
    }
}

impl Pass for ParticleDrawPass {
    const NAME: &'static str = "particles_draw";
    fn free(self, gpu: &Gpu) {
        gpu.shader_destroy(self.vert_shader);
        gpu.shader_destroy(self.frag_shader);
        for mesh in self.meshes {
            gpu.free(mesh.positions);
            gpu.free(mesh.normals);
            gpu.free(mesh.uvs);
            gpu.free(mesh.indices);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primitive_mesh_counts_and_indices_are_pinned() {
        let meshes = [
            quad_mesh(),
            disc_mesh(),
            cube_mesh(),
            icosphere_mesh(),
            cone_mesh(),
            prism_mesh(),
        ];
        let expected = [(4, 6), (13, 36), (24, 36), (42, 240), (18, 48), (18, 24)];
        for (mesh, expected) in meshes.iter().zip(expected) {
            assert_eq!((mesh.positions.len(), mesh.indices.len()), expected);
            assert!(
                mesh.indices
                    .iter()
                    .all(|&index| index < mesh.positions.len() as u32)
            );
        }
    }
}
