# Shader groups

Shader groups attach a registered material shader to selected mesh entities while
preserving the GPU-driven mesh path. A `MeshShader` component changes the entity's
batch identity; its clusters are rendered with the selected shader pair instead
of `mesh_vert`/`mesh_frag`.

A shader group is a **partition of the batch table, not a new render pass**. The
forward pass uses the same render pass, `Equal + Read` depth state, and
`MeshFrameData`. Only the bound shaders and indirect-argument window change
between group runs.

## Architecture

Extraction keys each batch by `(mesh, material, group)` and sorts by group:

- group key `0` is the standard shader pair;
- registered group `g` has key `g + 1`;
- instances retain query order; only `batch_index` is remapped;
- batches belonging to each group form one contiguous run.

Extraction publishes the runs as `ShaderGroupSlice`. `record_grouped` asserts
that the slices are contiguous, ordered, and cover the grouped batch table.
Cluster compaction, culling, indirect-command construction, and the depth
prepass remain group-blind. Under `ReplaceForward`, grouped geometry is included
in that prepass, allowing the forward draw to use `Equal` depth.

The forward pass records one `cmd_draw_instanced_indirect_multi` for each run.
Draw counts are CPU-authored candidate counts: empty batches receive zero-instance
commands, so per-group counts are written to the count ring without readback.
A frame with no custom groups records the standard single multi-draw.

Registration is index-authoritative. `ShaderGroups::register` assigns dense
indices on the main thread and returns the corresponding forgery-guarded
`MeshShader` component. Registration data is sent in the frame packet; the render
thread drains `FrameCtx::shader_group_uploads` into
`MeshForwardPass::register_group`. Late registration is supported, but shader
creation occurs when the render thread drains the upload.

No `MeshShader` component means the standard pair. This zero-cost default is also
restored when a custom group component is removed.

Relevant implementation points:

| Location | Responsibility |
| --- | --- |
| `crates/render/src/meshes.rs` | Registry, descriptors, component, batch key, sorting, and slices |
| `crates/render/src/lib.rs`, `crates/render/src/thread.rs` | Upload extraction and render-thread draining |
| `crates/mesh/src/forward.rs` | Group registration, grouped recording, indirect draws, and count ring |
| `shaders/lib/src/mesh/mod.rs` | Reference mesh shaders, including `mesh_flat_frag` |

## Shader contract

Group entry points are rust-gpu functions compiled to SPIR-V assets. A descriptor
names the resulting fragment and optional vertex shader.

For `ReplaceForward` fragments:

- Fragment inputs must match `mesh_vert` outputs in `mesh_frag` order, using
  positional locations. Declare all five: `normal_world`, `position_world`,
  `uv`, flat `material_index`, and flat `instance_color`.
- Write all four color outputs. The surface MRT is bound for the entire pass;
  an unwritten attachment is undefined. `out_surface_material = 0.0` excludes a
  pixel from deferred local lighting; `material_index + 1` includes it.
- `instance_color` is the per-entity parameter block. `MeshInstanceColor` is the
  host component, and `ShaderEffect::encode` is the typed host interface.
  A group defines the meaning of all four lanes; the standard shader uses them
  as a multiplicative tint.
- `MeshFrameData::time` is the host frame clock in seconds. Grouped recording
  supplies it; ungrouped recording supplies zero. Keep shaders pure functions of
  time, varyings, and parameters so CPU verification twins can match them.
- A custom vertex shader must reproduce `mesh_vert` positions exactly, including
  vertex pulling from compacted clusters through the `abi-mesh` code. Otherwise
  `Equal` depth rejects the forward fragments. `vert: None` retains the standard
  vertex stage.

`mesh_flat_frag` documents the reference fragment contract. Its animated example
is `hazard_pulse_frag`, verified against the CPU twin in
`crates/mesh/tests/grouped.rs`.

## Host usage

Register a shader during setup, attach the returned component, then drain uploads
and record the published slices on the render thread:

```rust
let flat = shader_groups.register(ShaderGroupDesc {
    vert: None,
    frag: "mesh_flat_frag".into(),
    mode: ShaderGroupMode::ReplaceForward,
});
commands.spawn((Mesh3d(mesh), material, flat, transform));

let mut forward =
    MeshForwardPass::with_groups(gpu, MAX_SHADER_GROUPS, FRAMES_IN_FLIGHT);
for upload in ctx.shader_group_uploads() {
    forward.register_group(gpu, upload.index, upload);
}
// Extract ShaderGroupSlice values and call record_grouped(..., ctx.time, ...).
```

## `ShaderEffect`

`ShaderEffect` in `crates/render/src/effects.rs` provides typed registration and
parameter encoding. Each effect type maps to one shader group and one per-instance
`vec4`; its implementation is equivalent to `MeshShader` plus
`MeshInstanceColor`.

```rust
#[derive(Component)]
struct HazardPulse { strain: f32, phase: f32 }

impl ShaderEffect for HazardPulse {
    const FRAG: &'static str = "hazard_pulse_frag";
    fn encode(&self) -> [f32; 4] { [self.strain, self.phase, 0.0, 0.0] }
}

app.add_shader_effect::<HazardPulse>();
commands.entity(entity).insert(HazardPulse { strain, phase });
commands.entity(entity).remove::<HazardPulse>();
```

Effects register when `add_shader_effect` is called, so indices follow plugin
build order. A sync system writes `MeshInstanceColor` when the effect changes;
the effect owns that parameter lane and overwrites manual color writes. Removing
the effect removes its seam components and restores standard forward rendering. One effect occupies each
shader slot; replacement and coat effects can coexist, but their encoders must
use a compatible four-lane convention.

## Additive coats

`ShaderGroupMode::Coat` adds a second draw after all forward runs. `MeshCoat`,
registered with `ShaderGroups::register_coat`, is independent of `MeshShader`,
so an entity may use either or both.

Coat batches are contiguous sub-runs within each forward group. Extraction
publishes them as `ShaderCoatSlice`; `record_grouped` checks that they are
ascending, disjoint, and in range. Coats record after the base forward runs,
using additive `One + One` blending with `Equal + Read` depth. All surface-MRT
writes are masked off, preventing coats from changing deferred-light data.

A coat fragment's `out_color` is a light contribution—the value to add—not a
replacement color. It must still declare the bound surface outputs. The
reference `glow_coat_frag` emits `instance_color.rgb * instance_color.w`.
Forward and coat fragments share one `instance_color` per entity, so entities
using both must define a lane split. `ShaderEffect::MODE` selects
`ReplaceForward` or `Coat`; `EffectGroup::coat()` supplies the coat slot.

The count ring reserves `1 + 2 * max_groups` counters per in-flight frame slot:
one standard counter, one counter for each forward slice, and one for each coat
slice.

## Hardware proof

`crates/mesh/tests/grouped.rs` exercises standard, flat-group, pulse-group, and
coat draws on hardware. It verifies that the standard mesh matches
`mesh_shade_slim`, the flat group matches its exact base color, and the pulse
group matches its CPU twin at non-zero frame time. The test also verifies surface
material identities, unchanged background pixels, and additive coat output while
surface material remains that of the base draw.
