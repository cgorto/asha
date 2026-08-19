//! Typed shader effects bridge ECS components to registered mesh groups.
//!
//! Registration assigns one group in app-build order. Synchronization stamps
//! the mode-specific group component and rewrites the shared instance-color
//! parameter lane whenever the effect changes; removing the effect removes its
//! slot and returns the entity to standard shading when no companion remains.

use std::marker::PhantomData;

use bevy::prelude::*;

use crate::meshes::{
    MeshCoat, MeshInstanceColor, MeshShader, ShaderGroupDesc, ShaderGroupMode, ShaderGroups,
};

/// A typed shader group and per-instance parameter encoder.
///
/// `FRAG` names the fragment entry. `VERT` may replace the standard vertex
/// entry, but replacement-forward vertices must preserve positions for the
/// depth prepass. `MODE` selects replacement or additive coat behavior.
pub trait ShaderEffect: Component {
    /// `.spv` entry name of the group's fragment shader.
    const FRAG: &'static str;
    /// Vertex entry; `None` uses the standard position-preserving vertex path.
    const VERT: Option<&'static str> = None;
    /// Selects replacement forward shading or an additive coat pass.
    const MODE: ShaderGroupMode = ShaderGroupMode::ReplaceForward;
    /// Encodes all four lanes of the shared instance parameter vector.
    fn encode(&self) -> [f32; 4];
}

/// Which seam component effect `T` stamps.
#[derive(Clone, Copy)]
enum EffectBinding {
    Shader(MeshShader),
    Coat(MeshCoat),
}

/// Registered group handle for effect `T`.
#[derive(Resource)]
pub struct EffectGroup<T: ShaderEffect> {
    binding: EffectBinding,
    _marker: PhantomData<fn() -> T>,
}

impl<T: ShaderEffect> EffectGroup<T> {
    /// Returns the forward group; panics for a coat effect.
    pub fn shader(&self) -> MeshShader {
        match self.binding {
            EffectBinding::Shader(shader) => shader,
            EffectBinding::Coat(_) => panic!(
                "{} is a Coat effect; use EffectGroup::coat",
                std::any::type_name::<T>()
            ),
        }
    }

    /// Returns the coat group; panics for a replacement effect.
    pub fn coat(&self) -> MeshCoat {
        match self.binding {
            EffectBinding::Coat(coat) => coat,
            EffectBinding::Shader(_) => panic!(
                "{} is a ReplaceForward effect; use EffectGroup::shader",
                std::any::type_name::<T>()
            ),
        }
    }
}

pub trait ShaderEffectAppExt {
    /// Registers `T` and synchronizes its seam components after updates.
    fn add_shader_effect<T: ShaderEffect>(&mut self) -> &mut Self;
}

impl ShaderEffectAppExt for App {
    fn add_shader_effect<T: ShaderEffect>(&mut self) -> &mut Self {
        assert!(
            !self.world().contains_resource::<EffectGroup<T>>(),
            "shader effect {} registered twice",
            std::any::type_name::<T>(),
        );
        self.init_resource::<ShaderGroups>();
        let desc = ShaderGroupDesc {
            vert: T::VERT.map(String::from),
            frag: T::FRAG.to_string(),
            mode: T::MODE,
        };
        let mut groups = self.world_mut().resource_mut::<ShaderGroups>();
        let binding = match T::MODE {
            ShaderGroupMode::ReplaceForward => EffectBinding::Shader(groups.register(desc)),
            ShaderGroupMode::Coat => EffectBinding::Coat(groups.register_coat(desc)),
        };
        self.insert_resource(EffectGroup::<T> {
            binding,
            _marker: PhantomData,
        });
        self.add_systems(PostUpdate, sync_effect::<T>)
    }
}

/// Synchronizes group and encoded-parameter components.
fn sync_effect<T: ShaderEffect>(
    mut commands: Commands,
    group: Res<EffectGroup<T>>,
    mut changed: Query<
        (
            Entity,
            &T,
            Option<&mut MeshInstanceColor>,
            Has<MeshShader>,
            Has<MeshCoat>,
        ),
        Changed<T>,
    >,
    slots: Query<(Has<MeshShader>, Has<MeshCoat>)>,
    mut removed: RemovedComponents<T>,
) {
    for (entity, effect, color, has_shader, has_coat) in &mut changed {
        let encoded = MeshInstanceColor(effect.encode());
        match color {
            Some(mut color) => *color = encoded,
            None => {
                commands.entity(entity).insert(encoded);
            }
        }
        match group.binding {
            EffectBinding::Shader(shader) if !has_shader => {
                commands.entity(entity).insert(shader);
            }
            EffectBinding::Coat(coat) if !has_coat => {
                commands.entity(entity).insert(coat);
            }
            _ => {}
        }
    }
    for entity in removed.read() {
        let other_occupied = match group.binding {
            EffectBinding::Shader(_) => slots.get(entity).is_ok_and(|(_, coat)| coat),
            EffectBinding::Coat(_) => slots.get(entity).is_ok_and(|(shader, _)| shader),
        };
        if let Ok(mut entity) = commands.get_entity(entity) {
            match group.binding {
                EffectBinding::Shader(_) => entity.remove::<MeshShader>(),
                EffectBinding::Coat(_) => entity.remove::<MeshCoat>(),
            };
            if !other_occupied {
                entity.remove::<MeshInstanceColor>();
            }
        }
    }
}
