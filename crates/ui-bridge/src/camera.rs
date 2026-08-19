//! Refreshes computed window-camera data without `bevy_render`.
//!
//! This is the window-target portion of `bevy_render::camera::camera_system`.
//! Image and texture-view targets remain at their defaults because this tree
//! has no corresponding render path. Unlike upstream, a missing window skips
//! only that camera; `Projection` is optional. Change detection avoids
//! rewriting cameras and re-propagating the UI hierarchy every frame.

use bevy::camera::{Camera, NormalizedRenderTarget, Projection, RenderTarget, RenderTargetInfo};
use bevy::ecs::entity::EntityHashSet;
use bevy::math::UVec2;
use bevy::prelude::*;
use bevy::window::{PrimaryWindow, Window, WindowCreated, WindowResized, WindowScaleFactorChanged};

/// Refreshes computed camera data for window targets, following
/// `bevy_render::camera::camera_system`.
fn camera_system(
    mut window_resized: MessageReader<WindowResized>,
    mut window_created: MessageReader<WindowCreated>,
    mut window_scale_factor_changed: MessageReader<WindowScaleFactorChanged>,
    primary_window: Query<Entity, With<PrimaryWindow>>,
    windows: Query<(Entity, &Window)>,
    mut cameras: Query<(&mut Camera, &RenderTarget, Option<&mut Projection>)>,
) {
    let primary_window = primary_window.iter().next();

    // Any window lifecycle change invalidates camera data.
    let mut changed_window_ids = EntityHashSet::default();
    changed_window_ids.extend(window_created.read().map(|event| event.window));
    changed_window_ids.extend(window_resized.read().map(|event| event.window));
    let scale_factor_changed_window_ids: EntityHashSet = window_scale_factor_changed
        .read()
        .map(|event| event.window)
        .collect();
    changed_window_ids.extend(scale_factor_changed_window_ids.iter().copied());

    for (mut camera, render_target, mut projection) in &mut cameras {
        let mut viewport_size = camera.viewport.as_ref().map(|v| v.physical_size);

        if let Some(NormalizedRenderTarget::Window(window_ref)) =
            render_target.normalize(primary_window)
        {
            let window_entity = window_ref.entity();
            let projection_changed = projection.as_ref().is_some_and(|p| p.is_changed());
            if changed_window_ids.contains(&window_entity)
                || camera.is_added()
                || projection_changed
                || camera.computed.old_viewport_size != viewport_size
                || camera.computed.old_sub_camera_view != camera.sub_camera_view
            {
                // A missing window affects only this camera.
                if let Ok((_, window)) = windows.get(window_entity) {
                    let new_info = RenderTargetInfo {
                        physical_size: window.physical_size(),
                        scale_factor: window.scale_factor(),
                    };

                    // Preserve viewport proportions across scale changes.
                    if scale_factor_changed_window_ids.contains(&window_entity)
                        && let Some(old_scale_factor) = camera
                            .computed
                            .target_info
                            .as_ref()
                            .map(|info| info.scale_factor)
                    {
                        let resize_factor = new_info.scale_factor / old_scale_factor;
                        if let Some(viewport) = &mut camera.viewport {
                            let resize = |v: UVec2| (v.as_vec2() * resize_factor).as_uvec2();
                            viewport.physical_position = resize(viewport.physical_position);
                            viewport.physical_size = resize(viewport.physical_size);
                            viewport_size = Some(viewport.physical_size);
                        }
                    }
                    if let Some(viewport) = &mut camera.viewport {
                        viewport.clamp_to_size(new_info.physical_size);
                    }
                    camera.computed.target_info = Some(new_info);

                    if let Some(projection) = projection.as_deref_mut()
                        && let Some(size) = camera.logical_viewport_size()
                        && size.x != 0.0
                        && size.y != 0.0
                    {
                        projection.update(size.x, size.y);
                        camera.computed.clip_from_view = match &camera.sub_camera_view {
                            Some(sub_view) => projection.get_clip_from_view_for_sub(sub_view),
                            None => projection.get_clip_from_view(),
                        };
                    }
                }
            }
        }

        if camera.computed.old_viewport_size != viewport_size {
            camera.computed.old_viewport_size = viewport_size;
        }
        if camera.computed.old_sub_camera_view != camera.sub_camera_view {
            camera.computed.old_sub_camera_view = camera.sub_camera_view;
        }
    }
}

/// Adds window support and schedules camera refreshes.
///
/// Runs in `PostStartup` and `PostUpdate` under `CameraUpdateSystems`.
pub(crate) fn build(app: &mut App) {
    if !app.is_plugin_added::<bevy::window::WindowPlugin>() {
        app.add_plugins(bevy::window::WindowPlugin::default());
    }
    app.add_systems(
        PostStartup,
        camera_system.in_set(bevy::camera::CameraUpdateSystems),
    )
    .add_systems(
        PostUpdate,
        camera_system.in_set(bevy::camera::CameraUpdateSystems),
    );
}
