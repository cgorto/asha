//! Main-thread registry for embedded bindless icons.
//!
//! The paths match assets registered by `widgets` with `embedded_asset!`.
//! Unlike upstream `bevy_feathers`, this fork's embedded namespace is
//! `widgets`; only the three static icons are supported here. Logical slots
//! are deterministic and upload payloads cross once; the render thread
//! creates textures and patches heap indices.

use std::collections::HashMap;

use bevy::asset::{AssetApp, AssetId, AssetServer, Assets};
use bevy::image::Image;
use bevy::prelude::*;

/// Embedded icon paths in deterministic logical-slot order.
///
/// These are the `widgets` crate's `embedded_asset!` paths, not the upstream
/// `embedded://bevy_feathers/...` namespace. Slot zero means untextured;
/// paths use one-based slots.
pub const ICON_PATHS: [&str; 3] = [
    "embedded://widgets/assets/icons/chevron-down.png",
    "embedded://widgets/assets/icons/chevron-right.png",
    "embedded://widgets/assets/icons/x.png",
];

/// Decoded icon pixels queued for one render-thread upload.
#[derive(Clone)]
pub struct IconUploadPayload {
    /// One-based logical slot; zero means untextured.
    pub logical_slot: u32,
    pub width: u32,
    pub height: u32,
    /// Decoded RGBA8 sRGB bytes; upload linearizes color channels.
    pub pixels: Vec<u8>,
}

impl IconUploadPayload {
    /// Returns `None` for missing or invalid RGBA8 data.
    fn from_image(logical_slot: u32, image: &Image) -> Option<Self> {
        let width = image.width();
        let height = image.height();
        let pixels = image.data.clone()?;
        if width == 0
            || height == 0
            || pixels.len() as u64 != u64::from(width) * u64::from(height) * 4
        {
            bevy::log::warn!(
                "ui-bridge: icon logical slot {logical_slot} has an unexpected pixel buffer \
                 ({width}x{height}, {} bytes) — expected RGBA8; skipping upload",
                pixels.len()
            );
            return None;
        }
        Some(Self {
            logical_slot,
            width,
            height,
            pixels,
        })
    }
}

/// Main-thread map from image assets to logical icon slots.
#[derive(Resource)]
pub struct IconRegistry {
    /// Keeps registered image handles alive.
    handles: Vec<Handle<Image>>,
    slots: HashMap<AssetId<Image>, u32>,
    /// Prevents re-queueing each logical slot.
    queued: [bool; ICON_PATHS.len()],
}

impl IconRegistry {
    /// Returns a logical slot for a registered image asset.
    pub fn logical_slot(&self, id: AssetId<Image>) -> Option<u32> {
        self.slots.get(&id).copied()
    }

    /// Builds a registry from fixture slots without asset loading.
    pub fn from_slots(slots: impl IntoIterator<Item = (AssetId<Image>, u32)>) -> Self {
        Self {
            handles: Vec::new(),
            slots: slots.into_iter().collect(),
            queued: [false; ICON_PATHS.len()],
        }
    }
}

/// Pending icon uploads awaiting render-thread extraction.
#[derive(Resource, Default)]
pub struct IconUploadQueue {
    pending: Vec<IconUploadPayload>,
}

impl IconUploadQueue {
    pub fn pending(&self) -> &[IconUploadPayload] {
        &self.pending
    }
}

/// Registers icon handles and their deterministic logical slots.
fn setup_icon_registry(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    existing: Option<Res<IconRegistry>>,
) {
    if existing.is_some() {
        return;
    }

    debug_assert!(
        ICON_PATHS.windows(2).all(|w| w[0] < w[1]),
        "ICON_PATHS must stay sorted — array order IS the logical slot assignment"
    );

    let mut handles = Vec::with_capacity(ICON_PATHS.len());
    let mut slots = HashMap::with_capacity(ICON_PATHS.len());
    for (i, path) in ICON_PATHS.iter().enumerate() {
        let handle: Handle<Image> = asset_server.load(*path);
        slots.insert(handle.id(), i as u32 + 1);
        handles.push(handle);
    }
    commands.insert_resource(IconRegistry {
        handles,
        slots,
        queued: [false; ICON_PATHS.len()],
    });
}

/// Refreshes the one-frame upload payload queue. `Last` extracts the current
/// pending slice after `PostUpdate`, so clearing here does not remove the
/// payload until its single extraction window has passed.
fn queue_icon_uploads(
    mut registry: ResMut<IconRegistry>,
    mut queue: ResMut<IconUploadQueue>,
    images: Res<Assets<Image>>,
) {
    queue.pending.clear();
    for i in 0..registry.handles.len() {
        if registry.queued[i] {
            continue;
        }
        let Some(image) = images.get(&registry.handles[i]) else {
            continue;
        };
        let Some(payload) = IconUploadPayload::from_image(i as u32 + 1, image) else {
            continue;
        };
        queue.pending.push(payload);
        registry.queued[i] = true;
    }
}

/// Installs icon assets, registry systems, and an RGBA8 PNG loader.
///
/// The loader is registered explicitly because no render plugin installs it.
pub(crate) fn build(app: &mut App) {
    if !app.is_plugin_added::<bevy::asset::AssetPlugin>() {
        app.add_plugins(bevy::asset::AssetPlugin::default());
    }
    if !app.is_plugin_added::<bevy::image::ImagePlugin>() {
        app.add_plugins(bevy::image::ImagePlugin::default());
    }
    app.register_asset_loader(bevy::image::ImageLoader::new(
        bevy::image::CompressedImageFormats::NONE,
    ));
    app.init_resource::<IconUploadQueue>()
        .add_systems(Startup, setup_icon_registry)
        .add_systems(PostUpdate, queue_icon_uploads);
}
