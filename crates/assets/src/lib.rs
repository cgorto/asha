//! Host-side runtime asset access.

/// Return a path below the runtime assets directory.
pub fn asset_path(rel: &str) -> std::path::PathBuf {
    let root = std::env::var("ASHA_ASSETS_DIR")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets").to_string());
    std::path::PathBuf::from(root).join(rel)
}

/// Load a compiled shader as SPIR-V words.
pub fn load_spv(name: &str) -> Vec<u32> {
    let path = asset_path(&format!("shaders/{name}.spv"));
    if !path.is_file() {
        panic!(
            "missing {} — run: cd shaders && cargo run -p builder --release",
            path.display()
        );
    }
    gpu::load_spv(path)
}
