//! `#[gpu_data]` — the attribute stanza for GPU-shared structs, spelled once.

use proc_macro::TokenStream;

/// Marks a struct as shared CPU and GPU data.
///
/// Expands to `repr(C)`, `Copy`, `Clone`, and `Default`, plus host-only
/// `Debug`, `Zeroable`, and `Pod` derives. `Default` is intentional: GPU
/// zero-initialization and BSN templates rely on it, so every field must be
/// `Default + Pod`. Optional `component` and `resource` arguments add ECS
/// derives on hosts; input validation is delegated to the derives.
#[proc_macro_attribute]
pub fn gpu_data(attr: TokenStream, item: TokenStream) -> TokenStream {
    let base = r#"
        #[repr(C)]
        #[derive(Copy, Clone, Default)]
        #[cfg_attr(not(target_arch = "spirv"), derive(Debug, bytemuck::Zeroable, bytemuck::Pod))]
    "#;
    let ecs = match attr.to_string().as_str() {
        "" => "",
        "component" => {
            r#"#[cfg_attr(all(feature = "bevy", not(target_arch = "spirv")), derive(bevy_ecs::prelude::Component))]"#
        }
        "resource" => {
            r#"#[cfg_attr(all(feature = "bevy", not(target_arch = "spirv")), derive(bevy_ecs::prelude::Resource))]"#
        }
        other => panic!("#[gpu_data] takes `component`, `resource`, or nothing — got `{other}`"),
    };
    let mut out: TokenStream = format!("{base}{ecs}")
        .parse()
        .expect("static attribute tokens always parse");
    out.extend(item);
    out
}
