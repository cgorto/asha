//! Dependency guard: the normal renderer path must not pull a GPU backend.

use std::process::Command;

/// Mechanical enforcement that the normal renderer dependency tree stays free
#[test]
fn normal_dependency_tree_is_render_free() {
    let output = Command::new("cargo")
        .args(["tree", "-p", "render", "--edges", "normal"])
        .output()
        .expect("cargo must be available on PATH");

    assert!(
        output.status.success(),
        "cargo tree failed:\n{}",
        String::from_utf8_lossy(&output.stderr),
    );

    let tree = String::from_utf8(output.stdout).expect("cargo tree output must be UTF-8");

    for forbidden in [
        "wgpu",
        "wgpu-core",
        "wgpu-hal",
        "naga",
        "bevy_render",
        "bevy_ui_render",
        "bevy_sprite_render",
    ] {
        assert!(
            !tree_has_package(&tree, forbidden),
            "forbidden package {forbidden:?} appeared in render's normal dependency tree:\n{tree}",
        );
    }

    for required in ["taffy", "parley", "swash"] {
        assert!(
            tree_has_package(&tree, required),
            "required package {required:?} is missing from render's normal dependency tree:\n{tree}",
        );
    }
}

/// Matches a cargo-tree package line by its standalone package-name token.
fn tree_has_package(tree: &str, package: &str) -> bool {
    tree.lines().any(|line| {
        let mut words = line.split_whitespace();
        while let Some(word) = words.next() {
            if word == package {
                return matches!(words.next(), Some(version) if version.starts_with('v'));
            }
        }
        false
    })
}
