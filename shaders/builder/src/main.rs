//! Compiles shaders to SPIR-V and copies entry-point binaries.

use spirv_builder::{Capability, ModuleResult, SpirvBuilder};
use std::{collections::BTreeSet, fs, path::Path};

fn asset_entry_name(entry: &str) -> &str {
    entry.rsplit("::").next().unwrap_or(entry)
}

fn write_asset_module(
    src: &Path,
    dst: &Path,
    entry: &str,
    asset_entry: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = fs::read(src)?;
    let bytes = rename_spirv_entry_point(bytes, entry, asset_entry)?;
    fs::write(dst, bytes)?;
    Ok(())
}

fn string_words(s: &str) -> Vec<u32> {
    let mut bytes = s.as_bytes().to_vec();
    bytes.push(0);
    while bytes.len() % 4 != 0 {
        bytes.push(0);
    }
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn spirv_words(bytes: &[u8]) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    if bytes.len() < 20 || bytes.len() % 4 != 0 {
        return Err("invalid SPIR-V module size".into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn spirv_bytes(words: Vec<u32>) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

fn literal_string_word_count(words: &[u32]) -> Option<usize> {
    for (i, word) in words.iter().enumerate() {
        if word.to_le_bytes().contains(&0) {
            return Some(i + 1);
        }
    }
    None
}

fn literal_string_matches(words: &[u32], expected: &str) -> bool {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    let Some(null) = bytes.iter().position(|b| *b == 0) else {
        return false;
    };
    &bytes[..null] == expected.as_bytes()
}

fn rename_spirv_entry_point(
    bytes: Vec<u8>,
    from: &str,
    to: &str,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    if from == to {
        return Ok(bytes);
    }

    let mut words = spirv_words(&bytes)?;
    let replacement = string_words(to);

    let mut offset = 5;
    let mut renamed = false;
    while offset < words.len() {
        let word = words[offset];
        let word_count = (word >> 16) as usize;
        let opcode = word & 0xffff;
        if word_count == 0 {
            return Err("invalid SPIR-V instruction word count".into());
        }
        let end = offset + word_count;
        if end > words.len() {
            return Err("SPIR-V instruction exceeds module size".into());
        }

        // OpEntryPoint is execution model, entry-point ID, a literal name,
        // then interface IDs. Replacing the name can change its word count,
        // so splice the literal, rewrite the instruction count, and resume
        // at the rewritten instruction; otherwise old name padding becomes
        // bogus interface IDs and can make pipeline creation fail or crash.
        if opcode == 15 && word_count >= 4 {
            let name_start = offset + 3;
            let Some(name_words) = literal_string_word_count(&words[name_start..end]) else {
                return Err("OpEntryPoint has no null-terminated name".into());
            };
            let name_end = name_start + name_words;
            if literal_string_matches(&words[name_start..name_end], from) {
                words.splice(name_start..name_end, replacement.iter().copied());
                let new_word_count = word_count - name_words + replacement.len();
                words[offset] = ((new_word_count as u32) << 16) | opcode;
                renamed = true;
                offset += new_word_count;
                continue;
            }
        }

        offset = end;
    }

    if !renamed {
        return Err(format!("SPIR-V entry point {from:?} not found").into());
    }
    Ok(spirv_bytes(words))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = SpirvBuilder::new(
        concat!(env!("CARGO_MANIFEST_DIR"), "/../lib"),
        "spirv-unknown-vulkan1.3",
    )
    .capability(Capability::PhysicalStorageBufferAddresses)
    .capability(Capability::Int64)
    .capability(Capability::RuntimeDescriptorArray)
    // Device-scope atomics require the Vulkan memory-model feature.
    .capability(Capability::VulkanMemoryModelDeviceScope)
    // `draw_index` requires shader draw parameters.
    .capability(Capability::DrawParameters)
    // Fragment `primitive_id` requires the geometry feature.
    .capability(Capability::Geometry)
    .extension("SPV_KHR_physical_storage_buffer");
    builder.multimodule = true;

    let result = builder.build()?;

    let out_dir = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/shaders"));
    fs::create_dir_all(out_dir)?;
    match result.module {
        ModuleResult::MultiModule(entries) => {
            let mut emitted = BTreeSet::new();
            for (entry, path) in entries {
                let asset_entry = asset_entry_name(&entry);
                if !emitted.insert(asset_entry.to_owned()) {
                    return Err(format!("duplicate flat shader entry name {asset_entry:?}").into());
                }
                let dst = out_dir.join(format!("{asset_entry}.spv"));
                write_asset_module(&path, &dst, &entry, asset_entry)?;
                println!("{entry} -> {}", dst.display());
            }
        }
        ModuleResult::SingleModule(path) => {
            let dst = out_dir.join("shaders.spv");
            std::fs::copy(&path, &dst)?;
            println!("shaders -> {}", dst.display());
        }
    }
    Ok(())
}
