use abi_mesh::{ALPHA_MODE_OPAQUE, MaterialEntry};

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_ne_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn read_f32(bytes: &[u8], offset: usize) -> f32 {
    f32::from_ne_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

#[test]
fn material_entry_default_is_zero() {
    let material = MaterialEntry::default();
    let bytes = bytemuck::bytes_of(&material);
    assert!(bytes.iter().all(|byte| *byte == 0));
}

#[test]
fn material_entry_standard_bytes_match_layout() {
    let material = MaterialEntry::standard();
    let bytes = bytemuck::bytes_of(&material);

    assert_eq!(read_u32(bytes, 0), 0);
    assert_eq!(read_u32(bytes, 4), 0);
    assert_eq!(read_u32(bytes, 8), 0);
    assert_eq!(read_u32(bytes, 12), 0);
    assert_eq!(read_u32(bytes, 16), 0);
    assert_eq!(read_u32(bytes, 20), 0);
    assert_eq!(read_u32(bytes, 24), 0);
    assert_eq!(read_u32(bytes, 28), 0);
    assert_eq!(read_u32(bytes, 32), ALPHA_MODE_OPAQUE);
    assert_eq!(read_f32(bytes, 36), 0.5);
    assert_eq!(read_u32(bytes, 40), 0);
    assert_eq!(read_u32(bytes, 44), 0);
    assert_eq!(read_f32(bytes, 48), 1.0);
    assert_eq!(read_f32(bytes, 52), 1.0);
    assert_eq!(read_f32(bytes, 56), 1.0);
    assert_eq!(read_f32(bytes, 60), 1.0);
    assert_eq!(read_f32(bytes, 64), 0.0);
    assert_eq!(read_f32(bytes, 68), 0.0);
    assert_eq!(read_f32(bytes, 72), 0.0);
    assert_eq!(read_f32(bytes, 76), 1.0);
    assert_eq!(read_f32(bytes, 80), 0.0);
    assert_eq!(read_f32(bytes, 84), 0.0);
    assert_eq!(read_f32(bytes, 88), 1.5);
    assert_eq!(read_u32(bytes, 92), 0);
    assert_eq!(read_u32(bytes, 96), 0);
    assert_eq!(read_u32(bytes, 100), 0);
    assert_eq!(read_f32(bytes, 104), 0.0);
    assert_eq!(read_f32(bytes, 108), 0.0);
    assert_eq!(bytes.len(), 112);
}
