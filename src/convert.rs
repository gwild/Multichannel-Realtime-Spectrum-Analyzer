// convert.rs

pub fn i32_to_f32(input: i32) -> f32 {
    input as f32 / i32::MAX as f32 // Normalize i32 to f32 range (-1.0 to 1.0)
}
