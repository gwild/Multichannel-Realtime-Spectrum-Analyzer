pub fn convert_i32_buffer_to_f32(input_buffer: &[i32]) -> Vec<f32> {
    input_buffer
        .iter()
        .map(|&sample| {
            if sample == i32::MIN {
                -1.0 // Handle edge case
            } else {
                sample as f32 / i32::MAX as f32 // Normalize to f32 range (-1.0 to 1.0)
            }
        })
        .collect() // Convert to Vec<f32>
}
