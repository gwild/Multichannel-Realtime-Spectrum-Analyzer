pub fn convert_i32_buffer_to_f32(input_buffer: &[i32], _num_channels: usize) -> Vec<f32> {
    input_buffer
        .iter()
        .map(|&sample| {
            if sample == i32::MIN {
                -1.0 // Handle edge case
            } else {
                sample as f32 / i32::MAX as f32 // Normalize to f32 range (-1.0 to 1.0)
            }
        })
        .collect()
}

pub fn f32_to_i16(samples: &[f32]) -> Vec<i16> {
    samples.iter().map(|&sample| (sample * i16::MAX as f32) as i16).collect()
}

pub fn i16_to_f32(samples: &[i16]) -> Vec<f32> {
    samples.iter().map(|&sample| sample as f32 / i16::MAX as f32).collect()
}

// Add more conversion functions as needed
