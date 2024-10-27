// conversion.rs

// Trait for processing audio samples
pub trait AudioSample {
    fn to_f32(&self) -> f32; // Convert to f32
}

// Implementations for different types
impl AudioSample for f32 {
    fn to_f32(&self) -> f32 { *self }
}

impl AudioSample for i32 {
    fn to_f32(&self) -> f32 {
        if *self == i32::MIN {
            -1.0
        } else {
            *self as f32 / i32::MAX as f32
        }
    }
}

impl AudioSample for i16 {
    fn to_f32(&self) -> f32 {
        (*self as f32 / i16::MAX as f32).clamp(-1.0, 1.0)
    }
}

impl AudioSample for u16 {
    fn to_f32(&self) -> f32 {
        (*self as f32 / u16::MAX as f32).clamp(0.0, 1.0)
    }
}

impl AudioSample for f64 {
    fn to_f32(&self) -> f32 { *self as f32 }
}

// Conversion function for I32 buffer to F32
pub fn convert_i32_buffer_to_f32(buffer: &[i32], channels: usize) -> Vec<f32> {
    buffer.iter().map(|&sample| sample.to_f32()).collect()
}

pub fn f32_to_i16(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|&sample| {
            let scaled = (sample * i16::MAX as f32).round();
            scaled.clamp(i16::MIN as f32, i16::MAX as f32) as i16
        })
        .collect()
}
