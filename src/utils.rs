use log::info;

pub const MIN_FREQ: f64 = 20.0;  // Lowest frequency we want to analyze
pub const MAX_FREQ: f64 = 500.0;  // Changed from 2000.0 to 500.0
pub const MIN_BUFFER_SIZE: usize = 512;  // Increased from previous value
pub const MAX_BUFFER_SIZE: usize = 16384;  // Absolute maximum buffer size
pub const DEFAULT_BUFFER_SIZE: usize = 4096;  // Add this constant
pub const DEFAULT_FRAMES_PER_BUFFER: u32 = 2048;  // Add this constant
pub const FRAME_SIZES: [u32; 7] = [64, 128, 256, 512, 1024, 2048, 4096];

pub fn calculate_optimal_buffer_size(sample_rate: f64) -> usize {
    // Calculate minimum size needed for frequency resolution
    let min_samples = (sample_rate / MIN_FREQ) as usize;
    
    // Calculate size needed for highest frequency
    let max_samples = (sample_rate / MAX_FREQ * 4.0) as usize; // 4x oversampling
    
    // Start with a compromise between min and max
    let initial_size = ((min_samples + max_samples) / 2)
        .next_power_of_two()  // Round to next power of 2 for FFT efficiency
        .clamp(MIN_BUFFER_SIZE, MAX_BUFFER_SIZE);
    
    info!(
        "Calculated buffer size - Min: {}, Max: {}, Selected: {}, SR: {}",
        min_samples, max_samples, initial_size, sample_rate
    );
    
    initial_size
}

/// Converts raw FFT dB values (-100 to 100) to display/plot range (-100 to 0)
pub fn map_db_range(raw_db: f32) -> f32 {
    if raw_db > -100.0 {
        ((raw_db + 100.0) / 2.0 - 100.0).max(-100.0).min(0.0)  // Map -100..100 to -100..0
    } else {
        -100.0
    }
} 