use log::info;

pub const MIN_FREQ: f64 = 20.0;  // Lowest frequency we want to analyze
pub const MAX_FREQ: f64 = 2000.0;  // Highest frequency we want to analyze
pub const MIN_BUFFER_SIZE: usize = 512;  // Increased from previous value
pub const MAX_BUFFER_SIZE: usize = 16384;  // Absolute maximum buffer size
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