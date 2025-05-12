use std::f32::consts::PI;
use log::debug;

/// Applies a linear fade in/out envelope to a buffer
/// 
/// # Arguments
/// 
/// * `buffer` - The buffer to apply the envelope to
/// * `fade_samples` - Number of samples to fade in/out at start/end
/// 
/// # Returns
/// 
/// A new buffer with the envelope applied
fn apply_fade_envelope(buffer: &[f32], fade_samples: usize) -> Vec<f32> {
    let mut result = buffer.to_vec();
    let total_samples = buffer.len();
    
    // Apply fade in
    for i in 0..fade_samples {
        let fade_factor = i as f32 / fade_samples as f32;
        result[i] *= fade_factor;
    }
    
    // Apply fade out
    for i in 0..fade_samples {
        let fade_factor = i as f32 / fade_samples as f32;
        result[total_samples - fade_samples + i] *= (1.0 - fade_factor);
    }
    
    result
}

/// Builds a segment buffer from a set of partials with support for crossfading
/// 
/// # Arguments
/// 
/// * `partials` - Vector of (frequency, amplitude) pairs
/// * `sample_rate` - Audio sample rate in Hz
/// * `update_rate` - How often the segment is updated in seconds
/// 
/// # Returns
/// 
/// A vector containing the segment buffer samples with fade in/out regions
pub fn build_segment_buffer(partials: &[(f32, f32)], sample_rate: f32, update_rate: f32) -> Vec<f32> {
    // Calculate base buffer size for one update period
    let base_size = (sample_rate * update_rate) as usize;
    
    // Calculate fade duration (1/3 of update period)
    let fade_samples = (base_size as f32 / 3.0) as usize;
    
    let buffer_size = base_size; // No extra samples for looping
    let mut buffer = vec![0.0; buffer_size];

    // For each partial, add its contribution to the buffer
    for &(freq, amp) in partials.iter().filter(|&&(f, a)| f > 0.0 && a > 0.0) {
        let phase_delta = 2.0 * PI * freq / sample_rate;
        for i in 0..buffer_size {
            let phase = phase_delta * i as f32;
            buffer[i] += amp * phase.sin();
        }
    }

    // Normalize the buffer
    let max_amplitude = buffer.iter().fold(0.0f32, |max, &x| max.max(x.abs()));
    
    if max_amplitude > 0.0 {
        for sample in buffer.iter_mut() {
            *sample /= max_amplitude;
        }
    }

    // --- Ensure last cycle ends at zero (zero crossing) ---
    // Find fundamental frequency (lowest nonzero freq)
    let fundamental = partials.iter()
        .filter(|&&(f, a)| f > 0.0 && a > 0.0)
        .map(|&(f, _)| f)
        .fold(f32::INFINITY, |min, f| if f < min { f } else { min });
    if fundamental < f32::INFINITY && fundamental > 0.0 {
        let period = (sample_rate / fundamental).round() as usize;
        if period < buffer.len() {
            let start = buffer.len() - period;
            let end = buffer.len();
            for i in start..end {
                let t = (i - start) as f32 / (period - 1) as f32;
                buffer[i] = buffer[i] * (1.0 - t) + 0.0 * t; // Linear ramp to zero
            }
            // Ensure last sample is exactly zero
            buffer[end - 1] = 0.0;
        }
    }
    // --- End zero crossing logic ---

    // Apply fade envelope
    buffer = apply_fade_envelope(&buffer, fade_samples);

    debug!("Built segment buffer - Size: {}, Fade samples: {}, Update rate: {}s", buffer_size, fade_samples, update_rate);

    buffer
}

/// Formats partials for debug output
pub fn format_partials_debug(partials: &[(f32, f32)]) -> String {
    partials.iter()
        .filter(|&&(f, a)| f > 0.0 && a > 0.0)
        .map(|&(freq, amp)| format!("({:.1} Hz, {:.3})", freq, amp))
        .collect::<Vec<_>>()
        .join(", ")
} 