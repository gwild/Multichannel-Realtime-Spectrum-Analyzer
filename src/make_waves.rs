use std::f32::consts::PI;
use log::debug;

/// Applies a linear fade in/out envelope to a wavetable
/// 
/// # Arguments
/// 
/// * `wavetable` - The wavetable to apply the envelope to
/// * `fade_samples` - Number of samples to fade in/out at start/end
/// 
/// # Returns
/// 
/// A new wavetable with the envelope applied
fn apply_fade_envelope(wavetable: &[f32], fade_samples: usize) -> Vec<f32> {
    let mut result = wavetable.to_vec();
    let total_samples = wavetable.len();
    
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

/// Builds a wavetable from a set of partials with support for crossfading
/// 
/// # Arguments
/// 
/// * `partials` - Vector of (frequency, amplitude) pairs
/// * `sample_rate` - Audio sample rate in Hz
/// * `update_rate` - How often the wavetable is updated in seconds
/// 
/// # Returns
/// 
/// A vector containing the wavetable samples with fade in/out regions
pub fn build_wavetable(partials: &[(f32, f32)], sample_rate: f32, update_rate: f32) -> Vec<f32> {
    // Calculate base wavetable size for one update period
    let base_size = (sample_rate * update_rate) as usize;
    
    // Calculate fade duration (1/3 of update period)
    let fade_samples = (base_size as f32 / 3.0) as usize;
    
    // Make wavetable longer to support crossfading
    let wavetable_size = base_size + fade_samples;
    let mut wavetable = vec![0.0; wavetable_size];

    // For each partial, add its contribution to the wavetable
    for &(freq, amp) in partials.iter().filter(|&&(f, a)| f > 0.0 && a > 0.0) {
        let phase_delta = 2.0 * PI * freq / sample_rate;
        for i in 0..wavetable_size {
            let phase = phase_delta * i as f32;
            wavetable[i] += amp * phase.sin();
        }
    }

    // Normalize the wavetable
    let max_amplitude = wavetable.iter()
        .fold(0.0f32, |max, &x| max.max(x.abs()));
    
    if max_amplitude > 0.0 {
        for sample in wavetable.iter_mut() {
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
        if period < wavetable.len() {
            let start = wavetable.len() - period;
            let end = wavetable.len();
            let last_val = wavetable[end - 1];
            for i in start..end {
                let t = (i - start) as f32 / (period - 1) as f32;
                wavetable[i] = wavetable[i] * (1.0 - t) + 0.0 * t; // Linear ramp to zero
            }
            // Ensure last sample is exactly zero
            wavetable[end - 1] = 0.0;
        }
    }
    // --- End zero crossing logic ---

    // Apply fade envelope
    wavetable = apply_fade_envelope(&wavetable, fade_samples);

    debug!("Built wavetable - Size: {}, Fade samples: {}, Update rate: {}s", 
           wavetable_size, fade_samples, update_rate);

    wavetable
}

/// Formats partials for debug output
pub fn format_partials_debug(partials: &[(f32, f32)]) -> String {
    partials.iter()
        .filter(|&&(f, a)| f > 0.0 && a > 0.0)
        .map(|&(freq, amp)| format!("({:.1} Hz, {:.3})", freq, amp))
        .collect::<Vec<_>>()
        .join(", ")
} 