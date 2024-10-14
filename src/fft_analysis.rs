use rustfft::{FftPlanner, num_complex::Complex};

const NUM_PARTIALS: usize = 12;
const MIN_FREQUENCY: f32 = 20.0;
const MAX_FREQUENCY: f32 = 1000.0;
const DB_THRESHOLD: f32 = -24.0; // 24 dB threshold

/// Computes the FFT spectrum from the given audio buffer.
/// Returns a vector of (frequency, amplitude_db) pairs.
pub fn compute_spectrum(buffer: &[f32], sample_rate: u32) -> Vec<(f32, f32)> {
    if buffer.is_empty() {
        return Vec::new();
    }

    // Compute linear amplitude threshold at runtime
    let linear_threshold = 10.0_f32.powf(DB_THRESHOLD / 20.0);

    // Filter the signal in the time domain based on amplitude
    let filtered_buffer: Vec<f32> = buffer
        .iter()
        .cloned()
        .filter(|&sample| sample.abs() >= linear_threshold) // Keep only samples above the linear amplitude threshold
        .collect();

    if filtered_buffer.is_empty() {
        return Vec::new(); // Return early if no signal is above the threshold
    }

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(filtered_buffer.len());

    // Prepare the complex buffer for FFT
    let mut complex_buffer: Vec<Complex<f32>> = filtered_buffer
        .iter()
        .map(|&x| Complex { re: x, im: 0.0 })
        .collect();
    fft.process(&mut complex_buffer);

    // Calculate magnitudes and frequencies
    let mut magnitudes: Vec<_> = complex_buffer
        .iter()
        .enumerate()
        .map(|(i, &value)| {
            let frequency = (i as f32) * (sample_rate as f32) / (filtered_buffer.len() as f32);
            let amplitude = value.norm();
            let amplitude_db = if amplitude > 0.0 {
                20.0 * amplitude.log(10.0)
            } else {
                f32::MIN // Avoid logarithm of zero by using a small negative value
            };
            (frequency, amplitude_db)
        })
        .collect();

    // Filter frequencies based on the defined limits
    magnitudes.retain(|&(freq, _)| freq >= MIN_FREQUENCY && freq <= MAX_FREQUENCY);
    magnitudes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Filter out frequencies that are too close together or below the dB threshold
    let mut filtered = Vec::new();
    let mut last_frequency = -1.0;

    for &(frequency, amplitude_db) in &magnitudes {
        if frequency - last_frequency >= MIN_FREQUENCY && amplitude_db >= DB_THRESHOLD {
            filtered.push((frequency, amplitude_db));
            last_frequency = frequency;
        }
    }

    // Limit the number of results to NUM_PARTIALS
    filtered.iter()
        .take(NUM_PARTIALS)
        .map(|&(frequency, amplitude)| {
            (round_to_two_decimals(frequency), round_to_two_decimals(amplitude))
        })
        .collect()
}

/// Rounds a floating point number to two decimal places.
fn round_to_two_decimals(value: f32) -> f32 {
    (value * 100.0).round() / 100.0
}
