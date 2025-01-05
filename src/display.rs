use rayon::prelude::*;

/// Handles conversion of FFT data for display purposes only.
/// This is completely separate from plotting and FFT calculations.
pub struct SpectralDisplay {
    channels: Vec<Vec<(f32, f32)>>  // All channels' data
}

impl SpectralDisplay {
    pub fn new(channels: &[Vec<(f32, f32)>]) -> Self {
        Self {
            channels: channels.to_vec()
        }
    }

    /// Convert all channels' dB values to magnitudes at once
    pub fn format_all(&self) -> Vec<String> {
        self.channels.par_iter()
            .enumerate()
            .map(|(channel, values)| {
                let magnitudes = values.par_iter()
                    .map(|&(freq, db)| {
                        let magnitude = if db > -100.0 {
                            let power = 10.0f32.powf(db / 10.0);
                            power.sqrt().round() as i32
                        } else {
                            0
                        };
                        format!("({:.1}, {})", freq, magnitude)
                    })
                    .collect::<Vec<_>>()
                    .join(", ");

                format!("Channel {}: [{}]", channel + 1, magnitudes)
            })
            .collect()
    }
} 