use rayon::prelude::*;
use log::warn;

pub struct SpectralDisplay {
    channels: Vec<Vec<(f32, f32)>>,
    fft_line_data: Vec<Vec<(f32, f32)>>,
}

impl SpectralDisplay {
    pub fn new(channels: &[Vec<(f32, f32)>]) -> Self {
        Self {
            channels: channels.to_vec(),
            fft_line_data: Vec::new(),
        }
    }

    pub fn update_fft_data(&mut self, fft_data: Vec<Vec<(f32, f32)>>) {
        if fft_data.is_empty() {
            warn!("Received empty fft_data! This may indicate a problem reading from the audio stream.");
        }
        self.fft_line_data = fft_data;
    }

    pub fn format_all(&self) -> Vec<String> {
        self.channels.par_iter()
            .enumerate()
            .map(|(channel, values)| {
                // Always format exactly 12 values
                let magnitudes = (0..12)
                    .map(|i| {
                        if i < values.len() {
                            let (freq, raw_val) = values[i];
                            if raw_val > 0.0 {
                                format!("({:.2}, {:.0})", freq, raw_val)
                            } else {
                                "(0.00, 0)".to_string()
                            }
                        } else {
                            "(0.00, 0)".to_string()
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(", ");

                format!("Channel {}: [{}]", channel + 1, magnitudes)
            })
            .collect()
    }
} 