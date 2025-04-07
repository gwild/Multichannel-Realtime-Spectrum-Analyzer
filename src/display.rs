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
                let magnitudes = values.par_iter()
                    .map(|&(freq, raw_val)| {
                        format!("({:.2}, {:.0})", freq, raw_val)
                    })
                    .collect::<Vec<_>>()
                    .join(", ");

                format!("Channel {}: [{}]", channel + 1, magnitudes)
            })
            .collect()
    }
} 