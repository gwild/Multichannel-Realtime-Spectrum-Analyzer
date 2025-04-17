use rayon::prelude::*;
use log::warn;
use crate::plot::display_utils;  // Import our utility function

pub struct SpectralDisplay {
    channels: Vec<Vec<(f32, f32)>>,
    fft_line_data: Vec<Vec<(f32, f32)>>,
    num_partials: usize,  // Add field to track the number of partials
}

impl SpectralDisplay {
    pub fn new(channels: &[Vec<(f32, f32)>]) -> Self {
        // Determine the number of partials based on first channel
        let num_partials = if !channels.is_empty() { 
            channels[0].len() 
        } else { 
            crate::DEFAULT_NUM_PARTIALS 
        };
        
        Self {
            channels: channels.to_vec(),
            fft_line_data: Vec::new(),
            num_partials,
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
                // Use our utility function for consistent formatting
                let magnitudes = display_utils::format_partials(values, self.num_partials);
                format!("Channel {}: [{}]", channel + 1, magnitudes)
            })
            .collect()
    }
} 