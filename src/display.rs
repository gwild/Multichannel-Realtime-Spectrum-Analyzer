use rayon::prelude::*;
use crate::utils::map_db_range;

pub struct SpectralDisplay {
    channels: Vec<Vec<(f32, f32)>>
}

impl SpectralDisplay {
    pub fn new(channels: &[Vec<(f32, f32)>]) -> Self {
        Self {
            channels: channels.to_vec()
        }
    }

    pub fn format_all(&self) -> Vec<String> {
        self.channels.par_iter()
            .enumerate()
            .map(|(channel, values)| {
                let magnitudes = values.par_iter()
                    .map(|&(freq, raw_db)| {
                        let db = map_db_range(raw_db);
                        format!("({:6.1}, {:3.0}dB)", freq, db)
                    })
                    .collect::<Vec<_>>()
                    .join(", ");

                format!("Channel {}: [{}]", channel + 1, magnitudes)
            })
            .collect()
    }
} 