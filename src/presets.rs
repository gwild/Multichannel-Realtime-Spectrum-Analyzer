use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use anyhow::{Result, anyhow};
use log::{info, error, warn};

use crate::fft_analysis::{FFTConfig, WindowType};
use crate::resynth::{ResynthConfig, DEFAULT_UPDATE_RATE};

// A single preset containing all configurable GUI values
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Preset {
    // FFTConfig fields
    pub min_frequency: f64,
    pub max_frequency: f64,
    pub magnitude_threshold: f64,
    pub min_freq_spacing: f64,
    pub window_type: WindowType,
    pub crosstalk_enabled: bool,
    pub crosstalk_threshold: f32,
    pub crosstalk_reduction: f32,
    pub harmonic_tolerance: f32,
    pub root_freq_min: f32,
    pub root_freq_max: f32,
    pub freq_match_distance: f32,
    pub fft_gain: f32,

    // ResynthConfig fields
    pub gain: f32,
    pub freq_scale: f32,
    pub update_rate: f32,

    // MyApp display fields
    pub y_scale: f32,
    pub alpha: u8,
    pub bar_width: f32,
    pub show_line_plot: bool,
    pub show_spectrograph: bool,
    pub show_results: bool,
    pub buffer_size: usize,
    // Note: buffer_size is handled separately and not part of a preset
}

// Manages loading, saving, and holding presets
pub struct PresetManager {
    pub presets: BTreeMap<String, Preset>,
    file_path: String,
}

impl PresetManager {
    pub fn new(file_path: &str) -> Result<Self> {
        let mut presets = BTreeMap::new();
        if Path::new(file_path).exists() {
            info!("Loading presets from {}", file_path);
            let yaml_str = fs::read_to_string(file_path)?;
            presets = serde_yaml::from_str(&yaml_str)
                .map_err(|e| anyhow!("Failed to parse presets.yaml: {}", e))?;
        } else {
            info!("No presets file found at {}. Creating with default preset.", file_path);
            let default_preset = Self::get_default_preset();
            presets.insert("default".to_string(), default_preset);
        }

        let mut manager = Self {
            presets,
            file_path: file_path.to_string(),
        };

        // Ensure the default preset exists and save if it was just created
        if !manager.presets.contains_key("default") {
            warn!("'default' preset not found. Creating and saving it.");
            manager.presets.insert("default".to_string(), Self::get_default_preset());
        }
        
        manager.save()?; // Save to create the file or ensure it's up-to-date

        Ok(manager)
    }

    pub fn save(&self) -> Result<()> {
        let yaml_str = serde_yaml::to_string(&self.presets)?;
        fs::write(&self.file_path, yaml_str)?;
        info!("Presets saved to {}", self.file_path);
        Ok(())
    }
    
    // This creates the "default" preset based on the logic from the "Reset to Defaults" button
    pub fn get_default_preset() -> Preset {
        let fft_config = FFTConfig::default();
        // The reset logic also sets max_frequency to 1400.0
        
        Preset {
            // FFTConfig fields
            min_frequency: fft_config.min_frequency,
            max_frequency: 1400.0, // Specific default value from reset logic
            magnitude_threshold: fft_config.magnitude_threshold,
            min_freq_spacing: fft_config.min_freq_spacing,
            window_type: fft_config.window_type,
            crosstalk_enabled: fft_config.crosstalk_enabled,
            crosstalk_threshold: fft_config.crosstalk_threshold,
            crosstalk_reduction: fft_config.crosstalk_reduction,
            harmonic_tolerance: fft_config.harmonic_tolerance,
            root_freq_min: fft_config.root_freq_min,
            root_freq_max: fft_config.root_freq_max,
            freq_match_distance: fft_config.freq_match_distance,
            fft_gain: fft_config.gain,

            // ResynthConfig fields
            gain: 0.5,
            freq_scale: 1.0,
            update_rate: DEFAULT_UPDATE_RATE,

            // MyApp display fields
            y_scale: 80.0,
            alpha: 255,
            bar_width: 5.0,
            show_line_plot: false,
            show_spectrograph: false,
            show_results: true,
            buffer_size: crate::DEFAULT_BUFFER_SIZE,
        }
    }
} 