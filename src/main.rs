mod audio_stream;
mod fft_analysis;
mod plot;
mod convert; // Add the new module

use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait};
use cpal::{SampleFormat, StreamConfig};
use std::io::{self, Write};

fn main() -> Result<()> {
    let host = cpal::default_host();

    // Get a list of available input devices
    let devices: Vec<_> = host.devices()?.collect();

    // Print device names and indexes
    println!("Available Input Devices:");
    for (i, device) in devices.iter().enumerate() {
        println!("  [{}] - {}", i, device.name()?);
    }

    // Prompt user for device selection
    print!("Enter the index of the desired device: ");
    io::stdout().flush()?;
    let mut user_input = String::new();
    io::stdin().read_line(&mut user_input)?;

    // Parse user input
    let device_index = user_input.trim().parse::<usize>()?;

    // Check for valid input within available devices
    if device_index < devices.len() {
        let selected_device = devices[device_index].clone();

        // Retrieve supported input configurations
        let supported_configs: Vec<_> = selected_device.supported_input_configs()?.collect();

        // Print available sample formats and configurations
        println!("Supported Input Configurations for {}:", selected_device.name()?);
        for config in &supported_configs {
            println!(
                "  - {} Hz ({} channels, format: {:?})",
                config.min_sample_rate().0,
                config.channels(),
                config.sample_format()
            );
        }

        // Example check for F32 support
        let supports_f32 = supported_configs.iter().any(|c| c.sample_format() == SampleFormat::F32);
        if supports_f32 {
            println!("The device supports F32 sample format.");
        } else {
            println!("The device does NOT support F32 sample format.");
        }
    } else {
        println!("Invalid device index. Exiting.");
    }

    Ok(())
}
