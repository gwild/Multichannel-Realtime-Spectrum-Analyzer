// src/main.rs

mod audio_stream;
mod fft_analysis;
mod plot;
mod conversion;

use anyhow::{anyhow, Result};
use portaudio as pa;
use std::io::{self, Write};
use std::sync::{Arc, Mutex};
use audio_stream::{build_input_stream, CircularBuffer};
use eframe::NativeOptions;

const MAX_BUFFER_SIZE: usize = 512;

fn main() {
    // Execute the run function and handle errors explicitly
    if let Err(e) = run() {
        eprintln!("Error: {:?}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    // Initialize PortAudio
    let pa = pa::PortAudio::new()?;

    // Get a list of available devices
    let devices = pa.devices()?.collect::<Result<Vec<_>, _>>()?;

    // Print available input devices
    println!("Available Input Devices:");
    let mut input_devices = Vec::new();
    for (i, device) in devices.iter().enumerate() {
        let (index, info) = device;
        if info.max_input_channels > 0 {
            println!("  [{}] - {}", i, info.name);
            input_devices.push(*index);
        }
    }

    // Prompt user for device selection
    print!("Enter the index of the desired device: ");
    io::stdout().flush()?;
    let mut user_input = String::new();
    io::stdin().read_line(&mut user_input)?;

    let device_index = user_input
        .trim()
        .parse::<usize>()
        .map_err(|_| anyhow!("Invalid device index"))?;

    if device_index >= input_devices.len() {
        return Err(anyhow!("Invalid device index. Exiting."));
    }

    let selected_device_index = input_devices[device_index];
    let selected_device_info = pa.device_info(selected_device_index)?;

    println!("Selected device: {}", selected_device_info.name);

    let default_sample_rate = selected_device_info.default_sample_rate as f64;

    // Print supported configurations
    println!("Supported input configs:");
    println!(
        "  Channels: 1 - {}",
        selected_device_info.max_input_channels
    );
    println!("  Sample Rate: {} Hz", default_sample_rate);

    // Handle or remove the unused `latency` variable
    // If you plan to use it later, prefix with an underscore
    // let _latency = selected_device_info.default_low_input_latency;

    println!(
        "\nEnter the number of channels to use (max {}):",
        selected_device_info.max_input_channels
    );
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let num_channels: i32 = input
        .trim()
        .parse()
        .map_err(|_| anyhow!("Invalid number of channels"))?;
    if num_channels < 1 || num_channels > selected_device_info.max_input_channels {
        return Err(anyhow!(
            "Invalid number of channels. Maximum is {}",
            selected_device_info.max_input_channels
        ));
    }

    println!("\nEnter the channel numbers to use (comma-separated, e.g., 0,1):");
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let selected_channels: Vec<usize> = input
        .trim()
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    let max_channel = num_channels as usize - 1;
    for &channel in &selected_channels {
        if channel > max_channel {
            return Err(anyhow!(
                "Invalid channel selected. Maximum channel is {}",
                max_channel
            ));
        }
    }

    println!("Selected channels: {:?}", selected_channels);

    // Initialize audio buffers
    let audio_buffers: Arc<Vec<Mutex<CircularBuffer>>> = Arc::new(
        selected_channels
            .iter()
            .map(|_| Mutex::new(CircularBuffer::new(MAX_BUFFER_SIZE)))
            .collect(),
    );

    // Initialize the spectrum analyzer application
    let spectrum_app = Arc::new(Mutex::new(plot::SpectrumApp::new(selected_channels.len())));

    // Stream setup
    // Removed the unused `latency` variable

    // Define stream parameters compatible with audio_stream.rs
    let num_channels_i32 = num_channels; // Ensure it's i32 as expected
    let stream = build_input_stream(
        &pa,
        selected_device_index,
        num_channels_i32,
        default_sample_rate,
        audio_buffers.clone(),
        spectrum_app.clone(),
        selected_channels.clone(),
    )?;

    // Declare `stream` as mutable to call `start()`
    let mut stream = stream;

    // Start the stream
    stream.start()?;
    println!(
        "Stream started with {} channels at {} Hz.",
        num_channels, default_sample_rate
    );

    // Run GUI in the main thread
    let native_options = NativeOptions {
        initial_window_size: Some(eframe::epaint::Vec2::new(960.0, 420.0)),
        ..Default::default()
    };

    // Handle eframe::run_native separately to avoid trait incompatibilities
    if let Err(e) = plot::run_native(
        "Real-Time Spectrum Analyzer",
        native_options,
        Box::new(move |_cc| Box::new(plot::MyApp::new(spectrum_app))),
    ) {
        eprintln!("Error launching application: {:?}", e);
    }

    // Clean up PortAudio explicitly (optional, as it will be cleaned up when `pa` goes out of scope)
    pa.terminate()?;

    Ok(())
}
