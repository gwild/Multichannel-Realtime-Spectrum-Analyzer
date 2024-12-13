// src/main.rs

mod audio_stream;
mod fft_analysis;
mod plot;
mod conversion;

use anyhow::{anyhow, Result};
use portaudio as pa;
use std::io::{self, Write};
use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};
use audio_stream::{build_input_stream, CircularBuffer};
use eframe::NativeOptions;
use log::{info, error};
use env_logger;
use ctrlc;

/// The maximum size of the circular audio buffer.
const MAX_BUFFER_SIZE: usize = 4096;

/// Entry point of the application.
/// Initializes logging, sets up audio streams, and launches the GUI.
fn main() {
    // Initialize the logger
    env_logger::init();

    // Execute the run function and handle errors explicitly
    if let Err(e) = run() {
        error!("Application encountered an error: {:?}", e);
        std::process::exit(1);
    }
}

/// Main application logic.
/// Sets up audio input streams, initializes the spectrum analyzer, and runs the GUI.
fn run() -> Result<()> {
    // Initialize PortAudio
    let pa = pa::PortAudio::new()?;
    info!("PortAudio initialized successfully.");

    // Get a list of available devices
    let devices = pa.devices()?.collect::<Result<Vec<_>, _>>()?;
    info!("Retrieved list of audio devices.");

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

    if input_devices.is_empty() {
        return Err(anyhow!("No input audio devices found."));
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
        return Err(anyhow!(
            "Invalid device index. Please select a valid index from the list."
        ));
    }

    let selected_device_index = input_devices[device_index];
    let selected_device_info = pa.device_info(selected_device_index)?;
    info!(
        "Selected device: {} (Channels: {}, Default Sample Rate: {} Hz)",
        selected_device_info.name,
        selected_device_info.max_input_channels,
        selected_device_info.default_sample_rate
    );

    let default_sample_rate = selected_device_info.default_sample_rate as f64;

    // Print supported configurations
    println!("Supported input configurations:");
    println!(
        "  Channels: 1 - {}",
        selected_device_info.max_input_channels
    );
    println!("  Sample Rate: {} Hz", default_sample_rate);

    // Handle the unused `latency` variable by prefixing with an underscore
    let _latency = selected_device_info.default_low_input_latency;

    // Prompt user for number of channels
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
            "Invalid number of channels. Please select between 1 and {} channels.",
            selected_device_info.max_input_channels
        ));
    }

    // Prompt user for specific channel numbers
    println!("\nEnter the channel numbers to use (comma-separated, e.g., 0,1):");
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let selected_channels: Vec<usize> = input
        .trim()
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    // Validate selected channels
    let max_channel = num_channels as usize - 1;
    for &channel in &selected_channels {
        if channel > max_channel {
            return Err(anyhow!(
                "Invalid channel selected: {}. Maximum channel is {}.",
                channel,
                max_channel
            ));
        }
    }

    if selected_channels.is_empty() {
        return Err(anyhow!("No channels selected. Please select at least one channel."));
    }

    println!("Selected channels: {:?}", selected_channels);
    info!("User selected {} channels: {:?}", selected_channels.len(), selected_channels);

    // Initialize audio buffers
    let audio_buffers: Arc<Vec<Mutex<CircularBuffer>>> = Arc::new(
        selected_channels
            .iter()
            .map(|_| Mutex::new(CircularBuffer::new(MAX_BUFFER_SIZE)))
            .collect(),
    );
    info!("Initialized audio buffers for selected channels.");

    // Initialize the spectrum analyzer application
    let spectrum_app = Arc::new(Mutex::new(plot::SpectrumApp::new(selected_channels.len())));
    info!("Initialized spectrum analyzer application.");

    // Define stream parameters compatible with audio_stream.rs
    let num_channels_i32 = num_channels; // Ensure it's i32 as expected
    let mut stream = build_input_stream(
        &pa,
        selected_device_index,
        num_channels_i32,
        default_sample_rate,
        audio_buffers.clone(),
        spectrum_app.clone(),
        selected_channels.clone(),
    )?;
    info!("Built audio input stream.");

    // Start the stream
    stream.start()?;
    info!(
        "Audio stream started with {} channels at {} Hz.",
        num_channels, default_sample_rate
    );

    // Setup for graceful shutdown
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();

    // Set up Ctrl+C handler for graceful shutdown
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");

    // Clone references for the audio processing thread (if needed)
    let spectrum_app_clone = spectrum_app.clone();

    // Launch the GUI on the main thread
    let native_options = NativeOptions {
        initial_window_size: Some(eframe::epaint::Vec2::new(960.0, 420.0)),
        ..Default::default()
    };

    // Since the GUI runs on the main thread, we'll run it here.
    // The audio processing is handled via PortAudio callbacks.
    // Upon GUI closure, the main thread will proceed to shutdown.
    if let Err(e) = plot::run_native(
        "Real-Time Spectrum Analyzer",
        native_options,
        Box::new(move |_cc| Box::new(plot::MyApp::new(spectrum_app_clone))),
    ) {
        error!("Error launching GUI: {:?}", e);
    }

    info!("GUI closed. Initiating shutdown...");

    // Signal to stop running
    running.store(false, Ordering::SeqCst);

    // Stop the audio stream
    stream.stop()?;
    info!("Audio stream stopped.");

    // Terminate PortAudio
    pa.terminate()?;
    info!("PortAudio terminated.");

    Ok(())
}
