mod audio_stream;
mod fft_analysis;
mod plot;

use anyhow::{anyhow, Result};
use portaudio as pa;
use std::io::{self, Write};
use std::sync::{
    Arc,
    Mutex,
    RwLock,
    atomic::{AtomicBool, Ordering}
};
use audio_stream::{CircularBuffer, build_input_stream};
use eframe::NativeOptions;
use log::{info, error, warn};
use env_logger;
use std::env;
use eframe::run_native;


use fft_analysis::FFTConfig;

const MAX_BUFFER_SIZE: usize = 4096;

fn main() {
    {
        let args: Vec<String> = env::args().collect();
        if !args.iter().any(|arg| arg == "--enable-logs") {
            env::set_var("RUST_LOG", "off");
        }
    }

    env_logger::init();

    if let Err(e) = run() {
        error!("Application encountered an error: {:?}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let pa = Arc::new(pa::PortAudio::new()?);
    info!("PortAudio initialized.");

    let devices = pa.devices()?.collect::<Result<Vec<_>, _>>()?;
    if devices.is_empty() {
        warn!("No devices found. Attempting to reset devices.");
        reset_audio_devices(&pa)?;
    }

    let devices = pa.devices()?.collect::<Result<Vec<_>, _>>()?;
    if devices.is_empty() {
        return Err(anyhow!("No audio devices available after reset."));
    }

    info!("Retrieved list of audio devices.");
    println!("Available Input Devices:");
    let mut input_devices = Vec::new();

    for (i, device) in devices.iter().enumerate() {
        let (index, info) = device;
        if info.max_input_channels > 0 {
            println!("  [{}] - {} ({} channels)", i, info.name, info.max_input_channels);
            if ensure_audio_device_ready(&pa, *index) {
                input_devices.push(*index);
            } else {
                warn!("Device {} is not ready for use.", info.name);
            }
        }
    }

    if input_devices.is_empty() {
        return Err(anyhow!("No input audio devices found."));
    }

    print!("Enter the index of the desired device: ");
    io::stdout().flush()?;
    let mut user_input = String::new();
    io::stdin().read_line(&mut user_input)?;
    let device_index = user_input
        .trim()
        .parse::<usize>()
        .map_err(|_| anyhow!("Invalid device index"))?;

    if device_index >= input_devices.len() {
        return Err(anyhow!("Invalid device index."));
    }
    let selected_device_index = input_devices[device_index];
    let selected_device_info = pa.device_info(selected_device_index)?;
    info!(
        "Selected device: {} ({} channels)",
        selected_device_info.name, selected_device_info.max_input_channels
    );

    let supported_sample_rates = get_supported_sample_rates(
        selected_device_index,
        selected_device_info.max_input_channels,
        &pa,
    );
    if supported_sample_rates.is_empty() {
        return Err(anyhow!("No supported sample rates for the selected device."));
    }

    println!("Supported sample rates:");
    for (i, rate) in supported_sample_rates.iter().enumerate() {
        println!("  [{}] - {} Hz", i, rate);
    }

    print!("Enter the index of the desired sample rate: ");
    io::stdout().flush()?;
    user_input.clear();
    io::stdin().read_line(&mut user_input)?;
    let sample_rate_index = user_input
        .trim()
        .parse::<usize>()
        .map_err(|_| anyhow!("Invalid sample rate index"))?;

    if sample_rate_index >= supported_sample_rates.len() {
        return Err(anyhow!("Invalid sample rate index."));
    }
    let selected_sample_rate = supported_sample_rates[sample_rate_index];

    println!(
        "Available channels: 0 to {}",
        selected_device_info.max_input_channels - 1
    );
    println!("Enter channels to use (comma-separated, e.g., 0,1): ");
    user_input.clear();
    io::stdin().read_line(&mut user_input)?;
    let selected_channels: Vec<usize> = user_input
        .trim()
        .split(',')
        .filter_map(|s| s.parse::<usize>().ok())
        .filter(|&ch| ch < selected_device_info.max_input_channels as usize)
        .collect();

    if selected_channels.is_empty() {
        return Err(anyhow!("No valid channels selected."));
    }
    info!("Selected channels: {:?}", selected_channels);

    let buffer_size = Arc::new(Mutex::new(MAX_BUFFER_SIZE));
    let audio_buffer = Arc::new(RwLock::new(CircularBuffer::new(
        *buffer_size.lock().unwrap(),
        selected_device_info.max_input_channels as usize,
    )));
    let spectrum_app = Arc::new(Mutex::new(plot::SpectrumApp::new(selected_channels.len())));
    let fft_config = Arc::new(Mutex::new(FFTConfig {
        min_frequency: 20.0,
        max_frequency: 20000.0,
        db_threshold: -32.0,
    }));

    let shutdown_flag = Arc::new(AtomicBool::new(false));
let shutdown_flag_for_thread = Arc::clone(&shutdown_flag);  // Clone for thread
let shutdown_flag_for_gui = Arc::clone(&shutdown_flag);     // Clone for GUI

std::thread::spawn(move || {
    while !shutdown_flag_for_thread.load(Ordering::Relaxed) {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
});

plot::run_native(
    "Real-Time Spectrum Analyzer",
    NativeOptions::default(),
    Box::new(move |_cc| {
        Box::new(plot::MyApp::new(
            spectrum_app.clone(),
            fft_config.clone(),
            buffer_size.clone(),
            audio_buffer.clone(),
            shutdown_flag_for_gui,  // Use the cloned version here
        ))
    }),
).expect("Failed to run native plot");

    Ok(())
}
fn get_supported_sample_rates(
    device_index: pa::DeviceIndex,
    num_channels: i32,
    pa: &pa::PortAudio,
) -> Vec<f64> {
    let common_rates = [8000.0, 16000.0, 22050.0, 44100.0, 48000.0, 96000.0, 192000.0];
    common_rates
        .iter()
        .cloned()
        .filter(|&rate| {
            let params = pa::StreamParameters::<f32>::new(device_index, num_channels, true, 0.0);
            pa.is_input_format_supported(params, rate).is_ok()
        })
        .collect()
}

fn reset_audio_devices(pa: &Arc<pa::PortAudio>) -> Result<()> {
    match Arc::try_unwrap(Arc::clone(pa)) {
        Ok(pa_inner) => {
            pa_inner.terminate()?;
        }
        Err(_) => {
            warn!("Unable to terminate PortAudio directly; multiple references exist.");
        }
    }
    let _ = pa::PortAudio::new()?;
    info!("PortAudio reset successful.");
    Ok(())
}

fn ensure_audio_device_ready(pa: &pa::PortAudio, device_index: pa::DeviceIndex) -> bool {
    let params = pa::StreamParameters::<f32>::new(device_index, 1, true, 0.0);
    if let Ok(mut stream) = pa.open_non_blocking_stream(
        pa::InputStreamSettings::new(params, 48000.0, 512),
        |_args| pa::Continue,
    ) {
        stream.close().is_ok()
    } else {
        false
    }
}
