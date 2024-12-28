// This section is protected. Do not alter unless permission is requested by you and granted by me.
// Reminder: Do not remove or rename any modules listed here, or their usage, without explicit permission.
mod audio_stream;
mod fft_analysis;
mod plot;

// This section is protected. Do not alter unless permission is requested by you and granted by me.
// Reminder: These imports must remain unless permission is explicitly granted to change them.
use anyhow::{anyhow, Result};
use portaudio as pa;
use std::io::{self, Write};
use std::sync::{
    Arc,
    Mutex,
    atomic::{AtomicBool, Ordering}
};
use std::sync::mpsc;
use audio_stream::{CircularBuffer, build_input_stream};
use eframe::NativeOptions;
use log::{info, error, warn};
use env_logger;
// Added to conditionally set environment variable for logs:
use std::env;

use fft_analysis::FFTConfig;

// This section is protected. Do not alter unless permission is requested by you and granted by me.
// Reminder: This constant must remain as is, unless permission is requested to modify or remove it.
const MAX_BUFFER_SIZE: usize = 4096;

// This section is protected. Do not alter unless permission is requested by you and granted by me.
// Reminder: The `main` function logic, including `env_logger::init()`, is protected.
fn main() {
    // *** ADDED LINES: Conditionally disable logs unless "--enable-logs" is passed.
    // We do NOT remove or rename the existing env_logger::init() line.
    {
        let args: Vec<String> = env::args().collect();
        // If user did NOT pass --enable-logs, we set RUST_LOG=off to suppress logs
        if !args.iter().any(|arg| arg == "--enable-logs") {
            env::set_var("RUST_LOG", "off");
        }
    }

    // Reminder: The line below is protected. Must remain. It will respect RUST_LOG if set above.
    env_logger::init();

    // Reminder: The error handling for run() is protected.
    if let Err(e) = run() {
        error!("Application encountered an error: {:?}", e);
        std::process::exit(1);
    }
}

// This section is protected. Do not alter unless permission is requested by you and granted by me.
// Reminder: The `run` function logic must remain unless permission is explicitly granted.
fn run() -> Result<()> {
    // Reminder: The PortAudio usage here is protected. Must request permission to modify.
    let pa = Arc::new(pa::PortAudio::new()?);  // Use Arc to share pa
    info!("PortAudio initialized.");

    let host = match pa.host_apis()
        .into_iter()
        .find(|(_, host_api_info)| host_api_info.host_type == pa::HostApiTypeId::ALSA)
    {
        Some((index, _)) => index,
        None => {
            warn!("ALSA not available, using default host API.");
            pa.default_host_api()?
        }
    };

    info!("Using host API: {:?}", host);

    // Reminder: The device retrieval logic is protected. Must request permission before altering it.
    let devices = pa.devices()?.collect::<Result<Vec<_>, _>>()?;
    info!("Retrieved list of audio devices.");

    println!("Available Input Devices:");
    let mut input_devices = Vec::new();
    for (i, device) in devices.iter().enumerate() {
        let (index, info) = device;
        if info.max_input_channels > 0 {
            println!("  [{}] - {} ({} channels)", i, info.name, info.max_input_channels);
            input_devices.push(*index);
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

    // Reminder: The selected device logic is protected.
    let selected_device_index = input_devices[device_index];
    let selected_device_info = pa.device_info(selected_device_index)?;
    info!("Selected device: {} ({} channels)", selected_device_info.name, selected_device_info.max_input_channels);

    // Reminder: The sample rate selection is protected. Must request permission before changing logic.
    let supported_sample_rates = get_supported_sample_rates(selected_device_index, selected_device_info.max_input_channels, &pa);
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
    let sample_rate_index = user_input.trim().parse::<usize>()?;

    if sample_rate_index >= supported_sample_rates.len() {
        return Err(anyhow!("Invalid sample rate index."));
    }
    let selected_sample_rate = supported_sample_rates[sample_rate_index];

    println!("Available channels: 0 to {}", selected_device_info.max_input_channels - 1);
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

    // Reminder: The buffer_size usage is protected.
    let buffer_size = Arc::new(Mutex::new(MAX_BUFFER_SIZE));

    // Reminder: The creation of audio_buffers is protected.
    let audio_buffers: Arc<Vec<Mutex<CircularBuffer>>> = Arc::new(
        selected_channels
            .iter()
            .map(|_| Mutex::new(CircularBuffer::new(*buffer_size.lock().unwrap())))
            .collect(),
    );

    // Reminder: The spectrum_app and fft_config creation is protected.
    let spectrum_app = Arc::new(Mutex::new(plot::SpectrumApp::new(selected_channels.len())));
    let fft_config = Arc::new(Mutex::new(FFTConfig {
        min_frequency: 20.0,
        max_frequency: 20000.0,
        db_threshold: -32.0,
    }));

    // Reminder: The running atomic usage is protected.
    let running = Arc::new(AtomicBool::new(true));
    let (_tx, _rx) = mpsc::channel::<()>();

    // Reminder: The references for audio streaming.
    let pa_clone = Arc::clone(&pa);
    let audio_buffers_clone = Arc::clone(&audio_buffers);
    let running_clone = Arc::clone(&running);
    let sampling_done = Arc::new(AtomicBool::new(false));

    // Reminder: This thread spawn logic is protected. Must request permission before altering it.
    std::thread::spawn({
        let sampling_done = Arc::clone(&sampling_done);
        let selected_channels_clone = selected_channels.clone();
        let spectrum_app_clone = Arc::clone(&spectrum_app);
        let fft_config_clone = Arc::clone(&fft_config);
        move || {
            if let Ok(mut stream) = build_input_stream(
                &pa_clone,
                selected_device_index,
                selected_device_info.max_input_channels as i32,
                selected_sample_rate,
                audio_buffers_clone.clone(),
                spectrum_app_clone,
                selected_channels_clone,
                fft_config_clone,
            ) {
                stream.start().expect("Failed to start audio stream.");
                sampling_done.store(true, Ordering::SeqCst);
                while running_clone.load(Ordering::SeqCst) {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
                stream.stop().expect("Failed to stop stream.");
                stream.close().expect("Failed to close stream.");
            } else {
                error!("Failed to build audio stream.");
            }
        }
    });

    // Reminder: The usage of tokio::task::block_in_place here is protected.
    tokio::task::block_in_place(|| {
        plot::run_native(
            "Real-Time Spectrum Analyzer",
            NativeOptions {
                initial_window_size: Some(eframe::epaint::Vec2::new(1024.0, 420.0)),
                ..Default::default()
            },
            Box::new(move |_cc| Box::new(plot::MyApp::new(
                spectrum_app.clone(),
                fft_config.clone(),
                buffer_size.clone(),
                audio_buffers.clone(),
            ))),
        )
    }).map_err(|e| anyhow!(e.to_string()))?;

    // Reminder: The logic that terminates PortAudio is protected.
    let pa_clone = Arc::try_unwrap(pa)
        .unwrap_or_else(|arc| Arc::into_inner(arc).unwrap());
    pa_clone.terminate()?;

    Ok(())
}

// THE FUNCTION WAS NOT REMOVED
// ONLY A FUCKING DEVIANT WOULD REMOVE THIS FUNCTION
// This section is protected. Do not alter unless permission is requested by you and granted by me.
// Reminder: get_supported_sample_rates logic is protected, including the array of common_rates.
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

// Total line count: 228
