mod audio_stream;
mod fft_analysis;
mod plot;
mod utils;
mod display;
mod resynth;

use anyhow::{anyhow, Result};
use portaudio as pa;
use std::io::{self, Write};
use std::sync::{
    Arc,
    Mutex,
    RwLock,
    atomic::{AtomicBool, Ordering}
};
use audio_stream::{CircularBuffer, start_sampling_thread};
use eframe::NativeOptions;
use log::{info, error, warn};
use env_logger;
use fft_analysis::{FFTConfig, MAX_SPECTROGRAPH_HISTORY};
use utils::{MIN_FREQ, MAX_FREQ, DEFAULT_BUFFER_SIZE};
use crate::fft_analysis::WindowType;
use crate::resynth::{ResynthConfig, start_resynth_thread};
use crate::plot::MyApp;
use std::ffi::CString;
use std::thread;
use std::time::Duration;
use std::collections::VecDeque;
use crate::plot::SpectrographSlice;
use std::time::Instant;

#[derive(Clone)]
pub struct SharedMemory {
    data: Vec<Vec<(f32, f32)>>,
    path: String,
}

// Add a new constant to replace hardcoded 12 throughout code
pub const DEFAULT_NUM_PARTIALS: usize = 12;

fn main() {
    // Parse command line arguments for logging
    let args: Vec<String> = std::env::args().collect();
    let log_level = if args.contains(&"--error".to_string()) {
        "error"
    } else if args.contains(&"--warn".to_string()) {
        "warn"
    } else if args.contains(&"--info".to_string()) {
        "info"
    } else if args.contains(&"--debug".to_string()) {
        "debug"
    } else if args.contains(&"--trace".to_string()) {
        "trace"
    } else if args.contains(&"--enable-logs".to_string()) {
        // Backward compatibility with old flag
        "info"
    } else if std::env::var("RUST_LOG").is_ok() {
        // Keep any manually set RUST_LOG value
        &std::env::var("RUST_LOG").unwrap_or_else(|_| "error".to_string())
    } else {
        // Default to error level only
        "error"
    };

    // Set up logging for all modules in the application
    std::env::set_var("RUST_LOG", format!("audio_streaming={}", log_level));
    env_logger::init();

    // Print logging level information
    if log_level != "error" {
        println!("Logging level: {}", log_level);
        println!("Run with --error, --warn, --info, --debug, or --trace to control verbosity");
    }

    if let Err(e) = run() {
        if std::env::args().any(|arg| arg == "--enable-logs") {
            error!("Application encountered an error: {:?}", e);
        } else {
            error!("Application error: {:?}", e);
        }
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
    
    // Create a mapping of display index to actual device index
    let mut input_devices = Vec::new();
    for (_i, device) in devices.iter().enumerate() {
        let (index, info) = device;
        if info.max_input_channels > 0 {
            println!("  [{}] - {} ({} channels)", input_devices.len(), info.name, info.max_input_channels);
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
        return Err(anyhow!("Invalid device index. Please choose a number between 0 and {}", input_devices.len() - 1));
    }
    
    let selected_device_index = input_devices[device_index];
    let selected_device_info = pa.device_info(selected_device_index)?;
    info!(
        "Selected device: {} ({} channels)",
        selected_device_info.name, selected_device_info.max_input_channels
    );

    if let Ok(device_info) = pa.device_info(selected_device_index) {
        info!("Device: {}", device_info.name);
        info!("Default sample rate: {}", device_info.default_sample_rate);
        info!("Input channels: {}", device_info.max_input_channels);
        info!("Default low latency: {}", device_info.default_low_input_latency);
        info!("Default high latency: {}", device_info.default_high_input_latency);
        
        // Try to get supported formats
        let input_params = pa::StreamParameters::<f32>::new(
            selected_device_index,
            device_info.max_input_channels,
            true,
            device_info.default_low_input_latency
        );
        
        // Test different sample formats
        for &rate in &[44100.0, 48000.0, 96000.0] {
            match pa.is_input_format_supported(input_params, rate) {
                Ok(_) => info!("Sample rate {} Hz is supported", rate),
                Err(e) => info!("Sample rate {} Hz not supported: {}", rate, e)
            }
        }
    }

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
    info!("Selected sample rate: {} Hz", selected_sample_rate);

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

    // Add prompt for number of partials here
    println!("Enter number of partials to detect per channel (default is {}): ", DEFAULT_NUM_PARTIALS);
    user_input.clear();
    io::stdin().read_line(&mut user_input)?;
    let num_partials = if user_input.trim().is_empty() {
        DEFAULT_NUM_PARTIALS
    } else {
        user_input
            .trim()
            .parse::<usize>()
            .map_err(|_| anyhow!("Invalid number of partials"))?
            .max(1) // Ensure at least 1 partial
    };
    info!("Using {} partials per channel", num_partials);

    let buffer_size = Arc::new(Mutex::new(DEFAULT_BUFFER_SIZE));
    let audio_buffer = Arc::new(RwLock::new(CircularBuffer::new(
        DEFAULT_BUFFER_SIZE,
        selected_channels.len()
    )));
    let spectrum_app = Arc::new(Mutex::new(plot::SpectrumApp::new(selected_channels.len())));
    let frames_per_buffer = if cfg!(target_os = "linux") {
        2048u32  // Larger buffer for Linux stability
    } else {
        match selected_sample_rate as u32 {
            48000 => 1024u32,   // Increased for better frequency resolution
            44100 => 1024u32,   // Increased for better frequency resolution
            96000 => 2048u32,   // Increased for higher frequency analysis
            192000 => 4096u32,  // Added option for very high sample rates
            _ => {
                let mut base_size = 1024u32;  // Increased base size
                while base_size * 2 <= (selected_sample_rate / 50.0) as u32 {
                    base_size *= 2;
                }
                base_size
            }
        }
    };

    let mut config = FFTConfig::default();
    // Override only what needs to be different from defaults
    config.num_channels = selected_channels.len();
    config.frames_per_buffer = frames_per_buffer;
    config.num_partials = num_partials;

    let fft_config = Arc::new(Mutex::new(config));

    let running = Arc::new(AtomicBool::new(false));
    let shutdown_flag = Arc::new(AtomicBool::new(false));
    let stream_ready = Arc::new(AtomicBool::new(false));
    let shutdown_complete = Arc::new(AtomicBool::new(false));

    let audio_buffer_clone = Arc::clone(&audio_buffer);
    let selected_channels_clone = selected_channels.clone();
    let buffer_size_clone = Arc::clone(&buffer_size);
    let shutdown_flag_audio = Arc::clone(&shutdown_flag);
    let stream_ready_audio = Arc::clone(&stream_ready);
    let shutdown_complete_audio = Arc::clone(&shutdown_complete);

    // Add before thread creation
    let resynth_config = Arc::new(Mutex::new(ResynthConfig::default()));

    // After input device selection but before thread creation
    println!("\nAvailable Output Devices:");
    let mut output_devices = Vec::new();
    for (_i, device) in devices.iter().enumerate() {
        let (index, info) = device;
        if info.max_output_channels >= 2 {  // Need at least stereo output
            println!("  [{}] - {} ({} channels)", output_devices.len(), info.name, info.max_output_channels);
            output_devices.push(*index);
        }
    }

    if output_devices.is_empty() {
        return Err(anyhow!("No stereo output devices found."));
    }

    print!("Enter the index of the desired output device: ");
    io::stdout().flush()?;
    user_input.clear();
    io::stdin().read_line(&mut user_input)?;
    let output_device_index = user_input
        .trim()
        .parse::<usize>()
        .map_err(|_| anyhow!("Invalid device index"))?;

    if output_device_index >= output_devices.len() {
        return Err(anyhow!("Invalid output device index"));
    }

    let selected_output_device = output_devices[output_device_index];

    // After device selection but before starting threads:
    let control_path = "/dev/shm/audio_control";
    let mut control_file = std::fs::File::create(&control_path)?;
    writeln!(control_file, "{}\n{}\n{}", std::process::id(), selected_channels.len(), num_partials)?;

    // Create shared memory without mutex BEFORE threads start
    let shared_partials = if std::env::args().any(|arg| arg == "--gui-ipc") {
        info!("Starting in GUI+IPC mode");
        let shmem_name = "audio_peaks";
        let shared_memory_path = format!("/dev/shm/{}", shmem_name);
        let file = std::fs::File::create(&shared_memory_path)?;
        file.set_len(4 * 1024 * 1024)?;
        
        Some(SharedMemory {
            data: Vec::new(),
            path: shared_memory_path,
        })
    } else {
        None
    };

    // Before starting the FFT thread, initialize spectrograph history with fixed capacity
    let spectrograph_history = Arc::new(Mutex::new(VecDeque::<SpectrographSlice>::with_capacity(MAX_SPECTROGRAPH_HISTORY)));

    // Pass the initialized history to the FFT thread
    let spectrograph_history_fft = Arc::clone(&spectrograph_history);

    // Pass shared_partials to FFT thread
    let shared_partials_clone = shared_partials.clone();
    let start_time = Arc::new(Instant::now());
    let start_time_fft = Arc::clone(&start_time);
    let fft_thread = std::thread::spawn({
        let audio_buffer = Arc::clone(&audio_buffer);
        let fft_config = Arc::clone(&fft_config);
        let spectrum_app = Arc::clone(&spectrum_app);
        let selected_channels = selected_channels.clone();
        let shutdown_flag_fft = Arc::clone(&shutdown_flag);
        let shared_partials_clone = shared_partials_clone.clone();
        let spectrograph_history = spectrograph_history_fft;
        let start_time = start_time_fft;

        move || {
            fft_analysis::start_fft_processing(
                audio_buffer,
                fft_config,
                spectrum_app,
                selected_channels,
                selected_sample_rate as u32,
                shutdown_flag_fft,
                shared_partials_clone,
                Some(spectrograph_history),
                Some(start_time),
            );
        }
    });

    // Start audio thread
    info!("Starting audio sampling thread...");
    let fft_config_clone = Arc::clone(&fft_config);
    let audio_thread = std::thread::spawn(move || {
        start_sampling_thread(
            running,
            audio_buffer_clone,
            selected_channels_clone,
            selected_sample_rate,
            buffer_size_clone,
            selected_device_index,
            shutdown_flag_audio,
            stream_ready_audio,
            fft_config_clone,
        );
        shutdown_complete_audio.store(true, Ordering::SeqCst);
    });

    // Start FFT processing thread only after stream is ready
    info!("Waiting for audio stream to initialize...");
    let stream_ready_fft = Arc::clone(&stream_ready);
    while !stream_ready_fft.load(Ordering::SeqCst) {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    // Start resynthesis thread
    info!("Starting resynthesis...");
    let resynth_thread = std::thread::spawn({
        let spectrum_app = Arc::clone(&spectrum_app);
        let resynth_config = Arc::clone(&resynth_config);
        let shutdown_flag = Arc::clone(&shutdown_flag);
        move || {
            start_resynth_thread(
                spectrum_app,
                resynth_config,
                selected_output_device,
                selected_sample_rate,
                shutdown_flag,
            );
        }
    });

    // Start GUI
    info!("Starting GUI...");
    let app = plot::MyApp::new(
        spectrum_app.clone(),
        fft_config.clone(),
        buffer_size.clone(),
        audio_buffer.clone(),
        resynth_config.clone(),
        shutdown_flag.clone(),
        spectrograph_history.clone(),
        start_time.clone(),
    );

    let native_options = NativeOptions {
        initial_window_size: Some(egui::vec2(1024.0, 440.0)),
        vsync: true,
        ..Default::default()
    };

    if let Err(e) = eframe::run_native(
        "Real-Time Spectrum Analyzer",
        native_options,
        Box::new(|_cc| Box::new(app)),
    ) {
        error!("GUI error: {}", e);
    }

    // Set shutdown flag to stop processing threads
    info!("Setting shutdown flag...");
    shutdown_flag.store(true, Ordering::SeqCst);

    // Wait for threads to finish with timeout
    let timeout = Duration::from_secs(5);
    let start = std::time::Instant::now();

    while start.elapsed() < timeout {
        if shutdown_complete.load(Ordering::SeqCst) {
            break;
        }
        thread::sleep(Duration::from_millis(100));
    }

    // Clean up PortAudio
    if let Ok(pa) = pa::PortAudio::new() {
        if let Err(e) = pa.terminate() {
            warn!("Error terminating PortAudio: {}", e);
        }
    }

    info!("Application shutdown complete.");
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
        pa::InputStreamSettings::new(params, 48000.0f64, 512),
        |_args| pa::Continue,
    ) {
        stream.close().is_ok()
    } else {
        false
    }
}

#[allow(dead_code)]
fn test_audio_input(pa: &pa::PortAudio, device_index: pa::DeviceIndex, channels: i32) -> Result<bool> {
    info!("Testing audio input for device...");
    
    let latency = pa.device_info(device_index)?.default_low_input_latency;
    let input_params = pa::StreamParameters::new(device_index, channels, true, latency);
    
    // Create a test buffer
    let mut test_buffer = vec![0.0f32; 1024 * channels as usize];
    
    // Create a blocking stream for testing
    let mut stream = pa.open_blocking_stream(
        pa::InputStreamSettings::new(input_params, 44100.0f64, 1024)
    )?;
    
    stream.start()?;
    info!("Reading test audio data...");
    
    // Try to read some data
    match stream.read(1024) {
        Ok(data) => {
            test_buffer.copy_from_slice(data);
            let non_zero = test_buffer.iter().filter(|&&x| x != 0.0).count();
            info!("Test read - Buffer size: {}, Non-zero samples: {}", test_buffer.len(), non_zero);
            if non_zero > 0 {
                info!("First few non-zero samples: {:?}", 
                    test_buffer.iter()
                        .filter(|&&x| x != 0.0)
                        .take(5)
                        .collect::<Vec<_>>());
            }
            stream.stop()?;
            Ok(non_zero > 0)
        },
        Err(e) => {
            stream.stop()?;
            Err(anyhow!("Failed to read audio data: {}", e))
        }
    }
}
