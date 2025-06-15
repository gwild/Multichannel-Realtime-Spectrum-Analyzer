mod audio_stream;
mod fft_analysis;
mod plot;
mod display;
mod resynth;
mod get_results;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input device index
    #[arg(short = 'i', long)]
    input_device: Option<usize>,

    /// Output device index (currently unused)
    #[arg(short = 'o', long)]
    output_device: Option<usize>,

    /// Sample rate in Hz (e.g. 44100, 48000)
    #[arg(short = 'r', long)]
    sample_rate: Option<f64>,

    /// Input channels to use, comma-separated list such as "0,1"
    #[arg(short = 'c', long)]
    channels: Option<String>,

    /// Number of partials to detect per channel
    #[arg(short = 'p', long)]
    num_partials: Option<usize>,

    /// Internal flag used when the app relaunches itself in a new terminal. Not meant for users.
    #[arg(long = "launched-by-python", hide = true, default_value_t = false)]
    launched_by_python: bool,
}

pub const MIN_FREQ: f64 = 20.0;
pub const MAX_FREQ: f64 = 20000.0;
pub const MIN_BUFFER_SIZE: usize = 512;
pub const MAX_BUFFER_SIZE: usize = 65536;
pub const DEFAULT_BUFFER_SIZE: usize = 8192;
pub const DEFAULT_FRAMES_PER_BUFFER: u32 = 2048;
pub const FRAME_SIZES: [u32; 7] = [64, 128, 256, 512, 1024, 2048, 4096];

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
use log::{info, error, warn, debug, LevelFilter};
use fern::Dispatch;
use env_logger;
use fft_analysis::{FFTConfig, MAX_SPECTROGRAPH_HISTORY};
use crate::resynth::{ResynthConfig, start_resynth_thread};
use std::thread;
use std::time::Duration;
use std::collections::VecDeque;
use crate::plot::SpectrographSlice;
use std::time::Instant;
use std::fs::OpenOptions;
use chrono;
use tokio::sync::broadcast;
use memmap2::MmapMut;
use crate::get_results::GuiParameter;
use std::sync::mpsc;

#[derive(Clone)]
pub struct SharedMemory {
    pub path: String,
}

// Add a new constant to replace hardcoded 12 throughout code
pub const DEFAULT_NUM_PARTIALS: usize = 12;

// Define type alias (should match fft_analysis)
type PartialsData = Vec<Vec<(f32, f32)>>;

// Async function to handle shared memory updates
async fn shared_memory_updater_loop(
    mut partials_rx: broadcast::Receiver<PartialsData>,
    shared_memory_path: String,
    shutdown_flag: Arc<AtomicBool>,
) {
    debug!(target: "shared_memory", "Starting shared memory update loop for path: {}", shared_memory_path);

    while !shutdown_flag.load(Ordering::Relaxed) {
        match partials_rx.recv().await {
            Ok(linear_partials) => {
                // Serialize LINEAR partials (unitless magnitudes) to bytes for Python
                // The received linear_partials already contains (frequency, linear_magnitude)
                // where linear_magnitude is 0.0 for padded/non-existent partials.
                let mut bytes_to_write = Vec::<u8>::new();
                for channel_data in linear_partials { // Iterate directly over the received linear_partials
                    for (freq, linear_mag) in channel_data { // 'linear_mag' is the original linear magnitude
                        bytes_to_write.extend_from_slice(&freq.to_ne_bytes());
                        bytes_to_write.extend_from_slice(&linear_mag.to_ne_bytes()); // Write the linear magnitude directly
                    }
                }

                // Write to shared memory file
                match OpenOptions::new().read(true).write(true).open(&shared_memory_path) {
                    Ok(file) => {
                        match unsafe { MmapMut::map_mut(&file) } {
                            Ok(mut mmap) => {
                                let len = bytes_to_write.len().min(mmap.len());
                                mmap[..len].copy_from_slice(&bytes_to_write[..len]);
                                // Optional: Write a sentinel/length if protocol requires
                                // mmap.flush(); // Ensure changes are written (usually optional)
                            }
                            Err(e) => {
                                error!(target: "shared_memory", "Failed to memory map file {}: {}", shared_memory_path, e);
                            }
                        }
                    }
                    Err(e) => {
                        error!(target: "shared_memory", "Failed to open shared memory file {}: {}", shared_memory_path, e);
                    }
                }
            }
            Err(broadcast::error::RecvError::Lagged(n)) => {
                warn!(target: "shared_memory", "Shared memory partials receiver lagged by {} messages.", n);
            }
            Err(broadcast::error::RecvError::Closed) => {
                info!(target: "shared_memory", "Partials broadcast channel closed for shared memory.");
                break; // Exit loop if channel is closed
            }
        }
    }
    info!(target: "shared_memory", "Shared memory update loop shutting down.");
}

// Make main async and return Result
#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Parse command-line arguments once
    let args = Args::parse();

    // Check if we're already running in the child terminal (flag injected by relaunch)
    let is_launched_by_python = args.launched_by_python;
    if !is_launched_by_python {
        // Relaunch in a new terminal
        println!("Relaunching in a new terminal for consistent environment...");
        let current_exe = std::env::current_exe().expect("Failed to get current executable path");
        let current_dir = std::env::current_dir().expect("Failed to get current directory");

        // Build the command, preserving all original arguments
        let orig_args: Vec<String> = std::env::args().skip(1).collect();
        let mut cmd_str = format!("cd '{}' && {} --launched-by-python", 
            current_dir.display(), 
            current_exe.display()
        );
        
        // Add all original arguments
        for arg in orig_args {
            cmd_str.push_str(&format!(" '{}'", arg));
        }
        
        cmd_str.push_str(" 2>&1 | tee debug.txt; read -p 'Press enter to close'");

        let cmd = vec![
            "xterm".to_string(),
            "-hold".to_string(),
            "-e".to_string(),
            "bash".to_string(),
            "-c".to_string(),
            cmd_str,
        ];

        let mut child = std::process::Command::new(&cmd[0])
            .args(&cmd[1..])
            .spawn()
            .expect("Failed to launch new terminal");
        child.wait().expect("Failed to wait for child process");
        return Ok(());
    }

    // Initialize logging first, before any other operations
    let args_vec: Vec<String> = std::env::args().collect();
    
    if args_vec.contains(&"--debug".to_string()) {
        let mut dispatch = Dispatch::new()
            .format(|out, message, record| {
                out.finish(format_args!(
                    "[{}:{}] {} - {}",
                    record.level(),
                    record.target(),
                    chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f"),
                    message
                ))
            })
            .level(LevelFilter::Info);  // Default level

        let mut debug_enabled = false;
        for arg in args_vec.iter() {
            match arg.as_str() {
                "resynth" => {
                    dispatch = dispatch.level_for("audio_streaming::resynth", LevelFilter::Debug);
                    debug_enabled = true;
                    println!("Enabling debug for resynth module");
                },
                "fft" => {
                    dispatch = dispatch.level_for("audio_streaming::fft_analysis", LevelFilter::Debug);
                    debug_enabled = true;
                    println!("Enabling debug for fft module");
                },
                "audio" => {
                    dispatch = dispatch.level_for("audio_streaming::audio_stream", LevelFilter::Debug);
                    debug_enabled = true;
                    println!("Enabling debug for audio module");
                },
                "plot" => {
                    dispatch = dispatch.level_for("audio_streaming::plot", LevelFilter::Debug);
                    debug_enabled = true;
                    println!("Enabling debug for plot module");
                },
                "main" => {
                    dispatch = dispatch.level_for("audio_streaming", LevelFilter::Debug);
                    debug_enabled = true;
                    println!("Enabling debug for main module");
                },
                "all" => {
                    dispatch = dispatch.level(LevelFilter::Debug);
                    debug_enabled = true;
                    println!("Enabling debug for all modules");
                    break;
                },
                _ => {}
            }
        }

        if !debug_enabled && args_vec.contains(&"--debug".to_string()) { // Ensure this condition captures --debug alone
            println!("Debug flag present but no module specified. Defaulting to global debug. Available modules: resynth, fft, audio, plot, main, all");
            dispatch = dispatch.level(LevelFilter::Debug); // Default to global debug if only --debug is present
        }

        dispatch
            .chain(std::io::stdout()) // Ensure it also goes to xterm's stdout
            .chain(fern::log_file("debug_explicit.log").map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?)
            .apply()
            .map_err(|e| anyhow!("Logger apply error: {}", e))?;

        debug!("Logging initialized with args: {:?}", args_vec);
        info!("FERN LOGGER INITIALIZED AND ACTIVE in main.rs instance. Debug level: {}", log::max_level());
    } else {
        std::env::set_var("RUST_LOG", "error");
        env_logger::init();
    }

    // Check if GStreamer is running, start it if not
    check_and_start_gstreamer();

    if let Err(e) = run(&args).await {
        error!("Application error in main: {:?}", e); // Clarify source of error log
        std::process::exit(1);
    }
    Ok(())
}

// Function to check if GStreamer is running and start it if not
fn check_and_start_gstreamer() {
    let gstreamer_check = std::process::Command::new("pgrep")
        .arg("-f")
        .arg("gst-launch-1.0")
        .output();

    match gstreamer_check {
        Ok(output) if output.status.success() => {
            info!("GStreamer is already running.");
        },
        _ => {
            info!("GStreamer is not running, attempting to start it in a new terminal...");
            // Build absolute path to stream.sh located beside the binary
            let script_path = std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|d| d.join("stream.sh")))
                .unwrap_or_else(|| std::path::PathBuf::from("./stream.sh"));

            let start_result = std::process::Command::new("xterm")
                .arg("-e")
                .arg("bash")
                .arg("-c")
                .arg(format!("'{}; read -p \"Press enter to close\"'", script_path.display()))
                .spawn();

            match start_result {
                Ok(child) => {
                    info!("GStreamer stream started with PID: {} in a new terminal", child.id());
                    // Wait a bit to ensure it starts
                    std::thread::sleep(std::time::Duration::from_secs(2));
                },
                Err(e) => {
                    error!("Failed to start GStreamer stream in a new terminal: {}", e);
                }
            }
        }
    }
}

// Make run async
async fn run(args: &Args) -> Result<()> {
    info!("run() function entered."); // New log
    let pa = Arc::new(pa::PortAudio::new()?);
    info!("PortAudio initialized successfully in run()."); // New log

    let devices = pa.devices()?.collect::<Result<Vec<_>, _>>()?;
    info!("Initial device list collected in run(). Count: {}", devices.len()); // New log
    if devices.is_empty() {
        warn!("No devices found. Attempting to reset devices.");
        reset_audio_devices(&pa)?;
        info!("reset_audio_devices() called."); // New log
    }

    // Add a log after potentially resetting devices, before trying to list them again.
    info!("Device reset attempted if necessary. Proceeding to list devices.");

    let devices = pa.devices()?.collect::<Result<Vec<_>, _>>()?;
    info!("Final device list collected in run(). Count: {}", devices.len()); // New log
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

    // Device selection: use CLI arg if provided, otherwise prompt
    let selected_device_index = if let Some(idx) = args.input_device {
        if idx >= input_devices.len() {
            return Err(anyhow!(
                "Invalid device index {} provided via --input-device. Must be 0..{}",
                idx,
                input_devices.len() - 1
            ));
        }
        idx
    } else {
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
                "Invalid device index. Please choose a number between 0 and {}",
                input_devices.len() - 1
            ));
        }
        device_index
    };
    let selected_device_index = input_devices[selected_device_index];
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

    let selected_sample_rate = if let Some(rate_cli) = args.sample_rate {
        if !supported_sample_rates.contains(&rate_cli) {
            return Err(anyhow!("Sample rate {} is not supported by selected device", rate_cli));
        }
        rate_cli
    } else {
        println!("Supported sample rates:");
        for (i, rate) in supported_sample_rates.iter().enumerate() {
            println!("  [{}] - {} Hz", i, rate);
        }

        print!("Enter the index of the desired sample rate: ");
        io::stdout().flush()?;
        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input)?;
        let sample_rate_index = user_input
            .trim()
            .parse::<usize>()
            .map_err(|_| anyhow!("Invalid sample rate index"))?;

        if sample_rate_index >= supported_sample_rates.len() {
            return Err(anyhow!("Invalid sample rate index."));
        }
        supported_sample_rates[sample_rate_index]
    };
    info!("Selected sample rate: {} Hz", selected_sample_rate);

    let selected_channels: Vec<usize> = if let Some(ref ch_str) = args.channels {
        ch_str
            .split(',')
            .filter_map(|s| s.parse::<usize>().ok())
            .filter(|&ch| ch < selected_device_info.max_input_channels as usize)
            .collect()
    } else {
        println!(
            "Available channels: 0 to {}",
            selected_device_info.max_input_channels - 1
        );
        println!("Enter channels to use (comma-separated, e.g., 0,1): ");
        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input)?;
        user_input
            .trim()
            .split(',')
            .filter_map(|s| s.parse::<usize>().ok())
            .filter(|&ch| ch < selected_device_info.max_input_channels as usize)
            .collect()
    };

    if selected_channels.is_empty() {
        return Err(anyhow!("No valid channels selected."));
    }
    info!("Selected channels: {:?}", selected_channels);

    // Add prompt for number of partials here
    let num_partials = if let Some(p) = args.num_partials {
        p.max(1)
    } else {
        println!("Enter number of partials to detect per channel (default is {}): ", DEFAULT_NUM_PARTIALS);
        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input)?;
        if user_input.trim().is_empty() {
            DEFAULT_NUM_PARTIALS
        } else {
            user_input
                .trim()
                .parse::<usize>()
                .map_err(|_| anyhow!("Invalid number of partials"))?
                .max(1)
        }
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

    // Create the MPSC channel for GUI parameter updates
    let (gui_param_tx, gui_param_rx) = mpsc::channel::<GuiParameter>();

    // Create the MPSC channel for instant gain updates
    let (gain_update_tx, gain_update_rx) = mpsc::channel::<f32>();

    // Create the MPSC channel for SynthUpdate (from get_results to wavegen)
    // This channel is now managed by start_resynth_thread internally.
    // let (update_tx_for_wavegen, update_rx_for_wavegen) = mpsc::channel::<crate::resynth::SynthUpdate>();

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

    let output_device_index = if let Some(idx) = args.output_device {
        idx
    } else {
        print!("Enter the index of the desired output device: ");
        io::stdout().flush()?;
        let mut input_line = String::new();
        io::stdin().read_line(&mut input_line)?;
        input_line
            .trim()
            .parse::<usize>()
            .map_err(|_| anyhow!("Invalid device index"))?
    };

    if output_device_index >= output_devices.len() {
        return Err(anyhow!("Invalid output device index"));
    }

    let selected_output_device = output_devices[output_device_index];

    // After device selection but before starting threads:
    let control_path = "/dev/shm/audio_control";
    let mut control_file = std::fs::File::create(&control_path)?;
    writeln!(control_file, "{}\n{}\n{}", std::process::id(), selected_channels.len(), num_partials)?;

    // Initialize shared memory without mutex BEFORE threads start
    let shared_partials = {
        let shmem_name = "audio_peaks";
        let shared_memory_path = format!("/dev/shm/{}", shmem_name);
        let file = std::fs::File::create(&shared_memory_path)?;
        file.set_len(4 * 1024 * 1024)?;
        info!("Shared memory initialized at {} for internal use", shared_memory_path);
        Some(SharedMemory {
            path: shared_memory_path,
        })
    };

    // Before starting the FFT thread, initialize spectrograph history with fixed capacity
    let spectrograph_history = Arc::new(Mutex::new(VecDeque::<SpectrographSlice>::with_capacity(MAX_SPECTROGRAPH_HISTORY)));
    let spectrograph_history_fft = Arc::clone(&spectrograph_history);

    // Create the broadcast channel for partials
    let (partials_tx, partials_rx) = broadcast::channel::<PartialsData>(16); 
    // Clone sender for FFT thread
    let fft_partials_tx = partials_tx.clone();

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
        let partials_tx_clone = fft_partials_tx;

        move || {
            fft_analysis::start_fft_processing(
                audio_buffer,
                fft_config,
                spectrum_app,
                selected_channels,
                selected_sample_rate as u32,
                shutdown_flag_fft,
                partials_tx_clone,
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
    let resynth_partials_rx_broadcast = partials_tx.subscribe();
    let resynth_thread = std::thread::spawn({
        let resynth_config_clone = Arc::clone(&resynth_config);
        let shutdown_flag_clone = Arc::clone(&shutdown_flag);
        let num_channels_val = selected_channels.len();
        let num_partials_val = num_partials;
        // Pass gui_param_rx to start_resynth_thread
        let gui_param_rx_for_resynth = gui_param_rx;
        let gain_update_rx_for_resynth = gain_update_rx;

        move || {
            start_resynth_thread(
                resynth_config_clone,
                selected_output_device,
                selected_sample_rate,
                shutdown_flag_clone,
                resynth_partials_rx_broadcast, // This is broadcast::Receiver<PartialsData>
                num_channels_val,
                num_partials_val,
                gui_param_rx_for_resynth, // Pass the GuiParameter receiver here
                gain_update_rx_for_resynth, // Pass the gain update receiver here
            );
        }
    });

    // Start GUI
    info!("Starting GUI...");
    let plot_partials_rx = partials_tx.subscribe();
    let app = plot::MyApp::new(
        spectrum_app.clone(),
        fft_config.clone(),
        buffer_size.clone(),
        audio_buffer.clone(),
        resynth_config.clone(),
        shutdown_flag.clone(),
        spectrograph_history.clone(),
        start_time.clone(),
        selected_sample_rate,
        plot_partials_rx,
        gui_param_tx, // Pass the sender to MyApp
        gain_update_tx, // Pass the gain update sender to MyApp
    );

    // Spawn SharedMemory update thread
    if let Some(shared_mem_info) = shared_partials {
        let shared_memory_partials_rx = partials_tx.subscribe();
        let sm_shutdown_flag = shutdown_flag.clone();
        tokio::spawn(async move {
            shared_memory_updater_loop(shared_memory_partials_rx, shared_mem_info.path, sm_shutdown_flag).await;
        });
    } else {
        warn!("SharedMemory struct not initialized, skipping shared memory update thread.");
    }

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

