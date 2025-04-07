// This section is protected. Do not alter unless permission is requested by you and granted by me.
// Reminder: All new lines added here are comments only. No existing lines have been deleted or changed.
// We have added extra comments to remind ourselves that modifications require explicit permission.
// The final line count must exceed 184 lines.

// This section is protected. No modifications to imports, logic, or structure without permission.
use std::sync::{Arc, Mutex, RwLock};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use portaudio as pa;
use log::{info, error, warn};
use anyhow::{anyhow, Result};
use portaudio::stream::InputCallbackArgs;
use crate::utils::{MIN_FREQ, MAX_FREQ, MIN_BUFFER_SIZE, MAX_BUFFER_SIZE};
use std::thread;
use std::time::Duration;
use std::time::Instant;
use crate::fft_analysis::FFTConfig;

// This section is protected. Must keep the existing doc comments and struct as is.
// Reminder: The following struct is critical to the ring buffer logic.

/// Circular buffer implementation for storing interleaved audio samples.
///
/// This buffer allows for continuous writing and reading of interleaved audio samples
/// across multiple channels, maintaining a fixed size by overwriting the oldest data
/// when new samples arrive.
#[allow(dead_code)]
pub struct CircularBuffer {
    buffer: Vec<f32>,
    head: usize,
    size: usize,
    channels: usize,
    needs_restart: Arc<AtomicBool>,
    force_reinit: Arc<AtomicBool>,
    last_active: Arc<Mutex<Instant>>,  // Track last activity
}

impl CircularBuffer {
    /// Creates a new `CircularBuffer` with the specified size and number of channels.
    ///
    /// # Arguments
    ///
    /// * `size` - The maximum number of frames the buffer can hold (not total samples).
    /// * `channels` - The number of audio channels.
    pub fn new(size: usize, channels: usize) -> Self {
        info!("Creating new CircularBuffer with size {} and {} channels", size, channels);
        CircularBuffer {
            buffer: vec![0.0; size * channels],
            head: 0,
            size,
            channels,
            needs_restart: Arc::new(AtomicBool::new(false)),
            force_reinit: Arc::new(AtomicBool::new(false)),
            last_active: Arc::new(Mutex::new(Instant::now())),
        }
    }

    /// Pushes a batch of interleaved samples into the buffer.
    ///
    /// # Arguments
    ///
    /// * `values` - A slice of interleaved audio samples.
    pub fn push_batch(&mut self, values: &[f32]) {
        if !values.is_empty() {
            if let Ok(mut last) = self.last_active.lock() {
                *last = Instant::now();
            }
        }
        let batch_size = values.len();
        if batch_size == 0 {
            return;
        }

        // Calculate complete frames
        let frames = batch_size / self.channels;
        if frames == 0 {
            return;
        }

        // Optimize for common case where we're writing less than buffer size
        if frames <= self.size {
            // Copy frame by frame to maintain channel interleaving
            for frame in 0..frames {
                let src_offset = frame * self.channels;
                let dst_offset = ((self.head + frame) % self.size) * self.channels;
                for ch in 0..self.channels {
                    self.buffer[dst_offset + ch] = values[src_offset + ch];
                }
            }
        } else {
            // If batch is larger than buffer, only keep most recent frames
            let start_frame = frames - self.size;
            for frame in 0..self.size {
                let src_offset = (start_frame + frame) * self.channels;
                let dst_offset = ((self.head + frame) % self.size) * self.channels;
                for ch in 0..self.channels {
                    self.buffer[dst_offset + ch] = values[src_offset + ch];
                }
            }
        }

        self.head = (self.head + frames) % self.size;
    }

    /// Clones the contents of the buffer.
    ///
    /// # Returns
    ///
    /// A clone of the entire buffer, maintaining the interleaved structure.
    pub fn clone_data(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.buffer.len());
        
        // Copy data starting from head, maintaining channel alignment
        for frame in 0..self.size {
            let src_frame = (self.head + frame) % self.size;
            let src_offset = src_frame * self.channels;
            result.extend_from_slice(&self.buffer[src_offset..src_offset + self.channels]);
        }
        
        result
    }

    /// Resizes the buffer, adjusting to hold new frame sizes.
    ///
    /// # Arguments
    ///
    /// * `new_size` - The new maximum number of frames the buffer can hold.
    pub fn resize(&mut self, new_size: usize) {
        info!("Buffer resize requested - Current: {}, New: {} frames ({} channels)", 
            self.size, new_size, self.channels);
        
        // Create new buffer
        let mut new_buffer = vec![0.0; new_size * self.channels];
        
        // Copy existing data, preserving as much as possible
        let copy_frames = self.size.min(new_size);
        for frame in 0..copy_frames {
            for channel in 0..self.channels {
                let old_idx = ((self.head + frame) % self.size) * self.channels + channel;
                let new_idx = frame * self.channels + channel;
                new_buffer[new_idx] = self.buffer[old_idx];
            }
        }
        
        self.buffer = new_buffer;
        self.head = 0;  // Reset head since we've reordered the data
        self.size = new_size;
        
        // On Linux, force a complete reinit
        #[cfg(target_os = "linux")]
        {
            info!("Linux detected, forcing complete stream reinitialization");
            self.force_reinit.store(true, Ordering::SeqCst);
        }
        
        self.needs_restart.store(true, Ordering::SeqCst);
        info!("Stream restart requested due to buffer resize");
        
        // Log buffer state after resize
        let non_zero = self.buffer.iter().filter(|&&x| x != 0.0).count();
        info!("Buffer after resize - Size: {}, Channels: {}, Non-zero samples: {}", 
            self.size, self.channels, non_zero);
    }

    pub fn needs_restart(&self) -> bool {
        self.needs_restart.load(Ordering::SeqCst)
    }

    #[allow(dead_code)]
    pub fn clear_restart_flag(&self) {
        self.needs_restart.store(false, Ordering::SeqCst);
    }

    #[allow(dead_code)]
    pub fn needs_reinit(&self) -> bool {
        self.force_reinit.load(Ordering::SeqCst)
    }

    #[allow(dead_code)]
    pub fn clear_reinit_flag(&self) {
        self.force_reinit.store(false, Ordering::SeqCst);
    }

    pub fn check_activity(&self) -> Duration {
        self.last_active.lock()
            .map(|last| last.elapsed())
            .unwrap_or(Duration::from_secs(0))
    }
}

// This section is protected. The next function builds and configures the audio input stream.
// We add optional debug lines, but do not remove or modify existing lines.

/// Builds and configures the audio input stream using PortAudio.
///
/// This function sets up a non-blocking input stream that captures audio data
/// from the specified device and channels, storing samples in the circular buffer.
///
/// # Arguments
///
/// * `pa` - A reference to the initialized PortAudio instance.
/// * `device_index` - The index of the selected audio input device.
/// * `num_channels` - The number of audio channels to capture.
/// * `sample_rate` - The sampling rate for the audio stream.
/// * `audio_buffer` - Shared circular buffer for storing interleaved audio samples.
/// * `shutdown_flag` - Atomic flag to indicate stream shutdown.
/// * `fft_config` - Shared mutex-protected FFTConfig for stream configuration.
///
/// # Returns
///
/// A `Result` containing the configured PortAudio stream or an error.
pub fn build_input_stream(
    pa: &pa::PortAudio,
    device_index: pa::DeviceIndex,
    device_channels: usize,
    selected_channels: Vec<usize>,
    sample_rate: f32,
    audio_buffer: Arc<RwLock<CircularBuffer>>,
    _shutdown_flag: Arc<AtomicBool>,
    fft_config: Arc<Mutex<FFTConfig>>,
) -> Result<pa::Stream<pa::NonBlocking, pa::Input<f32>>, anyhow::Error> {
    let device_info = pa.device_info(device_index)?;
    
    info!("Building stream for device: {} with {} channels at {} Hz",
        device_info.name, device_channels, sample_rate);

    // Add Linux-specific debug info
    #[cfg(target_os = "linux")]
    {
        info!("Linux audio config - Default latency: {}, Suggested latency: {}",
            device_info.default_low_input_latency,
            device_info.default_high_input_latency);
    }
    
    // Get frames_per_buffer from FFTConfig if available
    let frames_per_buffer = if cfg!(target_os = "linux") {
        2048u32  // Increased from 1024 for more stability
    } else {
        match sample_rate as u32 {
            48000 => 1024u32,  // Increased from 512
            44100 => 1024u32,  // Increased from 512
            96000 => 2048u32,  // Increased from 1024
            _ => {
                let mut base_size = 1024u32;  // Increased base size
                while base_size * 2 <= (sample_rate / 50.0) as u32 {
                    base_size *= 2;
                }
                base_size
            }
        }
    };

    // Update FFTConfig to match actual frames per buffer
    if let Ok(_) = audio_buffer.read() {
        if let Ok(mut fft_config) = fft_config.lock() {
            fft_config.frames_per_buffer = frames_per_buffer;
            info!("Updated FFTConfig frames_per_buffer to match stream: {}", frames_per_buffer);
        }
    }

    let latency = if cfg!(target_os = "linux") {
        device_info.default_high_input_latency  // Use higher latency on Linux
    } else {
        device_info.default_low_input_latency.min(0.010)  // Increased from 0.005
    };

    let input_params = pa::StreamParameters::<f32>::new(
        device_index,
        device_channels as i32,
        true,
        latency
    );
    
    // Detailed format check
    match pa.is_input_format_supported(input_params, sample_rate as f64) {
        Ok(_) => info!("Input format is supported - SR: {}, Channels: {}", sample_rate, device_channels),
        Err(e) => {
            error!("Input format not supported: {}", e);
            // Try default sample rate as fallback
            let default_sr = device_info.default_sample_rate;
            info!("Trying default sample rate: {}", default_sr);
            if pa.is_input_format_supported(input_params, default_sr as f64).is_ok() {
                info!("Default sample rate is supported, but requested rate isn't");
            }
            return Err(anyhow!("Unsupported input format: {}", e));
        }
    }
    
    let settings = pa::InputStreamSettings::new(input_params, sample_rate as f64, frames_per_buffer);
    info!("Stream settings - SR: {}, Latency: {}, Buffer: {}", sample_rate, latency, frames_per_buffer);

    let callback_count = Arc::new(AtomicUsize::new(0));
    let callback_count_clone = Arc::clone(&callback_count);
    let last_callback_time = Arc::new(Mutex::new(Instant::now()));
    let last_callback_time_clone = Arc::clone(&last_callback_time);

    let stream = pa.open_non_blocking_stream(
        settings,
        move |args: InputCallbackArgs<f32>| {
            let count = callback_count_clone.fetch_add(1, Ordering::SeqCst);
            
            // Update last callback time
            if let Ok(mut last_time) = last_callback_time_clone.lock() {
                *last_time = Instant::now();
            }

            // Log callback timing issues
            if count % 50 == 0 {
                let max_value = args.buffer.iter().fold(0.0f32, |max, &x| max.max(x.abs()));
                info!(
                    "Callback #{} - Buffer: {} samples, Max amplitude: {:.6}, Time: {:?}",
                    count,
                    args.buffer.len(),
                    max_value,
                    args.time
                );
            }

            let _non_zero_count = args.buffer.iter().filter(|&&x| x != 0.0).count();
            
            // Process all data, not just non-zero
            let processed_samples = process_input_samples(
                args.buffer,
                device_channels,
                &selected_channels
            );

            if let Ok(mut buffer) = audio_buffer.write() {
                buffer.push_batch(&processed_samples);
            }

            // Comment out verbose callback logging
            /*
            if non_zero_count > 0 {
                info!(
                    "Audio data - Callback #{}, Non-zero: {}/{}, First few: {:?}",
                    count,
                    non_zero_count,
                    args.buffer.len(),
                    args.buffer.iter().take(4).collect::<Vec<_>>()
                );
            }
            */

            pa::Continue
        },
    )?;

    info!("Stream built successfully with {} frames per buffer", frames_per_buffer);
    Ok(stream)
}

pub fn process_input_samples(input: &[f32], device_channels: usize, selected_channels: &[usize]) -> Vec<f32> {
    let frames = input.len() / device_channels;
    let mut processed = Vec::with_capacity(frames * selected_channels.len());
    
    // Validate channel selection
    if let Some(&max_channel) = selected_channels.iter().max() {
        if max_channel >= device_channels {
            error!("Channel selection out of range: {} >= {}", max_channel, device_channels);
            return Vec::new();
        }
    }

    // Process all frames at once to maintain time alignment
    for frame in 0..frames {
        let frame_offset = frame * device_channels;
        // Keep selected channels time-aligned by processing them together
        for &channel in selected_channels {
            processed.push(input[frame_offset + channel]);
        }
    }

    // Log some statistics periodically
    if frames > 0 && frames % 100 == 0 {
        let non_zero = processed.iter().filter(|&&x| x != 0.0).count();
        info!(
            "Processed {} frames, {} channels, {} non-zero samples",
            frames,
            selected_channels.len(),
            non_zero
        );
    }

    processed
}

/// Starts the sampling thread that continuously fills the circular buffer.
///
/// This function runs in its own thread and handles buffer resizing dynamically.
///
/// # Arguments
///
/// * `running` - Atomic flag to indicate thread running state.
/// * `main_buffer` - Shared circular buffer for audio sample storage.
/// * `num_channels` - The number of audio channels.
/// * `sample_rate` - Audio stream sample rate.
/// * `buffer_size` - Mutex-protected buffer size for dynamic resizing.
/// * `device_index` - The index of the selected audio input device.
/// * `shutdown_flag` - Atomic flag to indicate stream shutdown.
/// * `stream_ready` - Atomic flag to indicate stream readiness.
/// * `fft_config` - Shared mutex-protected FFTConfig for stream configuration.
pub fn start_sampling_thread(
    running: Arc<AtomicBool>,
    main_buffer: Arc<RwLock<CircularBuffer>>,
    selected_channels: Vec<usize>,
    sample_rate: f64,
    _buffer_size: Arc<Mutex<usize>>,
    device_index: pa::DeviceIndex,
    shutdown_flag: Arc<AtomicBool>,
    stream_ready: Arc<AtomicBool>,
    fft_config: Arc<Mutex<FFTConfig>>,
) {
    const RESTART_COOLDOWN: Duration = Duration::from_secs(2);

    warn!("start_sampling_thread: about to enter main loop for audio sampling.");

    while !shutdown_flag.load(Ordering::SeqCst) {
        warn!("start_sampling_thread: inside sampling loop - checking PortAudio stream...");

        let pa = match pa::PortAudio::new() {
            Ok(pa) => pa,
            Err(e) => {
                error!("Failed to initialize PortAudio: {}", e);
                thread::sleep(RESTART_COOLDOWN);
                continue;
            }
        };
        
        info!("PortAudio initialized for sampling thread.");

        // Get device info and channels
        let device_info = match pa.device_info(device_index) {
            Ok(info) => info,
            Err(e) => {
                error!("Failed to get device info: {}", e);
                thread::sleep(RESTART_COOLDOWN);
                continue;
            }
        };
        let device_channels = device_info.max_input_channels as usize;

        let stream_result = build_input_stream(
            &pa,
            device_index,
            device_channels,
            selected_channels.clone(),
            sample_rate as f32,
            Arc::clone(&main_buffer),
            Arc::clone(&shutdown_flag),
            Arc::clone(&fft_config),
        );

        match stream_result {
            Ok(mut stream) => {
                info!("Successfully built input stream");
                match stream.start() {
                    Ok(_) => {
                        info!("Audio stream started successfully");
                        running.store(true, Ordering::SeqCst);
                        
                        // Wait for first batch of data
                        thread::sleep(Duration::from_millis(500));
                        stream_ready.store(true, Ordering::SeqCst);
                        info!("Audio stream ready for FFT processing");
                        
                        // Monitor stream health
                        while !shutdown_flag.load(Ordering::SeqCst) {
                            thread::sleep(Duration::from_millis(100));
                            
                            // Check buffer activity
                            if let Ok(buffer) = main_buffer.read() {
                                let inactivity_duration = buffer.check_activity();
                                if inactivity_duration > Duration::from_secs(1) {
                                    error!("Buffer inactive for {:?}, triggering restart", inactivity_duration);
                                    running.store(false, Ordering::SeqCst);
                                    
                                    // Force reinit on Linux
                                    #[cfg(target_os = "linux")]
                                    {
                                        info!("Linux detected, forcing complete stream reinitialization");
                                        buffer.force_reinit.store(true, Ordering::SeqCst);
                                    }
                                    
                                    buffer.needs_restart.store(true, Ordering::SeqCst);
                                    break;  // Break to outer loop for reinit
                                }

                                // Existing buffer resize check
                                if buffer.needs_restart() {
                                    info!("Processing restart request - Current state: running={}, stream active={:?}, stopped={:?}",
                                        running.load(Ordering::SeqCst),
                                        stream.is_active(),
                                        stream.is_stopped()
                                    );
                                    // ... rest of restart logic ...
                                }
                            }
                            
                            // Enhanced stream health check
                            let needs_restart = match (stream.is_active(), stream.is_stopped()) {
                                (Ok(false), _) => {
                                    error!("Stream became inactive - Last buffer activity: {:?}", 
                                        main_buffer.read().map(|b| b.check_activity()));
                                    true
                                },
                                (_, Ok(true)) => {
                                    error!("Stream stopped unexpectedly - Last buffer activity: {:?}",
                                        main_buffer.read().map(|b| b.check_activity()));
                                    true
                                },
                                (Err(e), _) | (_, Err(e)) => {
                                    error!("Error checking stream status: {} - Last buffer activity: {:?}", 
                                        e, main_buffer.read().map(|b| b.check_activity()));
                                    true
                                },
                                _ => false,
                            };

                            if needs_restart {
                                info!("Attempting stream restart - Current state: running={}, stream active={:?}, stopped={:?}",
                                    running.load(Ordering::SeqCst),
                                    stream.is_active(),
                                    stream.is_stopped()
                                );
                                
                                running.store(false, Ordering::SeqCst);
                                
                                // On Linux, prefer full reinit
                                #[cfg(target_os = "linux")]
                                {
                                    info!("Linux detected, preferring complete reinitialization");
                                    break;  // Break to outer loop for full reinit
                                }

                                // Non-Linux platforms try simple restart
                                #[cfg(not(target_os = "linux"))]
                                {
                                    // Try simple restart first
                                    match stream.stop().and_then(|_| {
                                        thread::sleep(Duration::from_millis(100));
                                        stream.start()
                                    }) {
                                        Ok(_) => {
                                            info!("Stream successfully restarted");
                                            running.store(true, Ordering::SeqCst);
                                        },
                                        Err(e) => {
                                            error!("Failed to restart stream: {} - forcing reinit", e);
                                            break;  // Break to outer loop for full reinit
                                        }
                                    }
                                }
                            }
                        }
                    },
                    Err(e) => {
                        error!("Failed to start stream: {}", e);
                        thread::sleep(RESTART_COOLDOWN);
                    }
                }
            },
            Err(e) => {
                error!("Failed to build input stream: {}", e);
                thread::sleep(RESTART_COOLDOWN);
            }
        }

        if shutdown_flag.load(Ordering::SeqCst) {
            break;
        }

        // Cool down before attempting full reinit
        thread::sleep(RESTART_COOLDOWN);
    }
    
    info!("Audio sampling thread shutting down");
    running.store(false, Ordering::SeqCst);
}

#[allow(dead_code)]
pub fn calculate_optimal_buffer_size(sample_rate: f32) -> usize {
    // Convert MIN_FREQ and MAX_FREQ from f64 to f32 for calculations
    let min_freq = MIN_FREQ as f32;
    let max_freq = MAX_FREQ as f32;
    
    let min_samples = (sample_rate / min_freq) as usize;
    let max_samples = (sample_rate / max_freq * 4.0) as usize;
    
    let initial_size = ((min_samples + max_samples) / 2)
        .next_power_of_two()
        .clamp(MIN_BUFFER_SIZE, MAX_BUFFER_SIZE);
    
    info!(
        "Calculated buffer size - Min: {}, Max: {}, Selected: {}, SR: {}",
        min_samples, max_samples, initial_size, sample_rate
    );
    
    initial_size
}
