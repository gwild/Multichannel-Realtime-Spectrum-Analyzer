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

// This section is protected. Must keep the existing doc comments and struct as is.
// Reminder: The following struct is critical to the ring buffer logic.

/// Circular buffer implementation for storing interleaved audio samples.
///
/// This buffer allows for continuous writing and reading of interleaved audio samples
/// across multiple channels, maintaining a fixed size by overwriting the oldest data
/// when new samples arrive.
pub struct CircularBuffer {
    buffer: Vec<f32>,
    head: usize,
    size: usize,
    channels: usize,
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
        }
    }

    /// Pushes a batch of interleaved samples into the buffer.
    ///
    /// # Arguments
    ///
    /// * `values` - A slice of interleaved audio samples.
    pub fn push_batch(&mut self, values: &[f32]) {
        let batch_size = values.len();
        info!("Pushing batch of {} samples to buffer (capacity: {})", batch_size, self.buffer.len());
        
        if batch_size == 0 {
            warn!("Attempted to push empty batch to buffer");
            return;
        }

        // Log first few values being pushed
        info!("First few values in batch: {:?}", &values[..values.len().min(5)]);
        
        // Calculate how many complete frames we're pushing
        let frames = batch_size / self.channels;
        if frames == 0 {
            warn!("Batch size {} is smaller than number of channels {}", batch_size, self.channels);
            return;
        }

        let old_head = self.head;
        let mut any_non_zero = false;

        // Push frame by frame to maintain channel interleaving
        for frame in 0..frames {
            for channel in 0..self.channels {
                let buffer_index = ((self.head + frame) % self.size) * self.channels + channel;
                let value_index = frame * self.channels + channel;
                if value_index < batch_size {
                    let value = values[value_index];
                    self.buffer[buffer_index] = value;
                    if value != 0.0 {
                        any_non_zero = true;
                    }
                }
            }
        }
        
        // Update head position by number of complete frames
        self.head = (self.head + frames) % self.size;
        
        // Log detailed buffer state
        let non_zero = self.buffer.iter().filter(|&&x| x != 0.0).count();
        info!(
            "Buffer update - Old head: {}, New head: {}, Frames written: {}, Any non-zero: {}, Total non-zero: {}", 
            old_head, self.head, frames, any_non_zero, non_zero
        );
        
        if non_zero > 0 {
            info!(
                "Buffer sample check - First few: {:?}, Around head: {:?}", 
                self.buffer.iter().take(6).collect::<Vec<_>>(),
                self.buffer.iter()
                    .skip(self.head * self.channels)
                    .take(6)
                    .collect::<Vec<_>>()
            );
        }
    }

    /// Clones the contents of the buffer.
    ///
    /// # Returns
    ///
    /// A clone of the entire buffer, maintaining the interleaved structure.
    pub fn clone_data(&self) -> Vec<f32> {
        // Reorder the data to start from head
        let mut result = Vec::with_capacity(self.buffer.len());
        
        // Copy from head to end
        for frame in 0..self.size {
            let frame_index = (self.head + frame) % self.size;
            for channel in 0..self.channels {
                let index = frame_index * self.channels + channel;
                result.push(self.buffer[index]);
            }
        }
        
        result
    }

    /// Resizes the buffer, adjusting to hold new frame sizes.
    ///
    /// # Arguments
    ///
    /// * `new_size` - The new maximum number of frames the buffer can hold.
    pub fn resize(&mut self, new_size: usize) {
        info!("Resizing buffer from {} to {} frames ({} channels)", 
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
        
        // Log buffer state after resize
        let non_zero = self.buffer.iter().filter(|&&x| x != 0.0).count();
        info!("Buffer after resize - Size: {}, Channels: {}, Non-zero samples: {}", 
            self.size, self.channels, non_zero);
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
///
/// # Returns
///
/// A `Result` containing the configured PortAudio stream or an error.
pub fn build_input_stream(
    pa: &pa::PortAudio,
    device_index: pa::DeviceIndex,
    device_channels: usize,
    selected_channels: Vec<usize>,
    sample_rate: f64,
    audio_buffer: Arc<RwLock<CircularBuffer>>,
    shutdown_flag: Arc<AtomicBool>,
) -> Result<pa::Stream<pa::NonBlocking, pa::Input<f32>>, anyhow::Error> {
    let device_info = pa.device_info(device_index)?;
    
    info!(
        "Device details - Name: {}, Max channels: {}, Default SR: {}, Default latency: {}",
        device_info.name,
        device_info.max_input_channels,
        device_info.default_sample_rate,
        device_info.default_low_input_latency
    );
    
    // Use device's default latency and a moderate buffer size
    let latency = device_info.default_low_input_latency;
    let frames_per_buffer = 512; // Moderate buffer size
    
    let input_params = pa::StreamParameters::<f32>::new(
        device_index,
        device_channels as i32,
        true,
        latency
    );
    
    // Detailed format check
    match pa.is_input_format_supported(input_params, sample_rate) {
        Ok(_) => info!("Input format is supported - SR: {}, Channels: {}", sample_rate, device_channels),
        Err(e) => {
            error!("Input format not supported: {}", e);
            // Try default sample rate as fallback
            let default_sr = device_info.default_sample_rate;
            info!("Trying default sample rate: {}", default_sr);
            if pa.is_input_format_supported(input_params, default_sr).is_ok() {
                info!("Default sample rate is supported, but requested rate isn't");
            }
            return Err(anyhow!("Unsupported input format: {}", e));
        }
    }
    
    let settings = pa::InputStreamSettings::new(input_params, sample_rate, frames_per_buffer);
    info!("Stream settings - SR: {}, Latency: {}, Buffer: {}", sample_rate, latency, frames_per_buffer);

    let callback_count = Arc::new(AtomicUsize::new(0));
    let callback_count_clone: Arc<AtomicUsize> = Arc::clone(&callback_count);

    let stream = pa.open_non_blocking_stream(
        settings,
        move |args: InputCallbackArgs<f32>| {
            let count = callback_count_clone.fetch_add(1, Ordering::SeqCst);
            
            if shutdown_flag.load(Ordering::Relaxed) {
                info!("Shutdown flag detected in callback after {} calls", count);
                return pa::Complete;
            }

            // Log every 50th callback to track activity
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

            let non_zero_count = args.buffer.iter().filter(|&&x| x != 0.0).count();
            
            // Process all data, not just non-zero
            let processed_samples = process_input_samples(
                args.buffer,
                device_channels,
                &selected_channels
            );

            if let Ok(mut buffer) = audio_buffer.write() {
                buffer.push_batch(&processed_samples);
            }

            if non_zero_count > 0 {
                info!(
                    "Audio data - Callback #{}, Non-zero: {}/{}, First few: {:?}",
                    count,
                    non_zero_count,
                    args.buffer.len(),
                    args.buffer.iter().take(4).collect::<Vec<_>>()
                );
            }

            pa::Continue
        },
    )?;

    Ok(stream)
}

fn process_input_samples(input: &[f32], num_channels: usize, selected_channels: &[usize]) -> Vec<f32> {
    let frames = input.len() / num_channels;
    let mut processed = Vec::with_capacity(frames * selected_channels.len());
    
    info!(
        "Processing - Frames: {}, Device channels: {}, Selected channels: {:?}",
        frames, num_channels, selected_channels
    );

    // Validate channel selection
    if let Some(&max_channel) = selected_channels.iter().max() {
        if max_channel >= num_channels {
            error!("Channel selection out of range: {} >= {}", max_channel, num_channels);
            return Vec::new();
        }
    }

    // Extract selected channels frame by frame
    for frame in 0..frames {
        let frame_offset = frame * num_channels;
        for &channel in selected_channels {
            let sample = input[frame_offset + channel];
            processed.push(sample);
        }
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
pub fn start_sampling_thread(
    running: Arc<AtomicBool>,
    main_buffer: Arc<RwLock<CircularBuffer>>,
    selected_channels: Vec<usize>,
    sample_rate: f64,
    _buffer_size: Arc<Mutex<usize>>,
    device_index: pa::DeviceIndex,
) {
    let pa = pa::PortAudio::new().expect("Failed to initialize PortAudio");
    info!("PortAudio initialized for sampling thread.");

    let device_info = pa.device_info(device_index).unwrap();
    let device_channels = device_info.max_input_channels as usize;
    
    info!("Initializing main audio stream...");
    
    // Create and start the stream
    let stream_result = build_input_stream(
        &pa,
        device_index,
        device_channels,
        selected_channels.clone(),
        sample_rate,
        Arc::clone(&main_buffer),
        Arc::clone(&running),
    );

    match stream_result {
        Ok(mut stream) => {
            info!("Successfully built input stream");
            match stream.start() {
                Ok(_) => {
                    info!("Audio stream started successfully");
                    
                    // Keep the stream alive until shutdown is requested
                    while running.load(Ordering::SeqCst) {
                        std::thread::sleep(std::time::Duration::from_millis(100));
                        
                        // Only check stream status occasionally
                        if let Ok(is_active) = stream.is_active() {
                            if !is_active {
                                error!("Stream became inactive, attempting restart...");
                                if let Err(e) = stream.start() {
                                    error!("Failed to restart stream: {}", e);
                                    break;
                                }
                            }
                        }
                    }
                    
                    info!("Shutdown requested, stopping audio stream...");
                    if let Err(e) = stream.stop() {
                        error!("Error stopping stream: {}", e);
                    }
                },
                Err(e) => error!("Failed to start stream: {}", e),
            }
        },
        Err(e) => error!("Failed to build input stream: {}", e),
    }
}
