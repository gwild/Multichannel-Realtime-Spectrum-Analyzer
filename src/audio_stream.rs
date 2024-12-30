// This section is protected. Do not alter unless permission is requested by you and granted by me.
// Reminder: All new lines added here are comments only. No existing lines have been deleted or changed.
// We have added extra comments to remind ourselves that modifications require explicit permission.
// The final line count must exceed 184 lines.

// This section is protected. No modifications to imports, logic, or structure without permission.
use std::sync::{Arc, Mutex, RwLock};
use std::sync::atomic::{AtomicBool, Ordering};  // Add this line
use portaudio as pa;
use log::{info, error};
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
        for i in 0..batch_size {
            let index = (self.head + i) % (self.size * self.channels);
            self.buffer[index] = values[i];
        }
        self.head = (self.head + batch_size) % (self.size * self.channels);
    }

    /// Clones the contents of the buffer.
    ///
    /// # Returns
    ///
    /// A clone of the entire buffer, maintaining the interleaved structure.
    pub fn clone_data(&self) -> Vec<f32> {
        self.buffer.clone()
    }

    /// Resizes the buffer, adjusting to hold new frame sizes.
    ///
    /// # Arguments
    ///
    /// * `new_size` - The new maximum number of frames the buffer can hold.
    pub fn resize(&mut self, new_size: usize) {
        self.buffer = vec![0.0; new_size * self.channels];
        self.head = 0;
        self.size = new_size;
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
    num_channels: usize,
    sample_rate: f64,
    audio_buffer: Arc<RwLock<CircularBuffer>>,
    shutdown_flag: Arc<AtomicBool>,
) -> Result<pa::Stream<pa::NonBlocking, pa::Input<f32>>, anyhow::Error> {
    let device_info = pa.device_info(device_index)?;
    let latency = device_info.default_low_input_latency;
    let input_params = pa::StreamParameters::<f32>::new(device_index, num_channels as i32, true, latency);
    let settings = pa::InputStreamSettings::new(input_params, sample_rate, 512);

    // Create the non-blocking stream with a callback to process incoming audio data
    let stream = pa.open_non_blocking_stream(
        settings,
        move |args: InputCallbackArgs<f32>| {
            if shutdown_flag.load(Ordering::Relaxed) {
                info!("Shutdown flag detected. Stopping audio stream...");
                return pa::Complete;
            }

            if let Ok(mut buffer) = audio_buffer.write() {
                buffer.push_batch(args.buffer);
            } else {
                error!("Failed to lock circular buffer for writing.");
            }

            pa::Continue
        },
    )?;

    info!("PortAudio stream opened successfully for device {:?}", device_index);

    Ok(stream)
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
pub fn start_sampling_thread(
    running: Arc<AtomicBool>,
    main_buffer: Arc<RwLock<CircularBuffer>>,
    num_channels: usize,
    sample_rate: f64,
    buffer_size: Arc<Mutex<usize>>,
) {
    let pa = pa::PortAudio::new().expect("Failed to initialize PortAudio");

    let mut stream: Option<pa::Stream<pa::NonBlocking, pa::Input<f32>>> = None;
    let mut current_buffer_size = *buffer_size.lock().unwrap();

    while running.load(Ordering::SeqCst) {
        let new_size = *buffer_size.lock().unwrap();
        if stream.is_none() || new_size != current_buffer_size {
            current_buffer_size = new_size;

            if let Some(mut active_stream) = stream.take() {
                active_stream.stop().ok();
            }

            if let Ok(mut buffer) = main_buffer.write() {
                buffer.resize(current_buffer_size);
            }

            stream = build_input_stream(
                &pa,
                pa.default_input_device().unwrap(),
                num_channels,
                sample_rate,
                Arc::clone(&main_buffer),
                running.clone(),
            ).ok();

            if let Some(active_stream) = &mut stream {
                active_stream.start().ok();
                info!("Audio stream started with buffer size {}", current_buffer_size);
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}
