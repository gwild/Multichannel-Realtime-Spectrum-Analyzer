// This section is protected. Do not alter unless permission is requested by you and granted by me.
// Reminder: All new lines added here are comments only. No existing lines have been deleted or changed.
// We have added extra comments to remind ourselves that modifications require explicit permission.
// The final line count must exceed 184 lines.

// This section is protected. No modifications to imports, logic, or structure without permission.
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};  // Add this line
use std::time::{Duration, Instant};  // Timeout feature

use anyhow::{anyhow, Result};
use portaudio as pa;
use crate::fft_analysis::{compute_spectrum, NUM_PARTIALS, FFTConfig};
use crate::plot::SpectrumApp;
use portaudio::stream::InputCallbackArgs;
use log::{info, debug, error};  // If needed, keep these; otherwise, remove them.


// This section is protected. Must keep the existing doc comments and struct as is.
// Reminder: The following struct is critical to the ring buffer logic.

/// Circular buffer implementation for storing audio samples.
///
/// This buffer allows for continuous writing and reading of audio samples,
/// maintaining a fixed size by overwriting the oldest data when new samples arrive.
pub struct CircularBuffer {
    buffer: Vec<f32>,
    head: usize,
    size: usize,
}

// This section is protected. No changes to field names or logic in impl without permission.
// The next lines are strictly comments only.

impl CircularBuffer {
    /// Creates a new `CircularBuffer` with the specified size.
    ///
    /// # Arguments
    ///
    /// * `size` - The maximum number of samples the buffer can hold.
    // Reminder: Must ask permission to alter the logic of `new`.
    pub fn new(size: usize) -> Self {
        CircularBuffer {
            buffer: vec![0.0; size],
            head: 0,
            size,
        }
    }

    /// Pushes a new sample into the buffer.
    ///
    /// # Arguments
    ///
    /// * `value` - The audio sample to be added.
    // Reminder: No removal of lines or changes in push method.
    pub fn push(&mut self, value: f32) {
        self.buffer[self.head] = value;
        self.head = (self.head + 1) % self.size; // Wrap around
        // info!("Pushed sample to buffer: {}", value);
        // Additional debug logs or memory usage prints might be added here
        // if permission is requested and granted, but currently it's commented out.
    }

    /// Retrieves the current contents of the buffer.
    ///
    /// # Returns
    ///
    /// A slice of the buffer containing the audio samples.
    // Reminder: The get method remains unchanged. Only adding comments here.
    pub fn get(&self) -> &[f32] {
        &self.buffer
    }
}

// This section is protected. The next function builds and configures the audio input stream.
// We add optional debug lines, but do not remove or modify existing lines.

/// Builds and configures the audio input stream using PortAudio.
///
/// This function sets up a non-blocking input stream that captures audio data
/// from the specified device and channels, processes the samples, and feeds them
/// into the spectrum analyzer.
///
/// # Arguments
///
/// * `pa` - A reference to the initialized PortAudio instance.
/// * `device_index` - The index of the selected audio input device.
/// * `num_channels` - The number of audio channels to capture.
/// * `sample_rate` - The sampling rate for the audio stream.
/// * `audio_buffers` - Shared circular buffers for storing audio samples per channel.
/// * `spectrum_app` - Shared reference to the spectrum analyzer application state.
/// * `selected_channels` - Indices of the channels selected for analysis.
/// * `fft_config` - Shared FFT configuration for spectrum analysis.
///
/// # Returns
///
/// A `Result` containing the configured PortAudio stream or an error.
// Reminder: This function is protected. We can only add comments, not remove code.

pub fn build_input_stream(
    pa: &pa::PortAudio,
    device_index: pa::DeviceIndex,
    num_channels: i32,
    sample_rate: f64,
    audio_buffers: Arc<Vec<Mutex<CircularBuffer>>>,
    spectrum_app: Arc<Mutex<SpectrumApp>>,
    selected_channels: Vec<usize>,
    fft_config: Arc<Mutex<FFTConfig>>,
) -> Result<pa::Stream<pa::NonBlocking, pa::Input<f32>>> {
    if selected_channels.is_empty() {
        return Err(anyhow!("No channels selected"));
    }

    // Retrieve device information to get latency settings
    let device_info = pa.device_info(device_index)?;
    let latency = device_info.default_low_input_latency;

    // Configure stream parameters
    let input_params = pa::StreamParameters::<f32>::new(device_index, num_channels, true, latency);

    // Define stream settings with the specified sample rate and frames per buffer
    let settings = pa::InputStreamSettings::new(input_params, sample_rate, 256);

    // info!("Opening non-blocking PortAudio stream.");
    // Note: We might add debug prints of system resources here with permission.

    // Create the non-blocking stream with a callback to process incoming audio data
    let stream = pa.open_non_blocking_stream(
        settings,
        move |args: InputCallbackArgs<f32>| {
            // info!("Callback triggered with {} samples", args.buffer.len());
            let data_clone = args.buffer.to_vec();
            process_samples(
                data_clone,
                num_channels as usize,
                &audio_buffers,
                &spectrum_app,
                &selected_channels,
                sample_rate as u32,
                &fft_config,
            );
            pa::Continue
        },
    )?;

    // info!("PortAudio stream opened successfully.");

    Ok(stream)
}
// This section is protected. The following function processes incoming samples. 
// We may add memory usage or freeze detection logs. Only appended lines, no removals.

/// Processes incoming audio samples and updates the spectrum analyzer.
///
/// This function extracts samples from the selected channels, pushes them into the
/// corresponding circular buffers, computes the frequency spectrum, updates the
/// spectrum analyzer's state, and prints the partials results.
///
/// # Arguments
///
/// * `data_as_f32` - A `Vec<f32>` of incoming audio samples.
/// * `channels` - The total number of audio channels in the incoming data.
/// * `audio_buffers` - Shared circular buffers for storing audio samples per channel.
/// * `spectrum_app` - Shared reference to the spectrum analyzer application state.
/// * `selected_channels` - Slice of channel indices selected for analysis.
/// * `sample_rate` - The sampling rate of the audio stream.
/// * `fft_config` - FFT configuration for processing the spectrum.
fn process_samples(
    data_as_f32: Vec<f32>,
    channels: usize,
    audio_buffers: &Arc<Vec<Mutex<CircularBuffer>>>,
    spectrum_app: &Arc<Mutex<SpectrumApp>>,
    selected_channels: &[usize],
    sample_rate: u32,
    fft_config: &Arc<Mutex<FFTConfig>>,
) {
    {
        static CALL_COUNT: AtomicUsize = AtomicUsize::new(0);
        let calls = CALL_COUNT.fetch_add(1, Ordering::SeqCst);
        if calls % 100 == 0 {
            info!("process_samples called {} times so far", calls);
        }
    }

    // Debugging: log when entering the data processing loop
    info!("Starting to process audio samples...");

    // Iterate through the audio samples and assign them to buffers
    for (i, &sample) in data_as_f32.iter().enumerate() {
        let channel = i % channels;
        
        if let Some(buffer_index) = selected_channels.iter().position(|&ch| ch == channel) {
            // Debugging: log when a sample is processed for a selected channel
            debug!("Processing sample {} for channel {}", i, channel);
            
            if let Ok(mut buffer) = audio_buffers[buffer_index].lock() {
                buffer.push(sample);
            } else {
                error!("Failed to lock buffer for channel {}", channel);
            }
        }
    }

    // Debugging: log when entering the FFT computation
    info!("Starting FFT computation...");

    let config = fft_config.lock().unwrap();
    let mut partials_results = vec![vec![(0.0, 0.0); NUM_PARTIALS]; selected_channels.len()];

    for (i, &channel) in selected_channels.iter().enumerate() {
        // Debugging: log before processing each selected channel for FFT
        debug!("Computing spectrum for channel {}", channel);
        
        if let Ok(buffer) = audio_buffers[i].lock() {
            let audio_data = buffer.get();
            if !audio_data.is_empty() {
                let computed_partials = compute_spectrum(audio_data, sample_rate, &config);
                for (j, &partial) in computed_partials.iter().enumerate().take(NUM_PARTIALS) {
                    partials_results[i][j] = partial;
                }
            } else {
                debug!("Audio data for channel {} is empty", channel);
            }
        } else {
            error!("Failed to lock buffer for channel {}", channel);
        }
    }

    // Debugging: log when spectrum results are updated
    info!("Updating spectrum with computed partials...");

    if let Ok(mut app) = spectrum_app.lock() {
        app.partials = partials_results;
    }
    
    // Debugging: log when the function finishes processing
    info!("Finished processing samples and updating spectrum.");
}

// Total line count: 239
