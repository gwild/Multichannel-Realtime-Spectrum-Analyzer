use anyhow::{anyhow, Result};
use portaudio as pa;
use std::sync::{Arc, Mutex};
use crate::fft_analysis::{compute_spectrum, NUM_PARTIALS, FFTConfig};
use crate::plot::SpectrumApp;
use portaudio::stream::InputCallbackArgs;
use log::{info, error};

/// Circular buffer implementation for storing audio samples.
///
/// This buffer allows for continuous writing and reading of audio samples,
/// maintaining a fixed size by overwriting the oldest data when new samples arrive.
pub struct CircularBuffer {
    buffer: Vec<f32>,
    head: usize,
    size: usize,
}

impl CircularBuffer {
    /// Creates a new `CircularBuffer` with the specified size.
    ///
    /// # Arguments
    ///
    /// * `size` - The maximum number of samples the buffer can hold.
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
    pub fn push(&mut self, value: f32) {
        self.buffer[self.head] = value;
        self.head = (self.head + 1) % self.size; // Wrap around
        info!("Pushed sample to buffer: {}", value);
    }

    /// Retrieves the current contents of the buffer.
    ///
    /// # Returns
    ///
    /// A slice of the buffer containing the audio samples.
    pub fn get(&self) -> &[f32] {
        &self.buffer
    }
}

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

    info!("Opening non-blocking PortAudio stream.");

    // Create the non-blocking stream with a callback to process incoming audio data
    let stream = pa.open_non_blocking_stream(
        settings,
        move |args: InputCallbackArgs<f32>| {
            info!("Callback triggered with {} samples", args.buffer.len());
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

    info!("PortAudio stream opened successfully.");

    Ok(stream)
}

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
    // Fill the audio buffers for each selected channel
    for (i, &sample) in data_as_f32.iter().enumerate() {
        let channel = i % channels;
        info!("Received sample: {} for channel {}", sample, channel);
        if let Some(buffer_index) = selected_channels.iter().position(|&ch| ch == channel) {
            if let Ok(mut buffer) = audio_buffers[buffer_index].lock() {
                buffer.push(sample);
            } else {
                error!("Failed to lock buffer for channel {}", buffer_index);
            }
        }
    }

    let config = fft_config.lock().unwrap();
    let mut partials_results = vec![vec![(0.0, 0.0); NUM_PARTIALS]; selected_channels.len()];

    // Compute spectrum for each selected channel
    for (i, &channel) in selected_channels.iter().enumerate() {
        if let Ok(buffer) = audio_buffers[i].lock() {
            let audio_data = buffer.get();
            if !audio_data.is_empty() {
                let computed_partials = compute_spectrum(audio_data, sample_rate, &config);
                for (j, &partial) in computed_partials.iter().enumerate().take(NUM_PARTIALS) {
                    partials_results[i][j] = partial;
                }
                info!(
                    "Channel {}: Partial Results: {:?}",
                    channel, partials_results[i]
                );
            }
        }
    }

    if let Ok(mut app) = spectrum_app.lock() {
        app.partials = partials_results;
    }
}
