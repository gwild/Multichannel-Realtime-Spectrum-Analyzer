use anyhow::Result;
use cpal::traits::{DeviceTrait, StreamTrait};
use cpal::{Stream, StreamConfig, Sample, SizedSample};
use std::sync::{Arc, Mutex};
use num_traits::ToPrimitive;  // Import the ToPrimitive trait for conversion
use crate::fft_analysis::compute_spectrum;
use crate::plot::SpectrumApp;

const MAX_BUFFER_SIZE: usize = 1024; // Define a max buffer size

pub fn build_input_stream<T>(
    device: &cpal::Device,
    config: &StreamConfig,
    audio_buffers: Arc<Vec<Mutex<Vec<f32>>>>,
    spectrum_app: Arc<Mutex<SpectrumApp>>,
    selected_channels: Vec<usize>, // List of channels to use
) -> Result<Stream>
where
    T: Sample + SizedSample + ToPrimitive + std::fmt::Debug, // Add ToPrimitive here
{
    let channels = config.channels as usize;
    let sample_rate = config.sample_rate.0;

    let stream = device.build_input_stream(
        config,
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            // Fill the audio buffers for each selected channel
            for (i, sample) in data.iter().enumerate() {
                let channel = i % channels;
                if selected_channels.contains(&channel) {
                    let buffer_index = selected_channels.iter().position(|&ch| ch == channel).unwrap();
                    let mut buffer = audio_buffers[buffer_index].lock().unwrap();
                    
                    // Convert the sample to f32 using ToPrimitive
                    let sample_as_f32: f32 = sample.to_f64().unwrap_or(0.0) as f32;
                    
                    buffer.push(sample_as_f32);

                    // Manage buffer size by removing the oldest sample if it exceeds the max size
                    if buffer.len() > MAX_BUFFER_SIZE {
                        buffer.remove(0);
                    }
                }
            }

            // Compute spectrum for each selected channel and store the partials and FFT results
            let mut partials_results = Vec::with_capacity(selected_channels.len()); // Preallocate for efficiency
            let mut fft_results = Vec::with_capacity(selected_channels.len()); // Preallocate for FFT results

            for (i, &channel) in selected_channels.iter().enumerate() {
                let buffer = audio_buffers[i].lock().unwrap();

                // Compute the partials
                let partials = compute_spectrum(&buffer, sample_rate);
                let partials_converted: Vec<(f64, f64)> = partials
                    .iter()
                    .map(|&(freq, amp)| (f64::from(freq), f64::from(amp))) // Convert each tuple to f64
                    .collect();
                partials_results.push(partials_converted);

                // Compute FFT results (same function, adapt as needed)
                let fft = compute_spectrum(&buffer, sample_rate);
                let fft_converted: Vec<(f64, f64)> = fft
                    .iter()
                    .map(|&(freq, amp)| (f64::from(freq), f64::from(amp))) // Convert each tuple to f64
                    .collect();
                fft_results.push(fft_converted);
            }

            // Update the spectrum_app with the new partials and FFT results for each channel
            let mut app = spectrum_app.lock().unwrap();
            app.partials.clone_from_slice(&partials_results); // Update partials
            app.fft_results.clone_from_slice(&fft_results); // Update FFT results
        },
        move |err| {
            eprintln!("Stream error: {:?}", err);
        },
        None,
    )?;

    Ok(stream)
}
