// This section is protected. Do not alter unless permission is requested by you and granted by me.
use std::sync::{Arc, Mutex};
use eframe::egui;
use egui::plot::{Plot, BarChart};
pub use eframe::NativeOptions;
use crate::fft_analysis::FFTConfig;
use crate::audio_stream::CircularBuffer;
use log::{info, debug};
use std::sync::atomic::{AtomicBool, Ordering};// Importing necessary types for GUI throttling.
// Reminder: Added to implement GUI throttling. Do not modify without permission.
use std::time::{Duration, Instant};
use std::sync::RwLock;
use crate::utils::{MIN_FREQ, MAX_FREQ, MIN_BUFFER_SIZE, MAX_BUFFER_SIZE, calculate_optimal_buffer_size, FRAME_SIZES, map_db_range, DEFAULT_BUFFER_SIZE};
use crate::display::SpectralDisplay;
use crate::fft_analysis::WindowType;  // Add at top with other imports
use crate::fft_analysis::{apply_window, extract_channel_data};
use realfft::RealFftPlanner;
use crate::resynth::ResynthConfig;  // Add this import


// This section is protected. Do not alter unless permission is requested by you and granted by me.
pub struct SpectrumApp {
    absolute_values: Vec<Vec<(f32, f32)>>,       // Frequency, absolute pairs
}

// This section is protected. Do not alter unless permission is requested by you and granted by me.
impl SpectrumApp {
    pub fn new(num_channels: usize) -> Self {
        SpectrumApp {
            absolute_values: vec![vec![(0.0, 0.0); 12]; num_channels],
        }
    }

    pub fn update_partials(&mut self, new_values: Vec<Vec<(f32, f32)>>) {
        self.absolute_values = new_values;  // Store directly
    }

    #[allow(dead_code)]
    /// Get a copy of the current spectral data in absolute values (matching GUI display)
    pub fn clone_absolute_data(&self) -> Vec<Vec<(f32, f32)>> {
        self.absolute_values.clone()
    }
}

// This section is protected. Do not alter unless permission is requested by you and granted by me.
pub struct MyApp {
    pub spectrum: Arc<Mutex<SpectrumApp>>,
    pub fft_config: Arc<Mutex<FFTConfig>>,
    pub buffer_size: Arc<Mutex<usize>>,
    pub audio_buffer: Arc<RwLock<CircularBuffer>>,
    pub resynth_config: Arc<Mutex<ResynthConfig>>,
    colors: Vec<egui::Color32>,
    y_scale: f32,
    alpha: u8,
    bar_width: f32,
    show_line_plot: bool,
    last_repaint: Instant,
    shutdown_flag: Arc<AtomicBool>,
}

// This section is protected. Do not alter unless permission is requested by you and granted by me.
impl MyApp {
    pub fn new(
        spectrum: Arc<Mutex<SpectrumApp>>,
        fft_config: Arc<Mutex<FFTConfig>>,
        buffer_size: Arc<Mutex<usize>>,
        audio_buffer: Arc<RwLock<CircularBuffer>>,
        resynth_config: Arc<Mutex<ResynthConfig>>,
        shutdown_flag: Arc<AtomicBool>,
    ) -> Self {
        let colors = vec![
            egui::Color32::from_rgb(0, 0, 255),
            egui::Color32::from_rgb(255, 165, 0),
            egui::Color32::from_rgb(0, 255, 0),
            egui::Color32::from_rgb(255, 0, 0),
            egui::Color32::from_rgb(238, 130, 238),
            egui::Color32::from_rgb(165, 42, 42),
            egui::Color32::from_rgb(75, 0, 130),
            egui::Color32::from_rgb(255, 255, 0),
        ];

        let instance = MyApp {
            spectrum,
            fft_config,
            buffer_size,
            audio_buffer,
            resynth_config,
            colors,
            y_scale: 80.0,
            alpha: 50,
            bar_width: 5.0,
            show_line_plot: false,
            last_repaint: Instant::now(),
            shutdown_flag,
        };

        // FIX IMPLEMENTATION:
        // After we construct `instance`, we clamp `max_frequency` so the x scale
        // is correct on startup, if current max_frequency exceeds nyquist_limit.
        {
            let buffer_s = instance.buffer_size.lock().unwrap();
            let nyquist_limit = (*buffer_s as f64 / 2.0).min(20000.0);

            let mut cfg = instance.fft_config.lock().unwrap();
            if cfg.max_frequency > nyquist_limit {
                cfg.max_frequency = nyquist_limit;
                debug!("Clamped max frequency at startup to {}", cfg.max_frequency);
            }
        }

        // Return the newly created instance with the fix
        instance
    }

    pub fn update_buffer_size(&mut self, new_size: usize) {
        let validated_size = new_size
            .next_power_of_two()
            .max(512)
            .clamp(MIN_BUFFER_SIZE, MAX_BUFFER_SIZE);
        
        // Only log size adjustments at info level
        if validated_size != new_size {
            info!("Buffer size adjusted: {} → {}", new_size, validated_size);
        }
        
        // Log detailed calculations at debug level
        debug!("Buffer validation: min={}, max={}, power_of_2={}", 
               MIN_BUFFER_SIZE, MAX_BUFFER_SIZE, validated_size);
        
        // Update frame size if necessary
        if let Ok(mut fft_config) = self.fft_config.lock() {
            if fft_config.frames_per_buffer as usize > validated_size {
                let new_frames = FRAME_SIZES.iter()
                    .rev()
                    .find(|&&x| x as usize <= validated_size)
                    .copied()
                    .unwrap_or(FRAME_SIZES[0]);
                fft_config.frames_per_buffer = new_frames;
                info!("Adjusted frames per buffer to {} due to buffer size change", new_frames);
            }
        }
        
        if let Ok(mut size) = self.buffer_size.lock() {
            *size = validated_size;
        }
        
        if let Ok(mut buffer) = self.audio_buffer.write() {
            buffer.resize(validated_size);
        }
    }
}

// This section is protected. Do not alter unless permission is requested by you and granted by me.
// Implementing eframe::App for MyApp

impl eframe::App for MyApp {
    fn on_close_event(&mut self) -> bool {
        info!("Closing GUI window...");
        self.shutdown_flag.store(true, Ordering::SeqCst);  // Changed back to true for shutdown
        true // Allow the window to close
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Throttling: Limit repaint to at most 10 times per second (every 100 ms)
        let now = Instant::now();
        if now.duration_since(self.last_repaint) >= Duration::from_millis(100) {
            ctx.request_repaint();
            self.last_repaint = now;
        }

        // Force continuous updates every 100 ms
        ctx.request_repaint_after(Duration::from_millis(100));

        ctx.set_visuals(egui::Visuals::dark());

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label("First Twelve Partials per Channel");

            // Before all controls
            let mut size_changed = false;  // Single declaration for size_changed

            // 1) First row of sliders
            {
                let mut fft_config = self.fft_config.lock().unwrap();
                ui.horizontal(|ui| {
                    ui.label("Min Frequency:");
                    ui.add(egui::Slider::new(&mut fft_config.min_frequency, 10.0..=200.0).text("Hz"));
                    
                    ui.label("Max Frequency:");
                    let buffer_size = *self.buffer_size.lock().unwrap();
                    let nyquist_limit = (buffer_size as f64 / 2.0).min(20000.0);
                    ui.add(egui::Slider::new(&mut fft_config.max_frequency, 0.0..=nyquist_limit).text("Hz"));

                    ui.label("Magnitude Threshold:");
                    ui.add(egui::Slider::new(&mut fft_config.magnitude_threshold, 0.0..=30.0));
                });
            }

            // 2) Second row with frames, buffer size, and averaging
            {
                ui.horizontal(|ui| {
                    // Buffer size slider
                    {
                        let buffer_size = *self.buffer_size.lock().unwrap();
                        let mut buffer_log_slider = (buffer_size as f32).log2().round() as u32;
                        ui.label("Buffer Size:");
                        let min_power = (MIN_BUFFER_SIZE as f32).log2() as u32;
                        let max_power = (MAX_BUFFER_SIZE as f32).log2() as u32;
                        if ui
                            .add(egui::Slider::new(&mut buffer_log_slider, min_power..=max_power)
                                .custom_formatter(|n, _| {
                                    format!("{}", 1_usize << (n as usize))
                                }))
                            .changed()
                        {
                            let new_size = 1 << buffer_log_slider;
                            self.update_buffer_size(new_size);
                            size_changed = true;
                        }
                        ui.label("samples");
                    }

                    // Min freq spacing slider
                    {
                        let mut fft_config = self.fft_config.lock().unwrap();
                        ui.label("Min Freq Spacing:");
                        ui.add(egui::Slider::new(&mut fft_config.min_freq_spacing, 0.0..=80.0).text("Hz"));
                    }

                    // Window type selection
                    {
                        let mut fft_config = self.fft_config.lock().unwrap();
                        ui.label("Window Type:");
                        egui::ComboBox::from_label("")
                            .selected_text(format!("{:?}", fft_config.window_type))
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut fft_config.window_type, WindowType::Rectangular, "Rectangular");
                                ui.selectable_value(&mut fft_config.window_type, WindowType::Hanning, "Hanning");
                                ui.selectable_value(&mut fft_config.window_type, WindowType::Hamming, "Hamming");
                                ui.selectable_value(&mut fft_config.window_type, WindowType::BlackmanHarris, "Blackman-Harris");
                                ui.selectable_value(&mut fft_config.window_type, WindowType::FlatTop, "Flat Top");
                                if ui.selectable_value(&mut fft_config.window_type, WindowType::Kaiser(4.0), "Kaiser").clicked() {
                                    fft_config.window_type = WindowType::Kaiser(4.0);
                                }
                            });
                    }
                });

                // 3) Sliders for Y scale, alpha, bar width, and Kaiser beta
                ui.horizontal(|ui| {
                    ui.label("Y Max:");
                    ui.add(egui::Slider::new(&mut self.y_scale, 0.0..=100.0).text(""));
                    ui.label("Alpha:");
                    ui.add(egui::Slider::new(&mut self.alpha, 0..=255).text(""));
                    ui.label("Bar Width:");
                    ui.add(egui::Slider::new(&mut self.bar_width, 1.0..=10.0).text(""));
                    
                    // Add gain slider
                    {
                        let mut resynth_config = self.resynth_config.lock().unwrap();
                        ui.label("Gain:");
                        ui.add(egui::Slider::new(&mut resynth_config.gain, 0.0..=1.0));
                    }

                    ui.checkbox(&mut self.show_line_plot, "Show FFT");

                    // Add Kaiser beta slider here if Kaiser window is selected
                    {
                        let mut fft_config = self.fft_config.lock().unwrap();
                        if let WindowType::Kaiser(_) = fft_config.window_type {
                            ui.label("Kaiser β:");
                            let mut beta = match fft_config.window_type {
                                WindowType::Kaiser(b) => b,
                                _ => 4.0,
                            };
                            if ui.add(egui::Slider::new(&mut beta, 0.0..=20.0)).changed() {
                                fft_config.window_type = WindowType::Kaiser(beta);
                            }
                        }
                    }
                });

                // Handle max frequency adjustment if buffer size changed
                if size_changed {
                    let buffer_size = *self.buffer_size.lock().unwrap();
                    let nyquist_limit = (buffer_size as f64 / 2.0).min(20000.0);
                    let mut fft_config = self.fft_config.lock().unwrap();
                    if fft_config.max_frequency > nyquist_limit {
                        fft_config.max_frequency = nyquist_limit;
                    }
                }
            }

            // 4) Reset button
            let mut reset_clicked = false;
            if ui.button("Reset to Defaults").clicked() {
                {
                    let mut fft_config = self.fft_config.lock().unwrap();
                    let old_frames = fft_config.frames_per_buffer;
                    
                    // Reset FFT config
                    fft_config.min_frequency = MIN_FREQ;
                    fft_config.max_frequency = MAX_FREQ;
                    fft_config.magnitude_threshold = 6.0;
                    fft_config.min_freq_spacing = 20.0;  // Changed from 1.0 to 20.0
                    fft_config.window_type = WindowType::Hanning;  // Changed from BlackmanHarris to Hanning
                    
                    // Only change frames_per_buffer if platform requires it
                    let new_frames = if cfg!(target_os = "linux") {
                        1024  // Larger buffer for Linux stability
                    } else {
                        512   // Default for other platforms
                    };
                    
                    // Only update if different to avoid unnecessary restarts
                    if old_frames != new_frames {
                        fft_config.frames_per_buffer = new_frames;
                        debug!("Frames per buffer changed from {} to {}", old_frames, new_frames);
                    }
                }
                
                self.y_scale = 80.0;
                self.alpha = 50;
                self.bar_width = 5.0;
                self.show_line_plot = false;
                
                // Calculate optimal size based on current sample rate
                let sample_rate = 48000.0f64;
                let optimal_size = calculate_optimal_buffer_size(sample_rate);
                
                // Fix: Get current size without holding the lock during update
                let needs_update = {
                    let current_size = *self.buffer_size.lock().unwrap();
                    current_size != DEFAULT_BUFFER_SIZE
                };
                
                if needs_update {
                    self.update_buffer_size(DEFAULT_BUFFER_SIZE);
                }
                
                let mut resynth_config = self.resynth_config.lock().unwrap();
                resynth_config.gain = 0.5;  // Default gain
                
                reset_clicked = true;
            }

            // 5) After all sliders, handle changes outside the slider blocks
            if size_changed || reset_clicked {
                let buffer_size = *self.buffer_size.lock().unwrap();
                let nyquist_limit = (buffer_size as f64 / 2.0).min(20000.0);
                
                let mut fft_config = self.fft_config.lock().unwrap();
                if fft_config.max_frequency > nyquist_limit {
                    fft_config.max_frequency = nyquist_limit;
                }
            }

            // 6) Plot logic
            let absolute_values = {
                let spectrum = self.spectrum.lock().unwrap();
                spectrum.absolute_values.clone()  // Contains dB values used for both plotting and display
            };

            // Bar charts with static legend names - always show all channels
            let all_bar_charts: Vec<BarChart> = (0..absolute_values.len())
                .map(|channel| {
                    let channel_partials = &absolute_values[channel];
                    let mut bars: Vec<egui::plot::Bar> = channel_partials
                        .iter()
                        .filter(|&&(freq, raw_val)| freq > 0.0 && raw_val != 0.0)
                        .map(|&(freq, raw_val)| {
                            egui::plot::Bar::new(freq as f64, raw_val as f64)
                                .width(self.bar_width as f64)
                        })
                        .collect();

                    // If no visible bars, add a single invisible bar
                    if bars.is_empty() {
                        bars.push(egui::plot::Bar::new(0.0, 0.0).width(0.0));
                    }

                    let color = self.colors[channel % self.colors.len()]
                        .linear_multiply(self.alpha as f32 / 255.0);

                    BarChart::new(bars)
                        .name(format!("Channel {}", channel + 1))
                        .color(color)
                })
                .collect();

            // Line plots without legend names - more memory efficient
            let all_line_plots: Vec<egui::plot::Line> = if self.show_line_plot {
                (0..absolute_values.len())
                    .map(|channel| {
                        let fft_config = self.fft_config.lock().unwrap();
                        let buffer = self.audio_buffer.read().unwrap();
                        let buffer_data = buffer.clone_data();
                        
                        let channel_data = extract_channel_data(&buffer_data, channel, fft_config.num_channels);
                        let windowed = apply_window(&channel_data, fft_config.window_type);
                        
                        let mut planner = RealFftPlanner::<f32>::new();
                        let fft = planner.plan_fft_forward(windowed.len());
                        let mut spectrum = fft.make_output_vec();
                        let _ = fft.process(&mut windowed.clone(), &mut spectrum);
                        
                        let freq_step = 48000.0 / windowed.len() as f32;
                        let points: Vec<[f64; 2]> = spectrum
                            .iter()
                            .take((fft_config.max_frequency as f32 / freq_step) as usize)
                            .enumerate()
                            .map(|(i, &complex_val)| {
                                let freq = i as f64 * freq_step as f64;
                                let mag = ((complex_val.re * complex_val.re + complex_val.im * complex_val.im).sqrt() / 4.0) as f64;
                                [freq, mag]
                            })
                            .collect();

                        let color = self.colors[channel % self.colors.len()]
                            .linear_multiply(self.alpha as f32 / 255.0);

                        egui::plot::Line::new(points).color(color)
                    })
                    .collect()
            } else {
                Vec::new()
            };

            let max_freq = {
                let fft = self.fft_config.lock().unwrap();
                fft.max_frequency
            };

            Plot::new("spectrum_plot")
                .legend(egui::plot::Legend::default())
                .view_aspect(6.0)
                .include_x(0.0)
                .include_x(max_freq as f64)
                .include_y(0.0)
                .include_y(self.y_scale as f64)
                .show(ui, |plot_ui| {
                    // Draw both bar charts and line plots
                    for bar_chart in all_bar_charts {
                        plot_ui.bar_chart(bar_chart);
                    }
                    if self.show_line_plot {
                        for line in all_line_plots {
                            plot_ui.line(line);
                        }
                    }
                });

            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.label("Channel Results:");
                let display = SpectralDisplay::new(&absolute_values);
                for line in display.format_all() {
                    ui.label(line);
                }
            });
        });
    }
}

// This section is protected. Do not alter unless permission is requested by you and granted by me.
#[allow(dead_code)]
pub fn run_native(
    app_name: &str,
    native_options: NativeOptions,
    app_creator: Box<dyn FnOnce(&eframe::CreationContext<'_>) -> Box<MyApp>>,
) -> Result<(), eframe::Error> {
    let shutdown_flag = Arc::new(AtomicBool::new(false));

    eframe::run_native(
        app_name,
        native_options,
        Box::new(move |cc| {
            let mut app = app_creator(cc);
            app.shutdown_flag = shutdown_flag.clone();  // Directly assign to MyApp
            app as Box<dyn eframe::App>
        }),
    )
}

// Total line count: 259
