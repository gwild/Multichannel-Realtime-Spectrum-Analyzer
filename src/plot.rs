// This section is protected. Do not alter unless permission is requested by you and granted by me.
use std::sync::{Arc, Mutex};
use eframe::egui;
use egui::plot::{Plot, BarChart, Legend};
pub use eframe::NativeOptions;
use crate::fft_analysis::FFTConfig;
use crate::audio_stream::CircularBuffer;
use log::{info, debug, log_enabled, Level};
use std::sync::atomic::{AtomicBool, Ordering};// Importing necessary types for GUI throttling.
// Reminder: Added to implement GUI throttling. Do not modify without permission.
use std::time::{Duration, Instant};
use std::sync::RwLock;
use crate::utils::{MIN_FREQ, MAX_FREQ, MIN_BUFFER_SIZE, MAX_BUFFER_SIZE, calculate_optimal_buffer_size, FRAME_SIZES, DEFAULT_BUFFER_SIZE};
use crate::display::SpectralDisplay;
use crate::fft_analysis::WindowType;  // Add at top with other imports
use crate::fft_analysis::{apply_window, extract_channel_data};
use realfft::RealFftPlanner;
use crate::resynth::ResynthConfig;  // Add this import
use crate::resynth::DEFAULT_UPDATE_RATE;
use crate::DEFAULT_NUM_PARTIALS;  // Import the new constant
use egui::widgets::plot::uniform_grid_spacer;
use std::collections::VecDeque;
use std::collections::BTreeMap;
use chrono;
use egui::TextStyle;
use egui::FontId;
use egui::FontFamily;
use egui::Color32;

pub struct SpectrographSlice {
    pub time: f64,
    pub data: Vec<(f64, f32)>,
}

// This section is protected. Do not alter unless permission is requested by you and granted by me.
pub struct SpectrumApp {
    absolute_values: Vec<Vec<(f32, f32)>>,       // Frequency, absolute pairs
    partial_textures: Vec<Option<egui::TextureHandle>>,
    freq_axis_lines: Vec<Vec<[f32; 2]>>,
    num_channels: usize,
    num_partials: usize,  // Add num_partials field
    fft_line_data: Vec<Vec<(f32, f32)>>,  // Add this field
}

// This section is protected. Do not alter unless permission is requested by you and granted by me.
impl SpectrumApp {
    pub fn new(num_channels: usize) -> Self {
        SpectrumApp {
            absolute_values: vec![vec![(0.0, 0.0); DEFAULT_NUM_PARTIALS]; num_channels],  // Use DEFAULT_NUM_PARTIALS
            partial_textures: vec![None; num_channels],
            freq_axis_lines: vec![vec![]; 2],
            num_channels,
            num_partials: DEFAULT_NUM_PARTIALS,  // Initialize with default
            fft_line_data: Vec::new(),  // Initialize empty
        }
    }

    pub fn update_partials(&mut self, partials: Vec<Vec<(f32, f32)>>) {
        let num_channels = partials.len();
        self.absolute_values = partials;
        
        // Update num_partials if needed (assuming all channels have same number of partials)
        if !self.absolute_values.is_empty() {
            self.num_partials = self.absolute_values[0].len();
        }
        
        // Resize textures array to match channel count
        self.partial_textures.resize_with(num_channels, || None);
        
        // Always keep 2 frequency axis lines
        self.freq_axis_lines = vec![vec![]; 2]; 
        
        // Update channel count display
        self.num_channels = num_channels;
    }

    #[allow(dead_code)]
    /// Get a copy of the current spectral data in absolute values (matching GUI display)
    pub fn clone_absolute_data(&self) -> Vec<Vec<(f32, f32)>> {
        self.absolute_values.clone()
    }

    pub fn update_fft_line_data(&mut self, data: Vec<Vec<(f32, f32)>>) {
        self.fft_line_data = data;
    }

    pub fn get_fft_line_data(&self) -> &Vec<Vec<(f32, f32)>> {
        &self.fft_line_data
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
    show_spectrograph: bool,
    last_repaint: Instant,
    shutdown_flag: Arc<AtomicBool>,
    spectrograph_history: Arc<Mutex<VecDeque<SpectrographSlice>>>,
    start_time: Arc<Instant>,
    sample_rate: f64,
    show_results: bool,
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
        spectrograph_history: Arc<Mutex<VecDeque<SpectrographSlice>>>,
        start_time: Arc<Instant>,
        sample_rate: f64,
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
            alpha: 255,
            bar_width: 5.0,
            show_line_plot: false,
            show_spectrograph: false,
            last_repaint: Instant::now(),
            shutdown_flag,
            spectrograph_history,
            start_time,
            sample_rate,
            show_results: true,
        };

        // FIX IMPLEMENTATION:
        // After we construct `instance`, we clamp `max_frequency` so the x scale
        // is correct on startup, if current max_frequency exceeds nyquist_limit.
        {
            let buffer_s = instance.buffer_size.lock().unwrap();
            let nyquist_limit = (*buffer_s as f64 / 2.0).min(MAX_FREQ);

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

        // Clear spectrograph history and reset start time when buffer size changes
        if let Ok(mut history) = self.spectrograph_history.lock() {
            let slices_per_second = self.sample_rate / validated_size as f64;
            let history_len = (5.0 * slices_per_second).ceil() as usize;
            // Trim or extend, but do not clear
            while history.len() > history_len {
                history.pop_front();
            }
            // No need to extend; VecDeque will grow as needed
        }
        *Arc::make_mut(&mut self.start_time) = Instant::now();

        // Explicitly reset plot bounds by requesting a repaint
        self.last_repaint = Instant::now();

        // Explicitly log the clearing action for debugging with structured context
        info!("Spectrograph history trimmed and start time reset due to buffer size change (new size: {})", validated_size);
    }

    // Helper method to get the current nyquist limit
    fn get_nyquist_limit(&self) -> f32 {
        let buffer_size = *self.buffer_size.lock().unwrap();
        (buffer_size as f32 / 2.0)
    }

    pub fn update_ui(&mut self, ui: &mut egui::Ui) {
        let mut config = self.fft_config.lock().unwrap();

        // Get nyquist limit once using the helper method
        let nyquist_limit = self.get_nyquist_limit();

        ui.label("Root frequency max:");
        ui.add(
            egui::Slider::new(&mut config.root_freq_max, 0.0..=nyquist_limit)
                .logarithmic(true)
        );

        // 6) Advanced Crosstalk controls
        ui.horizontal(|ui| {
            let mut fft_config = self.fft_config.lock().unwrap();
            
            if fft_config.crosstalk_enabled {
                ui.label("Root Freq Min:");
                ui.add(egui::Slider::new(&mut fft_config.root_freq_min, 0.0..=nyquist_limit as f32));

                ui.label("Root Freq Max (Advanced):");
                ui.add(
                    egui::Slider::new(&mut fft_config.root_freq_max, 0.0..=nyquist_limit as f32)
                        .logarithmic(true)
                );
            }
        });

        ui.checkbox(&mut self.show_line_plot, "Show FFT");
        ui.checkbox(&mut self.show_spectrograph, "Show Spectrograph");
        ui.checkbox(&mut self.show_results, "Show Results");
    }
}

// This section is protected. Do not alter unless permission is requested by you and granted by me.
// Implementing eframe::App for MyApp

impl eframe::App for MyApp {
    fn on_close_event(&mut self) -> bool {
        info!("Closing GUI window...");
        self.shutdown_flag.store(true, Ordering::SeqCst);
        true
    }

    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        // Throttling: Limit repaint to at most 10 times per second (every 100 ms)
        let now = Instant::now();
        if now.duration_since(self.last_repaint) >= Duration::from_millis(100) {
            ctx.request_repaint();
            self.last_repaint = now;
        }

        // Force continuous updates every 100 ms
        ctx.request_repaint_after(Duration::from_millis(16)); // ~60 FPS

        ctx.set_visuals(egui::Visuals::dark());

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label("First Twelve Partials per Channel");

            let mut size_changed = false;

            // 1) First row of sliders
            {
                let mut fft_config = self.fft_config.lock().unwrap();
                ui.horizontal(|ui| {
                    ui.label("Min Frequency:");
                    ui.add(egui::Slider::new(&mut fft_config.min_frequency, MIN_FREQ..=200.0).text("Hz"));
                    
                    ui.label("Max Frequency:");
                    let buffer_size = *self.buffer_size.lock().unwrap();
                    let nyquist_limit = (buffer_size as f64 / 2.0).min(MAX_FREQ);
                    ui.add(egui::Slider::new(&mut fft_config.max_frequency, 0.0..=nyquist_limit).text("Hz"));

                    ui.label("Magnitude Threshold:");
                    ui.add(egui::Slider::new(&mut fft_config.magnitude_threshold, 0.0..=60.0));
                });
            }

            // 2) Second row with frames, buffer size, and averaging
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
                    ui.add(egui::Slider::new(&mut fft_config.min_freq_spacing, 0.0..=500.0).text("Hz"));
                }
                
                // Window Type section
                {
                    let mut fft_config = self.fft_config.lock().unwrap();
                    ui.label("Window Type:");
                    egui::ComboBox::from_id_source("window_type")
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
                        
                        // Kaiser beta slider appears if Kaiser window is selected
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

            // 3) Sliders for Y scale, alpha, bar width
            ui.horizontal(|ui| {
                ui.label("Y Max:");
                ui.add(egui::Slider::new(&mut self.y_scale, 0.0..=100.0).text(""));
                ui.label("Alpha:");
                ui.add(egui::Slider::new(&mut self.alpha, 0..=255).text(""));
                ui.label("Bar Width:");
                ui.add(egui::Slider::new(&mut self.bar_width, 1.0..=10.0).text(""));
                
                ui.separator();
                
                // Show FFT checkbox moved from row 2 to here
                ui.checkbox(&mut self.show_line_plot, "Show FFT");
                ui.checkbox(&mut self.show_spectrograph, "Show Spectrograph");
                ui.checkbox(&mut self.show_results, "Show Results");
                ui.separator();
            });

            // 4) Volume and Smoothing row + Crosstalk checkbox + Frequency Scale
            ui.horizontal(|ui| {
                // Volume slider (exactly matching update rate slider pattern)
                ui.label("Volume:");
                if let Ok(mut resynth_config) = self.resynth_config.lock() {
                    ui.add(
                        egui::Slider::new(&mut resynth_config.gain, 0.0..=1.0)
                            .text(""),
                    );
                }
             
            
                ui.separator();
                
                // Add Frequency Scale control right after Volume
                ui.label("Freq Scale:");
                if let Ok(mut resynth_config) = self.resynth_config.lock() {
                    // Create a custom widget with up/down buttons for octave changes
                    ui.horizontal(|ui| {
                        // Down button (divide by 2 = one octave down)
                        if ui.button("▼").clicked() {
                            resynth_config.freq_scale /= 2.0;
                        }
                        
                        // Display current value with 2 decimal places in a compact field
                        let mut freq_text = format!("{:.2}", resynth_config.freq_scale);
                        if ui.add(egui::TextEdit::singleline(&mut freq_text)
                            .desired_width(50.0)  // Make the text field narrower
                            .hint_text("1.00"))   // Add hint text
                            .changed() 
                        {
                            // Try to parse the new value
                            if let Ok(new_value) = freq_text.parse::<f32>() {
                                if new_value > 0.0 {  // Ensure positive value
                                    resynth_config.freq_scale = new_value;
                                }
                            }
                        }
                        
                        // Up button (multiply by 2 = one octave up)
                        if ui.button("▲").clicked() {
                            resynth_config.freq_scale *= 2.0;
                        }
                    });
                }
                
                ui.separator();
                
                // Smoothing slider with new minimum of 0
                ui.label("Smoothing:");
                if let Ok(mut resynth_config) = self.resynth_config.lock() {
                    ui.add(egui::Slider::new(&mut resynth_config.smoothing, 0.0..=0.9999));
                }
                
                ui.separator();
                
                // Move Crosstalk checkbox to this row
                let mut fft_config = self.fft_config.lock().unwrap();
                ui.checkbox(&mut fft_config.crosstalk_enabled, "Crosstalk Filtering");
            });

            // 5) Crosstalk basic parameters row + Harmonic Tolerance
            ui.horizontal(|ui| {
                let mut fft_config = self.fft_config.lock().unwrap();
                
                // Only show crosstalk parameters if enabled
                if fft_config.crosstalk_enabled {
                    ui.label("Threshold:");
                    ui.add(egui::Slider::new(&mut fft_config.crosstalk_threshold, 0.0..=1.0));
                    ui.label("Reduction:");
                    ui.add(egui::Slider::new(&mut fft_config.crosstalk_reduction, 0.0..=1.0));
                    
                    // Move Harmonic Tolerance to this row
                    ui.separator();
                    ui.label("Harmonic Tolerance:");
                    ui.add(egui::Slider::new(&mut fft_config.harmonic_tolerance, 0.01..=0.10)
                        .logarithmic(true));
                }
            });

            // 6) Advanced Crosstalk controls - only show if crosstalk is enabled
            ui.horizontal(|ui| {
                let mut fft_config = self.fft_config.lock().unwrap();
                
                // Use the helper method
                let nyquist_limit = self.get_nyquist_limit();
                
                // Only show advanced parameters if enabled
                if fft_config.crosstalk_enabled {
                    ui.label("Root Freq Min:");
                    ui.add(egui::Slider::new(&mut fft_config.root_freq_min, 0.0..=nyquist_limit as f32));
                    
                    ui.label("Root Freq Max (Advanced):");
                    ui.add(
                        egui::Slider::new(&mut fft_config.root_freq_max, 0.0..=nyquist_limit as f32)
                            .logarithmic(true)
                    );
                    
                    ui.label("Frequency Match:");
                    ui.add(egui::Slider::new(&mut fft_config.freq_match_distance, 1.0..=20.0));
                }
            });

            // 7) Resynth update timer control
            ui.horizontal(|ui| {
                if let Ok(mut resynth_config) = self.resynth_config.lock() {
                    ui.label("Resynth Update Rate:");
                    ui.add(
                        egui::Slider::new(&mut resynth_config.update_rate, 0.01..=60.0)
                            .text("seconds")
                            .logarithmic(true)
                    );
                }
            }); 

            // Handle max frequency adjustment if buffer size changed
            if size_changed {
                let nyquist_limit = self.get_nyquist_limit() as f64;
                let mut fft_config = self.fft_config.lock().unwrap();
                
                // Update max_frequency if needed
                if fft_config.max_frequency > nyquist_limit {
                    fft_config.max_frequency = nyquist_limit;
                    
                    // Also update root_freq_max to match the same nyquist limit
                    if fft_config.root_freq_max > nyquist_limit as f32 {
                        fft_config.root_freq_max = nyquist_limit as f32;
                    }
                }
            }

            // 8) Reset button
            if ui.button("Reset to Defaults").clicked() {
                {
                    let mut fft_config = self.fft_config.lock().unwrap();
                    let old_frames = fft_config.frames_per_buffer;
                    
                    // Reset FFT config
                    fft_config.min_frequency = MIN_FREQ;
                    fft_config.max_frequency = 8192.0;
                    fft_config.magnitude_threshold = 6.0;
                    fft_config.min_freq_spacing = 20.0;
                    fft_config.window_type = WindowType::Hanning;
                    
                    // Reset crosstalk settings
                    fft_config.crosstalk_enabled = false;
                    fft_config.crosstalk_threshold = 0.5;
                    fft_config.crosstalk_reduction = 0.5;
                    fft_config.harmonic_tolerance = 0.03;
                    fft_config.root_freq_min = 20.0;
                    fft_config.root_freq_max = DEFAULT_BUFFER_SIZE as f32 / 4.0;
                    fft_config.freq_match_distance = 5.0;
                    
                    // Only change frames_per_buffer if platform requires it
                    let new_frames = if cfg!(target_os = "linux") {
                        1024  // Larger buffer for Linux stability
                    } else {
                        512   // Default for other platforms
                    };
                    
                    if old_frames != new_frames {
                        fft_config.frames_per_buffer = new_frames;
                    }
                }
                
                // Reset resynth config
                {
                    let mut resynth_config = self.resynth_config.lock().unwrap();
                    resynth_config.gain = 0.25;
                    resynth_config.smoothing = 0.0;
                    resynth_config.freq_scale = 1.0;
                    resynth_config.update_rate = DEFAULT_UPDATE_RATE;
                }
                
                // Reset display settings
                self.y_scale = 80.0;
                self.alpha = 255;
                self.bar_width = 5.0;
                self.show_line_plot = false;
                self.show_spectrograph = false;
                
                // Reset buffer size
                if *self.buffer_size.lock().unwrap() != DEFAULT_BUFFER_SIZE {
                    self.update_buffer_size(DEFAULT_BUFFER_SIZE);
                }
            }

            // 8) Plot logic
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
                let spectrum = self.spectrum.lock().unwrap();
                let fft_data = spectrum.get_fft_line_data();
                
                fft_data.iter().enumerate().map(|(channel, points)| {
                    let color = self.colors[channel % self.colors.len()]
                        .linear_multiply(self.alpha as f32 / 255.0);

                    egui::plot::Line::new(
                        points.iter()
                            .map(|&(freq, mag)| [freq as f64, mag as f64])
                            .collect::<Vec<[f64; 2]>>()
                    )
                    .color(color)
                }).collect()
            } else {
                Vec::new()
            };

            let max_freq = {
                let fft = self.fft_config.lock().unwrap();
                fft.max_frequency
            };

            // Move the style modification outside the closure to avoid borrowing conflicts
            ui.style_mut().text_styles.insert(
                TextStyle::Monospace,
                FontId::new(14.0, FontFamily::Proportional)
            );
            ui.style_mut().text_styles.insert(
                TextStyle::Body,
                FontId::new(14.0, FontFamily::Proportional)
            );

            Plot::new("spectrum_plot")
                .legend(Legend::default())
                .view_aspect(6.0)
                .include_x(0.0)
                .include_x(max_freq as f64)
                .include_y(0.0)
                .include_y(self.y_scale as f64)
                .x_axis_formatter(|value, _range| format!("{} Hz", value as i32))
                .y_axis_formatter(|value, _range| format!("{} dB", value as i32))
                .y_grid_spacer(uniform_grid_spacer(|_input| [5.0, 10.0, 20.0]))  // More frequent grid lines
                .show_axes([true, true])
                .show_x(true)
                .show_y(true)
                .allow_drag(false)
                .allow_zoom(false)
                .allow_scroll(false)
                .allow_boxed_zoom(false)
                .allow_double_click_reset(false)
                .label_formatter(|name, value| {
                    if !name.is_empty() {
                        format!("{}: {:.1} Hz, {:.1} dB", name, value.x, value.y)
                    } else {
                        String::new()
                    }
                })
                .show(ui, |plot_ui| {
                    // 1. Clone the current style so we can mutate safely (compatible with older egui versions)
                    let mut style = (*plot_ui.ctx().style()).clone();

                    // 2. Adjust the desired text style (Body)
                    if let Some(body_style) = style.text_styles.get_mut(&TextStyle::Body) {
                        *body_style = FontId::new(14.0, FontFamily::Proportional);
                    }

                    // 3. Override the main text color (axis labels, etc.) with pure white for maximum contrast
                    style.visuals.override_text_color = Some(Color32::WHITE);
                    
                    // 4. Make sure widget foreground strokes are also bright for readability
                    style.visuals.widgets.noninteractive.fg_stroke.color = Color32::WHITE;
                    style.visuals.widgets.inactive.fg_stroke.color = Color32::WHITE;
                    style.visuals.widgets.hovered.fg_stroke.color = Color32::WHITE;
                    style.visuals.widgets.active.fg_stroke.color = Color32::WHITE;
                    style.visuals.widgets.open.fg_stroke.color = Color32::WHITE;
                    
                    // 5. Apply the modified style back to the context
                    plot_ui.ctx().set_style(style);
                    
                    // Plot the data using the new style
                    for bar_chart in all_bar_charts {
                        plot_ui.bar_chart(bar_chart);
                    }
                    if self.show_line_plot {
                        for line in all_line_plots {
                            plot_ui.line(line);
                        }
                    }
                });

            // Optimized spectrograph update logic
            if self.show_spectrograph {
                let current_time = Instant::now();
                let current_timestamp = chrono::Local::now();
                let elapsed = current_time.duration_since(self.start_time.as_ref().clone());
                let start_timestamp = current_timestamp - chrono::Duration::from_std(elapsed).unwrap_or_default();
                
                // Get the oldest timestamp from the history
                let (earliest_time, latest_time) = {
                    let history = self.spectrograph_history.lock().unwrap();
                    if let Some(oldest) = history.front() {
                        let oldest_time = oldest.time;
                        (oldest_time, oldest_time + 5.0) // Show 5 seconds from oldest timestamp
                    } else {
                        let current = elapsed.as_secs_f64();
                        (current, current + 5.0)
                    }
                };
                
                let max_freq = {
                    let fft = self.fft_config.lock().unwrap();
                    let buffer_size = *self.buffer_size.lock().unwrap();
                    (fft.max_frequency as f32).min(buffer_size as f32 / 2.0)
                };

                Plot::new("spectrograph_plot")
                    .legend(Legend::default())
                    .view_aspect(6.0)
                    .include_y(0.0)
                    .include_y(max_freq as f64)
                    .x_axis_formatter(move |value, _range| {
                        let timestamp = start_timestamp + chrono::Duration::milliseconds((value * 1000.0) as i64);
                        format!("{}", timestamp.format("%H:%M:%S"))
                    })
                    .y_axis_formatter(|value, _range| format!("{} Hz", value as i32))
                    .show_axes([true, true])
                    .show_x(true)
                    .show_y(true)
                    .allow_drag(false)
                    .allow_zoom(false)
                    .allow_scroll(false)
                    .allow_boxed_zoom(false)
                    .allow_double_click_reset(false)
                    .label_formatter(|name, value| {
                        if !name.is_empty() {
                            format!("{}: {:.1} Hz, {:.1} s", name, value.y, value.x)
                        } else {
                            String::new()
                        }
                    })
                    .show(ui, |plot_ui| {
                        let history = self.spectrograph_history.lock().unwrap();
                        if !history.is_empty() {
                            plot_ui.set_plot_bounds(egui::plot::PlotBounds::from_min_max(
                                [earliest_time * 1000.0, 0.0],
                                [latest_time * 1000.0, max_freq as f64]
                            ));

                            for slice in history.iter() {
                                if slice.time >= earliest_time && slice.time <= latest_time {
                                    for &(freq, magnitude) in &slice.data {
                                        let normalized_magnitude = (magnitude / self.y_scale).clamp(0.0, 1.0);
                                        let color = egui::Color32::from_rgb(
                                            (255.0 * normalized_magnitude) as u8,
                                            (255.0 * (1.0 - (normalized_magnitude - 0.5).abs() * 2.0)) as u8,
                                            (255.0 * (1.0 - normalized_magnitude)) as u8,
                                        );
                                        plot_ui.points(
                                            egui::plot::Points::new(vec![[slice.time * 1000.0, freq as f64]])
                                                .color(color)
                                                .radius(2.0)
                                        );
                                    }
                                }
                            }
                        }
                    });
            }

            egui::ScrollArea::vertical().show(ui, |ui| {
                if self.show_results {
                    let display = SpectralDisplay::new(&absolute_values);
                    for line in display.format_all() {
                        ui.label(egui::RichText::new(line).size(8.0));
                    }
                }
            });
        });

        // === Dynamic window resizing ===
        // Base height when only control panel is visible (no results text)
        let mut desired_height = 360.0;

        // Add extra space when optional panels are enabled
        if self.show_spectrograph {
            desired_height += 170.0; // Spectrograph plot
        }
        if self.show_results {
            desired_height += 160.0; // Text results panel (slightly more for padding)
        }

        // Only adjust height, preserve the original width from initial window setup
        let current_height = frame.info().window_info.size.y;
        
        // Apply resize if height difference is significant (avoid loop jitter)
        if (desired_height - current_height).abs() > 4.0 {
            // Use set_window_size but keep original width (from NativeOptions)
            let width = 1024.0; //frame.info().window_info.size.x;
            frame.set_window_size(egui::vec2(width, desired_height));
        }
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

// Update the format_all method in display.rs to use the configured number of partials
pub mod display_utils {
    // This helper function formats partials with any number of partials
    pub fn format_partials(values: &Vec<(f32, f32)>, num_partials: usize) -> String {
        // Format exactly num_partials values
        let magnitudes = (0..num_partials)
            .map(|i| {
                if i < values.len() {
                    let (freq, raw_val) = values[i];
                    if raw_val > 0.0 {
                        format!("({:.2}, {:.0})", freq, raw_val)
                    } else {
                        "(0.00, 0)".to_string()
                    }
                } else {
                    "(0.00, 0)".to_string()
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        
        magnitudes
    }
}
