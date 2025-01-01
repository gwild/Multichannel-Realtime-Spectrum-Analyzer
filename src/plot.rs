// This section is protected. Do not alter unless permission is requested by you and granted by me.
use std::sync::{Arc, Mutex};
use eframe::egui;
use egui::plot::{Plot, BarChart};
pub use eframe::NativeOptions;
use crate::fft_analysis::FFTConfig;
use crate::audio_stream::CircularBuffer;
use log::info;
use std::sync::atomic::{AtomicBool, Ordering};// Importing necessary types for GUI throttling.
// Reminder: Added to implement GUI throttling. Do not modify without permission.
use std::time::{Duration, Instant};
use std::sync::RwLock;
use crate::utils::{MIN_FREQ, MAX_FREQ, MIN_BUFFER_SIZE, MAX_BUFFER_SIZE, calculate_optimal_buffer_size};


// This section is protected. Do not alter unless permission is requested by you and granted by me.
pub struct SpectrumApp {
    pub partials: Vec<Vec<(f32, f32)>>, // Frequency, amplitude pairs for partials
}

// This section is protected. Do not alter unless permission is requested by you and granted by me.
impl SpectrumApp {
    pub fn new(num_channels: usize) -> Self {
        SpectrumApp {
            partials: vec![vec![(0.0, 0.0); 12]; num_channels],
        }
    }

    pub fn update_partials(&mut self, new_partials: Vec<Vec<(f32, f32)>>) {
        for (channel, data) in new_partials.into_iter().enumerate() {
            if channel < self.partials.len() {
                self.partials[channel] = data;
                // info!("Updated partials for channel {}: {:?}", channel + 1, log_data);
            }
        }
    }
}

// This section is protected. Do not alter unless permission is requested by you and granted by me.
pub struct MyApp {
    pub spectrum: Arc<Mutex<SpectrumApp>>,
    pub fft_config: Arc<Mutex<FFTConfig>>,
    pub buffer_size: Arc<Mutex<usize>>,
    pub audio_buffer: Arc<RwLock<CircularBuffer>>,
    colors: Vec<egui::Color32>,
    y_scale: f32,
    alpha: u8,
    bar_width: f32,

    // Throttling: Added to track the last repaint time
    last_repaint: Instant, // Reminder: This field was added to implement GUI throttling. Do not modify without permission.
    shutdown_flag: Arc<AtomicBool>,  // Add shutdown flag
}

// This section is protected. Do not alter unless permission is requested by you and granted by me.
impl MyApp {
    pub fn new(
        spectrum: Arc<Mutex<SpectrumApp>>,
        fft_config: Arc<Mutex<FFTConfig>>,
        buffer_size: Arc<Mutex<usize>>,
        audio_buffer: Arc<RwLock<CircularBuffer>>,
        shutdown_flag: Arc<AtomicBool>,  // Accept shutdown flag
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

        // Create the MyApp instance as before (NO alterations/deletions)
        let instance = MyApp {
            spectrum,
            fft_config,
            buffer_size,
            audio_buffer,  // Correct field name
            colors,
            y_scale: 80.0,
            alpha: 255,
            bar_width: 5.0,
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
                info!("Clamped max frequency at startup to {}", cfg.max_frequency);
            }
        }

        // Return the newly created instance with the fix
        instance
    }

    pub fn update_buffer_size(&mut self, new_size: usize) {
        let validated_size = new_size
            .next_power_of_two()  // Ensure power of 2 for FFT
            .clamp(MIN_BUFFER_SIZE, MAX_BUFFER_SIZE);
        
        if validated_size != new_size {
            info!(
                "Adjusted requested buffer size from {} to {} (power of 2 between {} and {})",
                new_size, validated_size, MIN_BUFFER_SIZE, MAX_BUFFER_SIZE
            );
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

            // 1) Sliders for min freq, max freq, db threshold
            {
                let mut fft_config = self.fft_config.lock().unwrap();
                ui.horizontal(|ui| {
                    ui.label("Min Frequency:");
                    ui.add(egui::Slider::new(&mut fft_config.min_frequency, 10.0..=200.0).text("Hz"));
                    
                    ui.label("Max Frequency:");
                    let buffer_size = *self.buffer_size.lock().unwrap();
                    let nyquist_limit = (buffer_size as f64 / 2.0).min(20000.0);
                    ui.add(egui::Slider::new(&mut fft_config.max_frequency, 0.0..=nyquist_limit).text("Hz"));

                    ui.label("DB Threshold:");
                    ui.add(egui::Slider::new(&mut fft_config.db_threshold, -80.0..=-10.0).text("dB"));
                });
            }

            // 2) Buffer size slider (no immediate clamp of max_frequency inside)
            let size_changed = false;
            let buffer_size = *self.buffer_size.lock().unwrap();
            let mut buffer_log_slider = (buffer_size as f32).log2().round() as u32;
            ui.horizontal(|ui| {
                ui.label("Buffer Size:");
                let min_power = (MIN_BUFFER_SIZE as f32).log2() as u32;
                let max_power = (MAX_BUFFER_SIZE as f32).log2() as u32;
                if ui
                    .add(egui::Slider::new(&mut buffer_log_slider, min_power..=max_power).text("Power of 2"))
                    .changed()
                {
                    let new_size = 1 << buffer_log_slider;
                    self.update_buffer_size(new_size);
                }
                ui.label(format!("{} samples", buffer_size));
            });

            // 3) Sliders for Y scale, alpha, bar width
            ui.horizontal(|ui| {
                ui.label("Y Max:");
                ui.add(egui::Slider::new(&mut self.y_scale, 0.0..=100.0).text("dB"));
                ui.label("Alpha:");
                ui.add(egui::Slider::new(&mut self.alpha, 0..=255).text(""));
                ui.label("Bar Width:");
                ui.add(egui::Slider::new(&mut self.bar_width, 1.0..=10.0).text(""));
            });

            // 4) Reset button
            let mut reset_clicked = false;
            if ui.button("Reset to Defaults").clicked() {
                {
                    let mut fft_config = self.fft_config.lock().unwrap();
                    fft_config.min_frequency = MIN_FREQ;
                    fft_config.max_frequency = MAX_FREQ;
                    fft_config.db_threshold = -16.0;
                }
                
                self.y_scale = 80.0;
                self.alpha = 255;
                self.bar_width = 5.0;
                
                // Calculate optimal size based on current sample rate
                let sample_rate = 48000.0f64;
                let optimal_size = calculate_optimal_buffer_size(sample_rate);
                self.update_buffer_size(optimal_size);
                
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
            let partials = {
                let spectrum = self.spectrum.lock().unwrap();
                spectrum.partials.clone()
            };

            let all_bar_charts: Vec<BarChart> = partials
                .iter()
                .enumerate()
                .map(|(channel, channel_partials)| {
                    let bars: Vec<egui::plot::Bar> = channel_partials
                        .iter()
                        .map(|&(freq, amp)| {
                            egui::plot::Bar::new(freq as f64, amp as f64)
                                .width(self.bar_width as f64)
                        })
                        .collect();

                    let color = self.colors[channel % self.colors.len()]
                        .linear_multiply(self.alpha as f32 / 255.0);

                    BarChart::new(bars)
                        .name(format!("Channel {}", channel + 1))
                        .color(color)
                })
                .collect();

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
                .include_y(self.y_scale)
                .show(ui, |plot_ui| {
                    for bar_chart in all_bar_charts {
                        plot_ui.bar_chart(bar_chart);
                    }
                });

            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.label("Channel Results:");
                for (channel, channel_partials) in partials.iter().enumerate() {
                    let formatted_partials: Vec<String> = channel_partials
                        .iter()
                        .map(|&(freq, amp)| format!("({:.2}, {:.0})", freq, amp))
                        .collect();
                    ui.label(format!(
                        "Channel {}: [{}]",
                        channel + 1,
                        formatted_partials.join(", ")
                    ));
                }
            });
        });
    }
}

// This section is protected. Do not alter unless permission is requested by you and granted by me.
pub fn run_native(
    app_name: &str,
    native_options: NativeOptions,
    app_creator: Box<dyn FnOnce(&eframe::CreationContext<'_>) -> Box<MyApp>>,  // Expect Box<MyApp>
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
