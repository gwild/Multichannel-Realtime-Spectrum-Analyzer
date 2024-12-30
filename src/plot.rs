// This section is protected. Do not alter unless permission is requested by you and granted by me.
use std::sync::{Arc, Mutex, RwLock};
use eframe::egui;
use egui::plot::{Plot, BarChart};
pub use eframe::NativeOptions;
use crate::fft_analysis::FFTConfig;
use crate::audio_stream::CircularBuffer;
use log::info;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

pub struct SpectrumApp {
    pub partials: Vec<Vec<(f32, f32)>>, // Frequency, amplitude pairs for partials
}

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
            }
        }
    }
}

pub struct MyApp {
    pub spectrum: Arc<Mutex<SpectrumApp>>,
    pub fft_config: Arc<Mutex<FFTConfig>>,
    pub buffer_size: Arc<Mutex<usize>>,
    pub audio_buffer: Arc<RwLock<CircularBuffer>>,  // Single shared buffer
    colors: Vec<egui::Color32>,
    y_scale: f32,
    alpha: u8,
    bar_width: f32,
    last_repaint: Instant,
    shutdown_flag: Arc<AtomicBool>,
}

impl MyApp {
    pub fn new(
        spectrum: Arc<Mutex<SpectrumApp>>,
        fft_config: Arc<Mutex<FFTConfig>>,
        buffer_size: Arc<Mutex<usize>>,
        audio_buffer: Arc<RwLock<CircularBuffer>>,  // Accept single buffer
        shutdown_flag: Arc<AtomicBool>,
    ) -> Self {
        let colors = vec![
            egui::Color32::from_rgb(0, 0, 255),
            egui::Color32::from_rgb(255, 165, 0),
            egui::Color32::from_rgb(0, 255, 0),
            egui::Color32::from_rgb(255, 0, 0),
        ];

        let instance = MyApp {
            spectrum,
            fft_config,
            buffer_size,
            audio_buffer,
            colors,
            y_scale: 80.0,
            alpha: 255,
            bar_width: 5.0,
            last_repaint: Instant::now(),
            shutdown_flag,
        };

        // Clamp max_frequency at startup based on buffer size
        {
            let buffer_s = instance.buffer_size.lock().unwrap();
            let nyquist_limit = (*buffer_s as f32 / 2.0).min(20000.0);
            let mut cfg = instance.fft_config.lock().unwrap();
            if cfg.max_frequency > nyquist_limit {
                cfg.max_frequency = nyquist_limit;
                info!("Clamped max frequency at startup to {}", cfg.max_frequency);
            }
        }

        instance
    }
}
impl eframe::App for MyApp {
    fn on_close_event(&mut self) -> bool {
        info!("Closing GUI window...");
        self.shutdown_flag.store(true, Ordering::Relaxed);
        true
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let now = Instant::now();
        if now.duration_since(self.last_repaint) >= Duration::from_millis(100) {
            ctx.request_repaint();
            self.last_repaint = now;
        }
        ctx.request_repaint_after(Duration::from_millis(100));
        ctx.set_visuals(egui::Visuals::dark());

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label("First Twelve Partials per Channel");

            // FFT Configuration Sliders
            {
                let mut fft_config = self.fft_config.lock().unwrap();
                ui.horizontal(|ui| {
                    ui.label("Min Frequency:");
                    ui.add(egui::Slider::new(&mut fft_config.min_frequency, 10.0..=200.0));
                    ui.label("Max Frequency:");

                    let nyquist_limit = (*self.buffer_size.lock().unwrap() as f32 / 2.0).min(20000.0);
                    ui.add(egui::Slider::new(&mut fft_config.max_frequency, 0.0..=nyquist_limit));
                    ui.label("DB Threshold:");
                    ui.add(egui::Slider::new(&mut fft_config.db_threshold, -80.0..=-10.0));
                });
            }

            // Dynamic Buffer Resizing
            let mut buffer_size = *self.buffer_size.lock().unwrap();
            let mut buffer_log_slider = (buffer_size as f32).log2().round() as u32;
            ui.horizontal(|ui| {
                ui.label("Buffer Size:");
                if ui.add(egui::Slider::new(&mut buffer_log_slider, 6..=14)).changed() {
                    buffer_size = 1 << buffer_log_slider;
                    *self.buffer_size.lock().unwrap() = buffer_size;

                    let mut buf = self.audio_buffer.write().unwrap();
                    buf.resize(buffer_size);  // Resize buffer dynamically
                }
                ui.label(format!("{} samples", buffer_size));
            });

            // Plot Partials
            let partials = {
                let spectrum = self.spectrum.lock().unwrap();
                spectrum.partials.clone()
            };

            let all_bar_charts: Vec<BarChart> = partials
                .iter()
                .enumerate()
                .map(|(channel, data)| {
                    let bars: Vec<_> = data
                        .iter()
                        .map(|&(freq, amp)| egui::plot::Bar::new(freq as f64, amp as f64).width(self.bar_width as f64))
                        .collect();

                    BarChart::new(bars).color(self.colors[channel % self.colors.len()])
                })
                .collect();

            Plot::new("spectrum_plot")
                .legend(egui::plot::Legend::default())
                .show(ui, |plot_ui| {
                    for chart in all_bar_charts {
                        plot_ui.bar_chart(chart);
                    }
                });
        });
    }
}
