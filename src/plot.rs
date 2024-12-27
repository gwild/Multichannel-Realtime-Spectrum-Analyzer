// This section is protected. Do not alter unless permission is requested by you and granted by me.
use std::sync::{Arc, Mutex};
use eframe::egui;
use egui::plot::{Plot, BarChart};
pub use eframe::NativeOptions;
use crate::fft_analysis::FFTConfig;
use crate::audio_stream::CircularBuffer;
use log::info;

// Importing necessary types for GUI throttling.
// Reminder: Added to implement GUI throttling. Do not modify without permission.
use std::time::{Duration, Instant};

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
                let log_data = data.clone();
                self.partials[channel] = data;
                info!("Updated partials for channel {}: {:?}", channel + 1, log_data);
            }
        }
    }
}

// This section is protected. Do not alter unless permission is requested by you and granted by me.
pub struct MyApp {
    pub spectrum: Arc<Mutex<SpectrumApp>>,
    pub fft_config: Arc<Mutex<FFTConfig>>,
    pub buffer_size: Arc<Mutex<usize>>,
    pub audio_buffers: Arc<Vec<Mutex<CircularBuffer>>>,
    colors: Vec<egui::Color32>,
    y_scale: f32,
    alpha: u8,
    bar_width: f32,

    // Throttling: Added to track the last repaint time
    last_repaint: Instant, // Reminder: This field was added to implement GUI throttling. Do not modify without permission.
}

// This section is protected. Do not alter unless permission is requested by you and granted by me.
impl MyApp {
    pub fn new(
        spectrum: Arc<Mutex<SpectrumApp>>,
        fft_config: Arc<Mutex<FFTConfig>>,
        buffer_size: Arc<Mutex<usize>>,
        audio_buffers: Arc<Vec<Mutex<CircularBuffer>>>,
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
        MyApp {
            spectrum,
            fft_config,
            buffer_size,
            audio_buffers,
            colors,
            y_scale: 80.0,
            alpha: 255,
            bar_width: 5.0,

            // Throttling: Initialize the last_repaint timer
            last_repaint: Instant::now(), // Reminder: Initialization of throttling timer added. Do not modify without permission.
        }
    }
}

// This section is protected. Do not alter unless permission is requested by you and granted by me.
// Implementing eframe::App for MyApp
impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Throttling: Limit repaint to at most 10 times per second (every 100 ms)
        let now = Instant::now();
        if now.duration_since(self.last_repaint) >= Duration::from_millis(100) {
            ctx.request_repaint();
            self.last_repaint = now;
        }

        // NEW LINE: force continuous updates every 100 ms, even without user interaction
        ctx.request_repaint_after(Duration::from_millis(100));

        ctx.set_visuals(egui::Visuals::dark());

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label("First Twelve Partials per Channel");

            let mut fft_config = self.fft_config.lock().unwrap();
            ui.horizontal(|ui| {
                ui.label("Min Frequency:");
                ui.add(egui::Slider::new(&mut fft_config.min_frequency, 10.0..=200.0).text("Hz"));
                ui.label("Max Frequency:");

                let nyquist_limit = (*self.buffer_size.lock().unwrap() as f32 / 2.0).min(20000.0);
                if ui
                    .add(egui::Slider::new(&mut fft_config.max_frequency, 500.0..=nyquist_limit).text("Hz"))
                    .changed()
                {
                    fft_config.max_frequency = fft_config.max_frequency.min(nyquist_limit);
                    info!("Max frequency adjusted to {}", fft_config.max_frequency);
                }

                ui.label("DB Threshold:");
                ui.add(egui::Slider::new(&mut fft_config.db_threshold, -80.0..=-10.0).text("dB"));
            });

            let mut buffer_size = *self.buffer_size.lock().unwrap();
            let mut buffer_log_slider = (buffer_size as f32).log2().round() as u32;
            ui.horizontal(|ui| {
                ui.label("Buffer Size:");
                if ui
                    .add(egui::Slider::new(&mut buffer_log_slider, 6..=14).text("Power of 2"))
                    .changed()
                {
                    buffer_size = 1 << buffer_log_slider;
                    *self.buffer_size.lock().unwrap() = buffer_size;

                    for buffer in self.audio_buffers.iter() {
                        let mut buf = buffer.lock().unwrap();
                        *buf = CircularBuffer::new(buffer_size);
                    }
                }
                ui.label(format!("{} samples", buffer_size));
            });

            ui.horizontal(|ui| {
                ui.label("Y Max:");
                ui.add(egui::Slider::new(&mut self.y_scale, 0.0..=100.0).text("dB"));
                ui.label("Alpha:");
                ui.add(egui::Slider::new(&mut self.alpha, 0..=255).text(""));
                ui.label("Bar Width:");
                ui.add(egui::Slider::new(&mut self.bar_width, 1.0..=10.0).text(""));
            });

            if ui.button("Reset to Defaults").clicked() {
                fft_config.min_frequency = 20.0;
                fft_config.max_frequency = 1000.0;
                fft_config.db_threshold = -32.0;
                self.y_scale = 80.0;
                self.alpha = 255;
                self.bar_width = 5.0;
                *self.buffer_size.lock().unwrap() = 8192;

                for buffer in self.audio_buffers.iter() {
                    let mut buf = buffer.lock().unwrap();
                    *buf = CircularBuffer::new(8192);
                }
            }

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

            Plot::new("spectrum_plot")
                .legend(egui::plot::Legend::default())
                .view_aspect(6.0)
                .include_x(0.0)
                .include_x(fft_config.max_frequency as f64)
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
                        .map(|&(freq, amp)| format!("({:.2}, {:.1})", freq, amp))
                        .collect();
                    ui.label(format!(
                        "Channel {}: [{}]",
                        channel + 1,
                        formatted_partials.join(", ")
                    ));
                }
            });
        }); // Closing the closure passed to CentralPanel::show
    } // Closing update method

} // <-- Single brace added here to fix unclosed delimiter

// This section is protected. Do not alter unless permission is requested by you and granted by me.
pub fn run_native(
    app_name: &str,
    native_options: NativeOptions,
    app_creator: Box<dyn FnOnce(&eframe::CreationContext<'_>) -> Box<dyn eframe::App>>,
) -> Result<(), eframe::Error> {
    eframe::run_native(app_name, native_options, app_creator)
}

// Total line count: 232
