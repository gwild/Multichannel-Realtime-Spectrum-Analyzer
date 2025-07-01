# Multichannel Realtime Spectrum Analyzer

A high-performance, real-time audio spectrum analyzer written in Rust with a modern GUI built using egui. This application provides real-time FFT analysis of audio input with support for both single and multi-channel audio. Features include advanced spectral analysis, partial detection, and audio resynthesis.

## Features

### Core Functionality
- **Real-time FFT Analysis**: Live spectral analysis with configurable sample rates (44.1kHz to 192kHz)
- **Single & Multi-channel Support**: Works with mono, stereo, or multi-channel audio input
- **Advanced Partial Detection**: Extract and track harmonic partials with configurable sensitivity
- **Crosstalk Filtering**: Reduce interference between channels with intelligent frequency domain filtering (multi-channel only)
- **Audio Resynthesis**: Real-time audio output based on detected partials
- **Shared Memory Interface**: Export spectral data for external applications (Python integration)

### Visualization
- **Real-time Spectrograph**: Waterfall display with configurable history depth
- **Bar Chart Display**: Real-time frequency magnitude visualization
- **Line Plot Mode**: Continuous frequency response curves
- **Multi-channel Color Coding**: Distinct colors for each audio channel
- **Configurable Display**: Adjustable Y-scale, transparency, and bar width

### Advanced Features
- **Preset Management**: Save and load analysis configurations
- **Configurable Windows**: Multiple FFT window functions (Hanning, Hamming, Blackman-Harris, Flat-top, Kaiser)
- **Harmonic Analysis**: Automatic detection and tracking of harmonic relationships
- **Gain Control**: Adjustable signal amplification
- **Buffer Size Optimization**: Dynamic buffer sizing for optimal performance
- **Logging System**: Comprehensive debug and info logging

## System Requirements

### Supported Platforms
- **Linux** (Primary support)
- **macOS** (Limited support)
- **Windows**: Not supported

### Hardware Requirements
- **Audio Interface**: Any audio input device (microphone, line-in, or multi-channel interface)
- **CPU**: Multi-core processor recommended for real-time processing
- **Memory**: 4GB RAM minimum, 8GB+ recommended
- **Storage**: 100MB free space

### Software Dependencies
- **Rust**: 1.65.0 or later
- **PortAudio**: Audio I/O library
- **ALSA/PulseAudio**: Linux audio backend
- **Core Audio**: macOS audio backend

## Installation

### Prerequisites

1. **Install Rust** (if not already installed):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source ~/.cargo/env
   ```

2. **Install PortAudio development libraries**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install portaudio19-dev
   
   # Fedora/RHEL
   sudo dnf install portaudio-devel
   
   # Arch Linux
   sudo pacman -S portaudio
   ```

### Building from Source

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Multichannel-Realtime-Spectrum-Analyzer.git
   cd Multichannel-Realtime-Spectrum-Analyzer
   ```

2. **Build the project**:
   ```bash
   cargo build --release
   ```

3. **Run the application**:
   ```bash
   ./target/release/audio_streaming
   ```

## Usage

### Basic Usage

```bash
# Run with default settings
./target/release/audio_streaming

# Run with default settings (single channel)
./target/release/audio_streaming

# Specify input device
./target/release/audio_streaming --input-device 0

# Specify sample rate
./target/release/audio_streaming --input-rate 48000

# Enable debug logging
./target/release/audio_streaming --debug
```

### Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `-i, --input-device` | Input device index | `--input-device 0` |
| `-o, --output-device` | Output device index | `--output-device 1` |
| `--input-rate` | Input sample rate (Hz) | `--input-rate 48000` |
| `--output-rate` | Output sample rate (Hz) | `--output-rate 44100` |
| `-c, --channels` | Input channels (comma-separated, optional for multi-channel) | `--channels "0,1,2"` |
| `-p, --num-partials` | Number of partials to detect | `--num-partials 16` |
| `--info` | Enable info logging | `--info` |
| `--debug` | Enable debug logging | `--debug` |

### GUI Controls

#### Main Display
- **Spectrograph**: Real-time waterfall display of frequency content
- **Bar Chart**: Magnitude bars for detected partials
- **Line Plot**: Continuous frequency response curves

#### Configuration Panel
- **FFT Settings**: Buffer size, window type, frequency range
- **Partial Detection**: Number of partials, magnitude threshold
- **Crosstalk Filtering**: Enable/disable, threshold, reduction factor
- **Display Options**: Y-scale, transparency, bar width
- **Preset Management**: Save, load, and manage analysis configurations

#### Audio Settings
- **Device Selection**: Input/output device configuration
- **Sample Rate**: Configurable sample rates
- **Channel Mapping**: Single or multi-channel input configuration
- **Gain Control**: Signal amplification settings

## Configuration

### Preset System

The application uses a YAML-based preset system for saving and loading configurations:

```yaml
# Example preset configuration
default:
  fft:
    buffer_size: 8192
    window_type: "Hanning"
    min_frequency: 20.0
    max_frequency: 20000.0
    magnitude_threshold: 6.0
    num_partials: 12
  crosstalk:
    enabled: false
    threshold: 0.3
    reduction: 0.5
  display:
    y_scale: 80.0
    alpha: 255
    bar_width: 5.0
```

### Shared Memory Interface

For external applications, spectral data is exported via shared memory:

```python
# Python example for reading spectral data
import mmap
import struct

# Open shared memory file
with open('/tmp/spectrum_data', 'r+b') as f:
    mm = mmap.mmap(f.fileno(), 0)
    
    # Read frequency and magnitude pairs
    data = []
    for i in range(0, len(mm), 8):  # 4 bytes freq + 4 bytes magnitude
        freq = struct.unpack('f', mm[i:i+4])[0]
        mag = struct.unpack('f', mm[i+4:i+8])[0]
        data.append((freq, mag))
```

## Performance Optimization

### Buffer Sizing
- **Small buffers** (512-2048): Lower latency, higher CPU usage
- **Large buffers** (4096-65536): Higher latency, lower CPU usage
- **Optimal range**: 2048-8192 for most applications

### Multi-threading
- FFT processing runs on dedicated threads
- GUI updates are throttled for smooth performance
- Audio I/O uses separate threads for input/output

### Memory Management
- Circular buffers prevent memory leaks
- Shared memory interface for efficient data sharing
- Automatic cleanup of unused resources

## Troubleshooting

### Common Issues

1. **No Audio Input**:
   - Check device permissions
   - Verify PortAudio installation
   - Test with `--debug` flag for detailed logs

2. **High CPU Usage**:
   - Increase buffer size
   - Reduce number of partials
   - Disable crosstalk filtering

3. **Audio Dropouts**:
   - Decrease buffer size
   - Check system audio settings
   - Verify sample rate compatibility

4. **GUI Performance**:
   - Reduce spectrograph history
   - Disable line plot mode
   - Adjust display update rate

### Debug Mode

Enable comprehensive logging for troubleshooting:

```bash
./target/release/audio_streaming --debug --info
```

## Development

### Project Structure

```
src/
├── main.rs              # Application entry point and CLI
├── audio_stream.rs      # Audio I/O and buffer management
├── fft_analysis.rs      # FFT processing and partial detection
├── plot.rs              # GUI rendering and visualization
├── display.rs           # Display formatting utilities
├── resynth.rs           # Audio resynthesis engine
├── get_results.rs       # Results processing and export
└── presets.rs           # Preset management system
```

### Building for Development

```bash
# Debug build with logging
cargo build
RUST_LOG=debug cargo run

# Release build
cargo build --release

# Run tests
cargo test
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Acknowledgments

- **PortAudio**: Cross-platform audio I/O library
- **egui**: Immediate mode GUI framework
- **realfft**: Real-valued FFT library
- **tokio**: Asynchronous runtime

## Support

For issues, questions, or contributions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review debug logs with `--debug` flag

---

**Note**: This application is designed for Linux systems and does not support Windows. macOS support is limited and may require additional configuration. 