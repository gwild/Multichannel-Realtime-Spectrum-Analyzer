fn main() {
    // Link against ALSA
    println!("cargo:rustc-link-lib=asound");
    
    // Link against PortAudio
    println!("cargo:rustc-link-lib=portaudio");
} 