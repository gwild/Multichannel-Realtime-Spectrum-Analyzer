fn main() {
    // Explicitly disable JACK
    println!("cargo:rustc-cfg=pa_skip_jack");
    
    // Link against ALSA
    println!("cargo:rustc-link-lib=asound");
    
    // Link against PortAudio
    println!("cargo:rustc-link-lib=portaudio");
} 