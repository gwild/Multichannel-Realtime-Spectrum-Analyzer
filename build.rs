fn main() {
    // Disable JACK support in PortAudio
    println!("cargo:rustc-env=PA_DISABLE_JACK=1");
    
    // Link against ALSA
    println!("cargo:rustc-link-lib=asound");
    
    // Link against PortAudio
    println!("cargo:rustc-link-lib=portaudio");
} 