fn main() {
    // Add system library paths
    println!("cargo:rustc-link-search=native=/usr/lib");
    println!("cargo:rustc-link-search=native=/usr/lib/aarch64-linux-gnu");
    println!("cargo:rustc-link-search=native=/usr/local/lib");

    // Link portaudio first (it references JACK)
    println!("cargo:rustc-link-lib=static=portaudio");
    // Then JACK
    println!("cargo:rustc-link-lib=jack");
    // Then ALSA
    println!("cargo:rustc-link-lib=asound");

    // Set rpath to ensure libraries can be found at runtime
    println!("cargo:rustc-link-arg=-Wl,-rpath=/usr/lib/aarch64-linux-gnu");
}