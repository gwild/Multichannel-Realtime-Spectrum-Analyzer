fn main() {
    // Add system library paths
    println!("cargo:rustc-link-search=native=/usr/lib");
    println!("cargo:rustc-link-search=native=/usr/lib/aarch64-linux-gnu");
    println!("cargo:rustc-link-search=native=/usr/local/lib");
    
    // Link against system libraries for Raspberry Pi
    println!("cargo:rustc-link-lib=jack");
    println!("cargo:rustc-link-lib=asound");
    println!("cargo:rustc-link-lib=static=portaudio");
    
    // Set rpath to ensure libraries can be found at runtime
    println!("cargo:rustc-link-arg=-Wl,-rpath=/usr/lib/aarch64-linux-gnu");
} 