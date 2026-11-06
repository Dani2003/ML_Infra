use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();
    let cuda_kernels_dir = workspace_root.join("cuda_kernels");

    let mut build = cc::Build::new();
    build.cpp(true);
    build.std("c++17");
    if build.get_compiler().is_like_msvc() {
        build.flag("/TP");
    }
    build.include(cuda_kernels_dir.join("include"));
    build.file(cuda_kernels_dir.join("src").join("memory_manager.cu"));
    build.file(cuda_kernels_dir.join("src").join("attention.cu"));

    // Check if CUDA / NVCC is available
    let has_nvcc = std::process::Command::new("nvcc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if has_nvcc {
        println!("cargo:rustc-cfg=feature=\"cuda\"");
        build.define("USE_CUDA", None);
        build.cuda(true);
    } else {
        println!("cargo:warning=NVCC not detected; building with CPU/portable reference kernels.");
    }

    build.compile("cuda_kernels");

    println!("cargo:rerun-if-changed={}", cuda_kernels_dir.join("include").display());
    println!("cargo:rerun-if-changed={}", cuda_kernels_dir.join("src").display());
}

