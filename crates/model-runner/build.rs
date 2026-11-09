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
        build.flag("/EHsc");
    }
    build.include(cuda_kernels_dir.join("include"));
    build.file(cuda_kernels_dir.join("src").join("memory_manager.cu"));
    build.file(cuda_kernels_dir.join("src").join("attention.cu"));

    // Check if CUDA / NVCC is available
    let nvcc_cmd = if std::process::Command::new("nvcc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
    {
        Some("nvcc".to_string())
    } else if std::path::Path::new("/usr/local/cuda/bin/nvcc").exists() {
        Some("/usr/local/cuda/bin/nvcc".to_string())
    } else {
        None
    };

    if let Some(nvcc_path) = nvcc_cmd {
        println!("cargo:warning=model-runner: NVCC detected at {}; compiling CUDA kernels.", nvcc_path);
        println!("cargo:rustc-cfg=feature=\"cuda\"");
        println!("cargo:rustc-link-lib=cudart");

        if let Ok(cuda_path) = std::env::var("CUDA_HOME").or_else(|_| std::env::var("CUDA_PATH")) {
            println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
            println!("cargo:rustc-link-search=native={}/lib/x64", cuda_path);
            println!("cargo:rustc-link-search=native={}/lib", cuda_path);
        } else {
            println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
            println!("cargo:rustc-link-search=native=/usr/local/cuda/lib");
        }

        build.compiler(&nvcc_path);
        build.define("USE_CUDA", None);
        build.cuda(true);
    } else {
        println!("cargo:warning=model-runner@0.1.0: NVCC not detected; building with CPU/portable reference kernels.");
    }

    build.compile("cuda_kernels");

    println!("cargo:rerun-if-changed={}", cuda_kernels_dir.join("include").display());
    println!("cargo:rerun-if-changed={}", cuda_kernels_dir.join("src").display());
}

