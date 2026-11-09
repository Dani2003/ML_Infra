use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();
    let cuda_kernels_dir = workspace_root.join("cuda_kernels");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

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
        println!("cargo:rustc-link-lib=static=cuda_kernels");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-search=native={}", out_dir.display());

        if let Ok(cuda_path) = std::env::var("CUDA_HOME").or_else(|_| std::env::var("CUDA_PATH")) {
            println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
            println!("cargo:rustc-link-search=native={}/lib/x64", cuda_path);
            println!("cargo:rustc-link-search=native={}/lib", cuda_path);
        } else {
            println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
            println!("cargo:rustc-link-search=native=/usr/local/cuda/lib");
        }

        let lib_name = if cfg!(windows) { "cuda_kernels.lib" } else { "libcuda_kernels.a" };
        let mut cmd = std::process::Command::new(&nvcc_path);
        cmd.arg("-std=c++17")
            .arg("-DUSE_CUDA")
            .arg("-O3")
            .arg("--lib")
            .arg("-I")
            .arg(cuda_kernels_dir.join("include"))
            .arg(cuda_kernels_dir.join("src").join("memory_manager.cu"))
            .arg(cuda_kernels_dir.join("src").join("attention.cu"))
            .arg("-o")
            .arg(out_dir.join(lib_name));

        if !cfg!(windows) {
            cmd.arg("-Xcompiler").arg("-fPIC");
            println!("cargo:rustc-link-lib=stdc++");
        }

        let status = cmd.status().expect("failed to execute nvcc");
        if !status.success() {
            panic!("nvcc compilation failed with status {}", status);
        }
    } else {
        println!("cargo:warning=model-runner@0.1.0: NVCC not detected; building with CPU/portable reference kernels.");

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
        build.compile("cuda_kernels");
    }

    println!("cargo:rerun-if-changed={}", cuda_kernels_dir.join("include").display());
    println!("cargo:rerun-if-changed={}", cuda_kernels_dir.join("src").display());
}

