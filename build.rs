use std::{env, path::PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let llama_dir = manifest_dir.join("llama.cpp");
    let include_main = llama_dir.join("include");
    let include_ggml = llama_dir.join("ggml").join("include");
    let header = manifest_dir.join("llama_wrapper.h");

    println!("cargo:rerun-if-changed={}", header.display());
    println!(
        "cargo:rerun-if-changed={}",
        include_main.join("llama.h").display()
    );
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_LIBDIR");

    let lib_dir = env::var("LLAMA_CPP_LIBDIR")
        .map(PathBuf::from)
        .ok()
        .filter(|p| p.exists())
        .unwrap_or_else(|| llama_dir.join("build").join("bin"));

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=llama");
    println!("cargo:rustc-link-lib=dylib=ggml");
    println!("cargo:rustc-link-lib=dylib=ggml-base");
    println!("cargo:rustc-link-lib=dylib=ggml-cpu");
    println!("cargo:rustc-link-lib=dylib=ggml-cuda");
    println!("cargo:rustc-link-lib=dylib=mtmd");
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    }

    let builder = bindgen::Builder::default()
        .header(header.to_string_lossy())
        .clang_arg(format!("-I{}", include_main.display()))
        .clang_arg(format!("-I{}", include_ggml.display()))
        .clang_arg(format!("-I{}", llama_dir.display()))
        .allowlist_function("llama_.*")
        .allowlist_type("llama_.*")
        .allowlist_var("LLAMA_.*")
        .allowlist_type("ggml_.*")
        .allowlist_var("GGML_.*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));

    let bindings = builder
        .generate()
        .expect("Unable to generate llama.cpp bindings");
    let out_path = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    bindings
        .write_to_file(out_path.join("llama_bindings.rs"))
        .expect("Couldn't write bindings");
}
