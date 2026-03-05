use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const FBGEMM_REPO: &str = "https://github.com/pytorch/FBGEMM.git";
const FBGEMM_COMMIT: &str = "a01527eb943f9d7d5ec814978a05811bc5467a9b";

fn run(cmd: &mut Command) {
    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("failed to run {:?}: {}", cmd, e));
    assert!(status.success(), "command failed: {:?}", cmd);
}

fn clone_fbgemm(dest: &Path) {
    if dest.join("CMakeLists.txt").exists() {
        return;
    }

    std::fs::create_dir_all(dest).unwrap();
    run(Command::new("git").args(["init"]).current_dir(dest));
    run(Command::new("git")
        .args(["remote", "add", "origin", FBGEMM_REPO])
        .current_dir(dest));
    run(Command::new("git")
        .args(["fetch", "--depth", "1", "origin", FBGEMM_COMMIT])
        .current_dir(dest));
    run(Command::new("git")
        .args(["checkout", "FETCH_HEAD"])
        .current_dir(dest));

    run(Command::new("git")
        .args([
            "submodule",
            "update",
            "--init",
            "--depth",
            "1",
            "external/cpuinfo",
            "external/asmjit",
        ])
        .current_dir(dest));
}

fn has_ninja() -> bool {
    Command::new("ninja")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn find_clang_cl() -> Option<String> {
    // Check PATH first
    if Command::new("clang-cl")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
    {
        return Some("clang-cl".to_string());
    }

    // Check common Visual Studio locations on Windows
    if cfg!(target_os = "windows") {
        let program_files = env::var("ProgramFiles").unwrap_or_default();
        for year in ["2022", "2019"] {
            for edition in ["Enterprise", "Professional", "Community", "BuildTools"] {
                let path = format!(
                    "{}\\Microsoft Visual Studio\\{}\\{}\\VC\\Tools\\Llvm\\x64\\bin\\clang-cl.exe",
                    program_files, year, edition
                );
                if Path::new(&path).exists() {
                    return Some(path);
                }
            }
        }
    }

    None
}

/// On MSVC, patch FBGEMM's CMakeLists.txt so the FP32 inline-asm sources are
/// compiled when using clang-cl (which is MSVC-ABI compatible but supports
/// GCC-style inline assembly).
///
/// Two guards need relaxing:
///   1. `if(NOT MSVC)` around `-masm=intel` flags — clang-cl handles this fine
///   2. `if(MSVC) list(FILTER ... EXCLUDE src/fp32/)` — don't strip FP32 sources
fn patch_cmakelists_for_clang_cl(fbgemm_src: &Path) {
    let path = fbgemm_src.join("CMakeLists.txt");
    let content = std::fs::read_to_string(&path).unwrap();

    // Only patch once
    if content.contains("# patched-by-fbgemm-rs") {
        return;
    }

    let patched = content
        // 1. Allow -masm=intel when the compiler is Clang (even under MSVC)
        .replace(
            "if(NOT MSVC)\n  set_source_files_properties(",
            "if(NOT MSVC OR CMAKE_CXX_COMPILER_ID MATCHES \"Clang\") # patched-by-fbgemm-rs\n  set_source_files_properties(",
        )
        // 2. Only strip fp32 sources for real cl.exe, not clang-cl
        .replace(
            "if(MSVC)\n  list(FILTER FBGEMM_GENERIC_SRCS EXCLUDE REGEX \"src/fp32/.*\\\\.cc$\")\nendif()",
            "if(MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES \"Clang\") # patched-by-fbgemm-rs\n  list(FILTER FBGEMM_GENERIC_SRCS EXCLUDE REGEX \"src/fp32/.*\\\\.cc$\")\nendif()",
        );

    std::fs::write(&path, patched).unwrap();
}

/// Search for a static library in a directory, checking multi-config subdirs.
fn find_lib_dir(base: &Path, lib_name: &str) -> PathBuf {
    let candidates = if cfg!(target_env = "msvc") {
        vec![
            base.join("Release"),
            base.join("Debug"),
            base.to_path_buf(),
        ]
    } else {
        vec![base.to_path_buf()]
    };

    let file_name = if cfg!(target_env = "msvc") {
        format!("{}.lib", lib_name)
    } else {
        format!("lib{}.a", lib_name)
    };

    for dir in &candidates {
        if dir.join(&file_name).exists() {
            return dir.clone();
        }
    }

    base.to_path_buf()
}

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let fbgemm_src = out_dir.join("fbgemm");

    clone_fbgemm(&fbgemm_src);

    // Configure cmake
    let mut cfg = cmake::Config::new(&fbgemm_src);

    if has_ninja() {
        cfg.generator("Ninja");
    }

    // On MSVC targets, use clang-cl so that inline-asm FP32 kernels compile.
    // clang-cl produces MSVC-ABI-compatible objects while supporting GCC asm.
    if cfg!(target_env = "msvc") {
        let clang_cl = find_clang_cl().unwrap_or_else(|| {
            panic!(
                "fbgemm-rs: building for MSVC requires clang-cl (for inline assembly support). \
                 Install it via the Visual Studio Installer ('C++ Clang tools for Windows') \
                 or from https://releases.llvm.org"
            )
        });
        patch_cmakelists_for_clang_cl(&fbgemm_src);
        // Find the matching clang (same dir as clang-cl) for the C compiler
        let clang_c = clang_cl.replace("clang-cl", "clang");
        cfg.define("CMAKE_C_COMPILER", &clang_c);
        cfg.define("CMAKE_CXX_COMPILER", &clang_cl);
    }

    let dst = cfg
        .define("FBGEMM_LIBRARY_TYPE", "STATIC")
        .define("FBGEMM_BUILD_TESTS", "OFF")
        .define("FBGEMM_BUILD_BENCHMARKS", "OFF")
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
        .build();

    let cmake_build_dir = out_dir.join("build");

    // Compile the C++ wrapper
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .file(manifest_dir.join("wrapper/fbgemm_sgemm.cpp"))
        .include(fbgemm_src.join("include"))
        .include(fbgemm_src.join("external/cpuinfo/include"))
        .include(fbgemm_src.join("external/asmjit/src"));

    if cfg!(target_env = "msvc") {
        build.flag("/std:c++17");
    } else {
        build.flag("-std=c++17");
    }

    if cfg!(target_arch = "aarch64") {
        build
            .define("FBGEMM_ENABLE_KLEIDIAI", None)
            .define("FBGEMM_FP32_FALLBACK_TO_REF_KERNEL", None);
    }

    build.compile("fbgemm_sgemm_wrapper");

    // Link FBGEMM (installed into dst/lib by cmake)
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=fbgemm");

    // Link cpuinfo and asmjit from the cmake build tree
    let cpuinfo_dir = find_lib_dir(&cmake_build_dir.join("cpuinfo"), "cpuinfo");
    println!("cargo:rustc-link-search=native={}", cpuinfo_dir.display());
    println!("cargo:rustc-link-lib=static=cpuinfo");

    let cpuinfo_int_dir =
        find_lib_dir(&cmake_build_dir.join("cpuinfo"), "cpuinfo_internals");
    println!(
        "cargo:rustc-link-search=native={}",
        cpuinfo_int_dir.display()
    );
    println!("cargo:rustc-link-lib=static=cpuinfo_internals");

    let asmjit_dir = find_lib_dir(&cmake_build_dir.join("asmjit"), "asmjit");
    println!(
        "cargo:rustc-link-search=native={}",
        asmjit_dir.display()
    );
    println!("cargo:rustc-link-lib=static=asmjit");

    // Link C++ standard library
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=c++");
    } else if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=stdc++");
    }
    // On Windows/MSVC the C++ runtime is linked automatically

    println!("cargo:rerun-if-changed=wrapper/fbgemm_sgemm.cpp");
    println!("cargo:rerun-if-changed=wrapper/fbgemm_sgemm.h");
}
