# Setup script for jpegli-rs FFI tests (Windows)
#
# This script:
#   1. Initializes the jpegli-cpp git submodule
#   2. Builds the C++ jpegli library with CMake/MSVC
#   3. Enables the ffi-tests feature in Cargo.toml
#   4. Runs the FFI comparison tests
#
# Usage:
#   .\internal\setup-ffi-tests.ps1 [-BuildOnly] [-TestOnly] [-Clean]
#
# Requirements:
#   - Visual Studio 2019+ with C++ workload
#   - CMake 3.16+ (in PATH)
#   - Git (in PATH)
#   - Rust/Cargo

param(
    [switch]$BuildOnly,
    [switch]$TestOnly,
    [switch]$Clean,
    [int]$Jobs = 0
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$JpegliCpp = Join-Path $ScriptDir "jpegli-cpp"
$BuildDir = Join-Path $JpegliCpp "build"

function Write-Info($msg) { Write-Host "[INFO] $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Err($msg) { Write-Host "[ERROR] $msg" -ForegroundColor Red; exit 1 }

# Check prerequisites
function Check-Prerequisites {
    Write-Info "Checking prerequisites..."

    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Write-Err "git is required"
    }
    if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
        Write-Err "cmake is required"
    }
    if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
        Write-Err "cargo is required"
    }

    # Check for Visual Studio
    $vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vsWhere)) {
        Write-Warn "Visual Studio not detected via vswhere. CMake may fail."
    }
}

# Initialize git submodule
function Init-Submodule {
    Write-Info "Initializing jpegli-cpp submodule..."

    Push-Location $RepoRoot
    try {
        if (-not (Test-Path (Join-Path $JpegliCpp "CMakeLists.txt"))) {
            git submodule update --init --recursive internal/jpegli-cpp
        } else {
            Write-Info "Submodule already initialized"
        }
    } finally {
        Pop-Location
    }

    if (-not (Test-Path (Join-Path $JpegliCpp "CMakeLists.txt"))) {
        Write-Err "Failed to initialize submodule"
    }
}

# Build C++ jpegli
function Build-Cpp {
    Write-Info "Building C++ jpegli..."

    if ($Clean -and (Test-Path $BuildDir)) {
        Write-Info "Cleaning build directory..."
        Remove-Item -Recurse -Force $BuildDir
    }

    New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
    Push-Location $BuildDir

    try {
        # Configure
        Write-Info "Configuring CMake..."
        cmake -G "Visual Studio 17 2022" -A x64 `
            -DCMAKE_BUILD_TYPE=Release `
            -DBUILD_TESTING=OFF `
            -DJPEGXL_ENABLE_TOOLS=ON `
            -DJPEGXL_ENABLE_JPEGLI_LIBJPEG=ON `
            -DJPEGXL_ENABLE_SJPEG=OFF `
            -DJPEGXL_ENABLE_OPENEXR=OFF `
            -DJPEGXL_ENABLE_SKCMS=OFF `
            -DJPEGXL_STATIC=ON `
            ..

        if ($LASTEXITCODE -ne 0) { Write-Err "CMake configuration failed" }

        # Build
        Write-Info "Building (this may take 10-15 minutes)..."
        $buildArgs = @("--build", ".", "--config", "Release", "--target", "jpegli-static")
        if ($Jobs -gt 0) {
            $buildArgs += @("--parallel", $Jobs)
        } else {
            $buildArgs += "--parallel"
        }
        cmake @buildArgs

        if ($LASTEXITCODE -ne 0) { Write-Err "Build failed" }

        # Verify
        $libPath = Join-Path $BuildDir "lib\Release\jpegli-static.lib"
        if (-not (Test-Path $libPath)) {
            Write-Err "Build failed - jpegli-static.lib not found"
        }

        Write-Info "C++ build complete!"
    } finally {
        Pop-Location
    }
}

# Enable ffi-tests feature
function Enable-FfiFeature {
    Write-Info "Enabling ffi-tests feature..."

    $cargoToml = Join-Path $RepoRoot "jpegli-rs\Cargo.toml"
    $content = Get-Content $cargoToml -Raw

    if ($content -match '(?m)^jpegli-internals-sys = \{ path') {
        Write-Info "ffi-tests feature already enabled"
        return
    }

    # Uncomment the dependency
    $content = $content -replace '(?m)^# jpegli-internals-sys = \{ path', 'jpegli-internals-sys = { path'

    # Update the feature
    $content = $content -replace '(?m)^ffi-tests = \[\]', 'ffi-tests = ["dep:jpegli-internals-sys"]'

    Set-Content $cargoToml $content

    Write-Info "ffi-tests feature enabled in Cargo.toml"
    Write-Warn "Remember to revert before committing if you don't want local changes!"
}

# Run FFI tests
function Run-Tests {
    Write-Info "Running FFI comparison tests..."

    Push-Location $RepoRoot
    try {
        cargo build --features ffi-tests -p jpegli-rs
        if ($LASTEXITCODE -ne 0) { Write-Err "Build failed" }

        cargo test --features ffi-tests -p jpegli-rs -- --nocapture
        if ($LASTEXITCODE -ne 0) { Write-Err "Tests failed" }

        Write-Info "FFI tests complete!"
    } finally {
        Pop-Location
    }
}

# Main
Write-Host "============================================"
Write-Host "  jpegli-rs FFI Test Setup (Windows)"
Write-Host "============================================"
Write-Host

Check-Prerequisites

if (-not $TestOnly) {
    Init-Submodule
    Build-Cpp
}

if (-not $BuildOnly) {
    Enable-FfiFeature
    Run-Tests
}

Write-Host
Write-Host "============================================"
Write-Info "Setup complete!"
Write-Host
Write-Host "To run FFI tests again:"
Write-Host "  cargo test --features ffi-tests -p jpegli-rs"
Write-Host "============================================"
