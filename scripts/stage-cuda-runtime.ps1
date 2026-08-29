# Copy user-mode CUDA 12 DLLs next to ggml-cuda.dll. Never ship nvcuda.dll (driver).
param(
    [Parameter(Mandatory = $true)]
    [string]$OutDir
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if (-not (Test-Path $OutDir)) { throw "overlay dir missing: $OutDir" }
if (-not (Test-Path (Join-Path $OutDir "ggml-cuda.dll"))) {
    throw "ggml-cuda.dll missing in $OutDir"
}

function Test-DriverDll([string]$Name) {
    return $Name -match '^(nvcuda|cuda)\.dll$'
}

$search = New-Object System.Collections.Generic.List[string]
if ($env:CUDA_PATH) {
    $bin = Join-Path $env:CUDA_PATH "bin"
    if (Test-Path $bin) { [void]$search.Add($bin) }
}
foreach ($p in @(
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
    )) {
    if (Test-Path $p) { [void]$search.Add($p) }
}

$required = @("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll")
foreach ($name in $required) {
    if (Test-DriverDll $name) { throw "refusing to ship driver $name" }
    $dest = Join-Path $OutDir $name
    if (Test-Path $dest) { continue }
    $hit = $null
    foreach ($dir in $search) {
        $cand = Join-Path $dir $name
        if (Test-Path $cand) { $hit = $cand; break }
    }
    if (-not $hit) { throw "CUDA user-mode $name not found (set CUDA_PATH)" }
    Copy-Item $hit $dest -Force
}

Get-ChildItem $OutDir -Filter "nvcuda.dll" -ErrorAction SilentlyContinue | ForEach-Object {
    throw "refusing to ship NVIDIA driver $($_.Name)"
}
Get-ChildItem $OutDir -Filter "cuda.dll" -ErrorAction SilentlyContinue | ForEach-Object {
    throw "refusing to ship NVIDIA driver $($_.Name)"
}

Write-Host "staged CUDA user-mode runtime into $OutDir"
