# Install CUDA 12.4 toolkit pieces from NVIDIA redist zips (llama.cpp CI shape).
# Used by publish-oci.yml on windows-2022. Local builds should use a full toolkit.
# Expand each zip individually — pwsh expands *.zip and unzip then treats
# extra names as members of the first archive.
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Prefix = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
New-Item -ItemType Directory -Force -Path $Prefix | Out-Null

$pkgs = @(
    "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-12.4.127-archive.zip",
    "https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/windows-x86_64/cuda_nvcc-windows-x86_64-12.4.131-archive.zip",
    "https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvrtc/windows-x86_64/cuda_nvrtc-windows-x86_64-12.4.127-archive.zip",
    "https://developer.download.nvidia.com/compute/cuda/redist/libcublas/windows-x86_64/libcublas-windows-x86_64-12.4.5.8-archive.zip",
    "https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvtx/windows-x86_64/cuda_nvtx-windows-x86_64-12.4.127-archive.zip",
    "https://developer.download.nvidia.com/compute/cuda/redist/cuda_profiler_api/windows-x86_64/cuda_profiler_api-windows-x86_64-12.4.127-archive.zip",
    "https://developer.download.nvidia.com/compute/cuda/redist/visual_studio_integration/windows-x86_64/visual_studio_integration-windows-x86_64-12.4.127-archive.zip",
    "https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvprof/windows-x86_64/cuda_nvprof-windows-x86_64-12.4.127-archive.zip",
    "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cccl/windows-x86_64/cuda_cccl-windows-x86_64-12.4.127-archive.zip"
)

foreach ($url in $pkgs) {
    $zip = Join-Path $Prefix (Split-Path $url -Leaf)
    if (-not (Test-Path $zip)) {
        Write-Host "==> download $(Split-Path $zip -Leaf)"
        curl.exe -fsSL -o $zip $url
    }
    $len = (Get-Item $zip).Length
    if ($len -lt 10000) {
        throw "download too small ($len bytes): $zip"
    }
    Write-Host "==> expand $(Split-Path $zip -Leaf) ($len bytes)"
    Expand-Archive -Path $zip -DestinationPath $Prefix -Force
}

Get-ChildItem $Prefix -Directory -Filter "*-archive" | ForEach-Object {
    Copy-Item -Path (Join-Path $_.FullName "*") -Destination $Prefix -Recurse -Force
    Remove-Item $_.FullName -Recurse -Force
}

$bin = Join-Path $Prefix "bin"
if (-not (Test-Path (Join-Path $bin "nvcc.exe"))) {
    Get-ChildItem $Prefix | Format-Table Name, Length
    throw "nvcc.exe missing after CUDA 12.4 redist install"
}

if ($env:GITHUB_PATH) {
    Add-Content $env:GITHUB_PATH $bin
    Add-Content $env:GITHUB_PATH (Join-Path $Prefix "libnvvp")
}
if ($env:GITHUB_ENV) {
    Add-Content $env:GITHUB_ENV "CUDA_PATH=$Prefix"
    Add-Content $env:GITHUB_ENV "CUDA_PATH_V12_4=$Prefix"
}
$env:CUDA_PATH = $Prefix
$env:Path = "$bin;$env:Path"
Write-Host "CUDA 12.4 installed at $Prefix"
