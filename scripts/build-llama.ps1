# Build ggml-org/llama.cpp shared libs + llamastack.dll into lib/windows-amd64/.
# Published windows/amd64 overlay is CPU (MSVC). CUDA is not a release target.
# Env: LLAMA_CPP_REF (default master), DEST_DIR, JOBS
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Ref = if ($env:LLAMA_CPP_REF) { $env:LLAMA_CPP_REF } else { "master" }
$Jobs = if ($env:JOBS) { [int]$env:JOBS } else { [Environment]::ProcessorCount }
$SrcUrl = "https://github.com/ggml-org/llama.cpp.git"
$Out = if ($env:DEST_DIR) { $env:DEST_DIR } else { Join-Path $Root "lib\windows-amd64" }
$Src = Join-Path $Root "build\llama.cpp"
$Build = Join-Path $Root "build\windows-amd64"

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere) {
    $vsRoot = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
    if ($vsRoot) {
        $devCmd = Join-Path $vsRoot "Common7\Tools\VsDevCmd.bat"
        if (Test-Path $devCmd) {
            Write-Host "==> enter VS x64 env via VsDevCmd.bat"
            cmd /c "`"$devCmd`" -arch=amd64 -host_arch=amd64 && set" | ForEach-Object {
                if ($_ -match '^(.*?)=(.*)$') {
                    Set-Item -Path "env:$($matches[1])" -Value $matches[2]
                }
            }
        }
    }
}

if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    throw "cmake not found"
}
if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
    throw "cl.exe not found (need VS Build Tools x64)"
}

New-Item -ItemType Directory -Force -Path (Join-Path $Root "build") | Out-Null
if (Test-Path (Join-Path $Src ".git")) {
    git -C $Src fetch --depth 1 origin $Ref
    if ($LASTEXITCODE -ne 0) { throw "git fetch failed: $LASTEXITCODE" }
    git -C $Src checkout --force FETCH_HEAD
    if ($LASTEXITCODE -ne 0) { throw "git checkout failed: $LASTEXITCODE" }
} else {
    if (Test-Path $Src) { Remove-Item -Recurse -Force $Src }
    git clone --depth 1 --branch $Ref $SrcUrl $Src
    if ($LASTEXITCODE -ne 0) {
        git clone --depth 1 $SrcUrl $Src
        if ($LASTEXITCODE -ne 0) { throw "git clone failed: $LASTEXITCODE" }
        git -C $Src fetch --depth 1 origin $Ref
        if ($LASTEXITCODE -ne 0) { throw "git fetch $Ref failed: $LASTEXITCODE" }
        git -C $Src checkout --force FETCH_HEAD
        if ($LASTEXITCODE -ne 0) { throw "git checkout failed: $LASTEXITCODE" }
    }
}

Write-Host "==> cmake llama.cpp $Ref flavor=cpu (MSVC) -> $Out"
$cmakeArgs = @(
    "-S", $Src,
    "-B", $Build,
    "-G", "Visual Studio 17 2022",
    "-A", "x64",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBUILD_SHARED_LIBS=ON",
    "-DGGML_NATIVE=OFF",
    "-DGGML_CUDA=OFF",
    "-DGGML_METAL=OFF",
    "-DGGML_VULKAN=OFF",
    "-DGGML_HIP=OFF",
    "-DGGML_BLAS=OFF",
    "-DLLAMA_BUILD_TESTS=OFF",
    "-DLLAMA_BUILD_EXAMPLES=OFF",
    "-DLLAMA_BUILD_SERVER=OFF",
    "-DLLAMA_BUILD_TOOLS=OFF"
)
& cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed: $LASTEXITCODE" }

& cmake --build $Build --config Release --target llama --parallel $Jobs
if ($LASTEXITCODE -ne 0) { throw "cmake build failed: $LASTEXITCODE" }

if (Test-Path $Out) { Remove-Item -Recurse -Force $Out }
New-Item -ItemType Directory -Force -Path $Out | Out-Null

$want = @("llama.dll", "ggml.dll", "ggml-base.dll", "ggml-cpu.dll")
$dlls = Get-ChildItem -Path $Build -Recurse -File |
    Where-Object {
        $_.Name -in $want -and
        $_.FullName -notmatch '[\\/](?:_deps|third_party|CMakeFiles)[\\/]'
    }
if (-not ($dlls | Where-Object { $_.Name -eq "llama.dll" })) {
    Get-ChildItem -Path $Build -Recurse -File -Filter "*llama*" | Select-Object -First 40 FullName
    throw "llama.dll not found under $Build"
}
foreach ($dll in $dlls) {
    Copy-Item $dll.FullName (Join-Path $Out $dll.Name) -Force
    $lib = [IO.Path]::ChangeExtension($dll.FullName, ".lib")
    if (Test-Path $lib) {
        Copy-Item $lib (Join-Path $Out (Split-Path $lib -Leaf)) -Force
    }
}

$includeLlama = Join-Path $Src "include"
$includeGgml = Join-Path $Src "ggml\include"
$shimC = Join-Path $Root "native\llama-stack.c"
$shimDll = Join-Path $Out "llamastack.dll"
Write-Host "==> cl llamastack.dll"
Push-Location $Out
try {
    & cl /nologo /O2 /LD /DLLAMA_STACK_BUILD `
        /I"$($Root)\include" /I$includeLlama /I$includeGgml `
        $shimC `
        /Fe:llamastack.dll `
        /link /LIBPATH:$Out llama.lib
    if ($LASTEXITCODE -ne 0) { throw "cl llamastack.dll failed: $LASTEXITCODE" }
} finally {
    Pop-Location
}

# Import libs are link-only; overlay is the DLLs.
Get-ChildItem $Out -Filter *.lib -ErrorAction SilentlyContinue | Remove-Item -Force
Get-ChildItem $Out -Filter *.exp -ErrorAction SilentlyContinue | Remove-Item -Force
Get-ChildItem $Out -Filter *.obj -ErrorAction SilentlyContinue | Remove-Item -Force
Get-ChildItem $Out -Filter llama-stack.obj -ErrorAction SilentlyContinue | Remove-Item -Force

if (-not (Test-Path $shimDll)) { throw "llamastack.dll missing" }
if (-not (Test-Path (Join-Path $Out "llama.dll"))) { throw "llama.dll missing" }
if (-not (Test-Path (Join-Path $Out "ggml.dll"))) { throw "ggml.dll missing" }

Write-Host "==> staged:"
Get-ChildItem $Out | Format-Table Name, Length
Write-Host "staged $Out"
