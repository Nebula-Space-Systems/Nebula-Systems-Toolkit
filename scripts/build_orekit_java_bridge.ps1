param(
    [string]$PythonExe = ".\.venv\Scripts\python.exe",
    [string]$JavacExe = "javac",
    [string]$JarExe = "jar"
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$src = Join-Path $repoRoot "nstk\propagation\_java_orbit_propagation\com\nstk\propagation\OrekitOrbitPropagationBridge.java"
$jarOut = Join-Path $repoRoot "nstk\propagation\_java_orbit_propagation\OrekitOrbitPropagationBridge.jar"
$tmp = Join-Path $env:TEMP "nstk_java_bridge_build"
$classes = Join-Path $tmp "classes"

$jarsDir = & $PythonExe -c "import pathlib, orekit_jpype; print(pathlib.Path(orekit_jpype.__file__).resolve().parent / 'jars')"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to discover orekit_jpype jars directory."
}
$cp = "$jarsDir/*"

if (Test-Path $tmp) {
    Remove-Item -Recurse -Force $tmp
}
New-Item -ItemType Directory -Path $classes -Force | Out-Null

& $JavacExe -encoding UTF-8 -cp $cp -d $classes $src
if ($LASTEXITCODE -ne 0) {
    throw "javac failed."
}

& $JarExe --create --file $jarOut -C $classes .
if ($LASTEXITCODE -ne 0) {
    throw "jar creation failed."
}

Write-Output "Built $jarOut"
