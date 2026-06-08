param(
  [string[]]$Hosts = @("RaspberryPi_2", "RaspberryPi_3"),
  [string]$RemoteDir = "~/fedavg_course"
)
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$items = @("src", "configs", "scripts", "tests", "pyproject.toml", "requirements-pi.txt", "README.md")

foreach ($hostName in $Hosts) {
  Write-Host "Preparing ${hostName}:${RemoteDir}"
  ssh -o ClearAllForwardings=yes $hostName "mkdir -p $RemoteDir"
  foreach ($item in $items) {
    scp -o ClearAllForwardings=yes -r (Join-Path $root $item) "${hostName}:${RemoteDir}/"
  }
}
