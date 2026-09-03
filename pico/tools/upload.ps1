<#
.SYNOPSIS
    Copy the harness firmware onto a Pico running MicroPython, over the serial port.

.DESCRIPTION
    A standalone replacement for `mpremote fs cp` for machines with no Python
    installed - which is the normal case on the Windows boxes this project is
    developed on. It drives MicroPython's raw REPL directly: file contents go
    across base64-encoded so binary-safe transfer needs nothing on the board
    beyond `binascii`, which is built in.

    The board must already have MicroPython on it. To install it, hold BOOTSEL
    while plugging the board in and copy the Pico 2 W .uf2 from
    https://micropython.org/download/RPI_PICO2_W/ onto the drive that appears.

.PARAMETER Port
    Serial port, e.g. COM9. Auto-detected from the USB vendor ID if omitted.

.PARAMETER Source
    Firmware directory to copy. Defaults to ..\firmware next to this script.

.PARAMETER NoReset
    Leave the board sitting at the REPL instead of soft-resetting into main.py.

.EXAMPLE
    .\upload.ps1
    .\upload.ps1 -Port COM9 -NoReset
#>
[CmdletBinding()]
param(
    [string]$Port,
    [string]$Source = (Join-Path $PSScriptRoot '..\firmware'),
    [switch]$NoReset
)

$ErrorActionPreference = 'Stop'

# Raspberry Pi's USB vendor ID. A board running MicroPython enumerates as a
# CDC device under it; one in BOOTSEL mode shows up as mass storage instead,
# which is why a board that was just plugged in with the button held is not
# found here.
$PICO_VID = 'VID_2E8A'

# 1 KB per round trip. Larger chunks mean fewer round trips but a bigger
# compile buffer on the board; this is comfortably inside what a Pico can take
# and still moves the whole firmware in a few seconds.
$CHUNK = 1024

$FILES = @(
    'main.py', 'config.py', 'sampler.py', 'netcfg.py', 'proxy.py', 'server.py',
    'www/index.html', 'www/app.js', 'www/style.css'
)

function Find-PicoPort {
    $matches = Get-CimInstance Win32_PnPEntity |
        Where-Object { $_.DeviceID -like "*$PICO_VID*" -and $_.Name -match '\(COM(\d+)\)' } |
        ForEach-Object { [regex]::Match($_.Name, '\((COM\d+)\)').Groups[1].Value }
    if (-not $matches) {
        throw "No Pico serial port found. Is the board plugged in and running MicroPython? Pass -Port to override."
    }
    if ($matches.Count -gt 1) {
        throw "Several Pico ports found ($($matches -join ', ')). Pass -Port to pick one."
    }
    return $matches[0]
}

function Open-Board([string]$name) {
    $serial = New-Object System.IO.Ports.SerialPort $name, 115200, 'None', 8, 'One'
    $serial.ReadTimeout = 3000
    $serial.WriteTimeout = 3000
    $serial.DtrEnable = $true
    $serial.Open()
    Start-Sleep -Milliseconds 200
    return $serial
}

function Read-Until($serial, [string]$terminator, [int]$timeoutMs = 10000) {
    $buffer = ''
    $deadline = (Get-Date).AddMilliseconds($timeoutMs)
    while ((Get-Date) -lt $deadline) {
        $buffer += $serial.ReadExisting()
        if ($buffer.Contains($terminator)) { return $buffer }
        Start-Sleep -Milliseconds 10
    }
    throw "Timed out waiting for '$([regex]::Escape($terminator))'. Got: $buffer"
}

function Enter-RawRepl($serial) {
    $serial.Write([string][char]3 + [string][char]3)   # Ctrl-C: stop main.py if it is running
    Start-Sleep -Milliseconds 200
    $serial.DiscardInBuffer()
    $serial.Write([string][char]1)                      # Ctrl-A: raw REPL
    # The banner and its prompt arrive in one go, so wait for the prompt only -
    # waiting for the banner first would swallow the prompt with it.
    Read-Until $serial 'raw REPL; CTRL-B to exit' 5000 | Out-Null
}

function Exit-RawRepl($serial) {
    $serial.Write([string][char]2)                      # Ctrl-B: back to the friendly REPL
    Start-Sleep -Milliseconds 200
    $serial.DiscardInBuffer()
}

function Invoke-Board($serial, [string]$code) {
    <# Run one statement and return its stdout. Raw REPL answers with
       'OK<stdout>\x04<stderr>\x04>' - a non-empty stderr is a traceback from
       the board and is surfaced as a terminating error rather than ignored. #>
    $serial.Write($code)
    $serial.Write([string][char]4)                      # Ctrl-D: execute

    $buffer = ''
    $deadline = (Get-Date).AddMilliseconds(15000)
    while ((Get-Date) -lt $deadline) {
        $buffer += $serial.ReadExisting()
        # Two \x04 mark the end of stdout and of stderr respectively.
        if (($buffer.ToCharArray() | Where-Object { $_ -eq [char]4 }).Count -ge 2) { break }
        Start-Sleep -Milliseconds 5
    }

    $body = $buffer -replace '^\s*OK', ''
    $parts = $body.Split([char]4)
    if ($parts.Count -lt 2) { throw "Malformed reply from board: $buffer" }
    $stdout = $parts[0]
    $stderr = $parts[1].Trim()
    if ($stderr) { throw "Board raised:`n$stderr" }
    return $stdout
}

function Send-File($serial, [string]$localPath, [string]$remotePath) {
    $bytes = [System.IO.File]::ReadAllBytes($localPath)
    Invoke-Board $serial "f=open('$remotePath','wb')" | Out-Null
    for ($offset = 0; $offset -lt $bytes.Length; $offset += $CHUNK) {
        $length = [Math]::Min($CHUNK, $bytes.Length - $offset)
        $slice = New-Object byte[] $length
        [Array]::Copy($bytes, $offset, $slice, 0, $length)
        $b64 = [Convert]::ToBase64String($slice)
        Invoke-Board $serial "f.write(_b('$b64'))" | Out-Null
    }
    Invoke-Board $serial 'f.close()' | Out-Null

    $onBoard = [int](Invoke-Board $serial "print(os.stat('$remotePath')[6])").Trim()
    if ($onBoard -ne $bytes.Length) {
        throw "$remotePath is $onBoard bytes on the board, expected $($bytes.Length)"
    }
    return $bytes.Length
}

# ─────────────────────────────────────────────────────────────────────────────

if (-not $Port) { $Port = Find-PicoPort }
$Source = (Resolve-Path $Source).Path
Write-Host "Port:   $Port"
Write-Host "Source: $Source"

$serial = Open-Board $Port
try {
    Enter-RawRepl $serial

    # `binascii` is the modern name; older builds only have `ubinascii`.
    Invoke-Board $serial @'
import os
try:
    from binascii import a2b_base64 as _b
except ImportError:
    from ubinascii import a2b_base64 as _b
'@ | Out-Null

    $version = (Invoke-Board $serial 'import sys; print(sys.version, sys.implementation._machine)').Trim()
    Write-Host "Board:  $version`n"

    Invoke-Board $serial @'
try:
    os.mkdir('www')
except OSError:
    pass
'@ | Out-Null

    $total = 0
    foreach ($relative in $FILES) {
        $local = Join-Path $Source ($relative -replace '/', '\')
        if (-not (Test-Path $local)) { throw "Missing source file: $local" }
        $size = Send-File $serial $local $relative
        $total += $size
        Write-Host ("  {0,-20} {1,7:N0} bytes" -f $relative, $size)
    }
    Write-Host ("`n{0} files, {1:N0} bytes total" -f $FILES.Count, $total)

    $free = (Invoke-Board $serial @'
s = os.statvfs('/')
print(s[0] * s[3] // 1024)
'@).Trim()
    Write-Host "$free KB free on the board"

    Exit-RawRepl $serial
    if (-not $NoReset) {
        $serial.Write([string][char]4)                  # Ctrl-D: soft reset, runs main.py
        Start-Sleep -Milliseconds 2500
        Write-Host "`n--- boot output ---"
        Write-Host $serial.ReadExisting()
    }
}
finally {
    $serial.Close()
    $serial.Dispose()
}
