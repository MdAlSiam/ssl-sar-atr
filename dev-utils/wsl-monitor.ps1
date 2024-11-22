$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument @'
while ($true) {
    $wslRunning = wsl --list --running
    if (-not $wslRunning) {
        Start-Process wsl -Verb RunAs
        Start-Sleep -Seconds 5
        wsl -e code .
    }
    Start-Sleep -Seconds 60
}
'@

$trigger = New-ScheduledTaskTrigger -AtStartup
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -MultipleInstances Parallel -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries

Register-ScheduledTask -TaskName "WSL Monitor" -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force

# Powershell (In this location): Set-ExecutionPolicy Bypass -Scope Process -Force; .\wsl-monitor.ps1