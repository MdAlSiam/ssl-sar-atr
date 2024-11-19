# Create this as startup-wsl.ps1
while ($true) {
    # Check if WSL is running
    $wslRunning = wsl --list --running
    
    if (-not $wslRunning) {
        # Start WSL with admin privileges
        Start-Process wsl -Verb RunAs
        
        # Wait for WSL to initialize
        Start-Sleep -Seconds 5
        
        # Run VS Code in WSL
        wsl -e code .
    }
    
    # Wait 1 minute(s) before next check
    Start-Sleep -Seconds 60
}

# To execute this setup:
# 1. Press `Win + R`
# 2. Type `shell:startup` and press Enter
# 3. Right-click in the opened folder → New → Text Document
# 4. Rename it to `startup-wsl.ps1`
# 5. Paste the script above
# 6. Right-click the `startup-wsl.ps1` file → Create shortcut
# 7. Right-click the shortcut → Properties
# 8. In "Target", enter:
# ```
# powershell.exe -ExecutionPolicy Bypass -File "startup-wsl.ps1"
# ```
# 9. * Click "Advanced" → Check "Run as administrator"
# 10. Click Apply anbd OK / OK and Apply
# The script will now run automatically when Windows starts.

