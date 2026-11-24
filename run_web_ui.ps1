$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location "$ScriptDir\web"

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " Wildfire Risk Intelligence - Node.js Web UI" -ForegroundColor Cyan
Write-Host " URL: http://localhost:3000" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan

Start-Process "http://localhost:3000"
node server.js
