@echo off
echo ========================================================
echo  Wildfire Risk Intelligence - Node.js Web UI
echo ========================================================
cd /d "%~dp0\web"

echo [*] Launching Node.js Server on http://localhost:3000 ...
start http://localhost:3000
node server.js
pause
