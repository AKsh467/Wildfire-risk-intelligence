@echo off
echo ==================================================================
echo Pushing Wildfire Risk Intelligence to GitHub (AKsh467)
echo Repository: https://github.com/AKsh467/wildfire-risk-intelligence
echo ==================================================================
git push -u origin main
if %ERRORLEVEL% EQU 0 (
    echo.
    echo SUCCESS: Repository successfully pushed to GitHub!
    echo Next step: Connect to Render.com to deploy in 1 click.
) else (
    echo.
    echo FAILED: If you see "Repository not found", please make sure
    echo you created the repo at: https://github.com/new
    echo with the name: wildfire-risk-intelligence
)
pause
