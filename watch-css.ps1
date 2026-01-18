# Watch Tailwind CSS changes during development
# This will automatically rebuild CSS when you modify templates

Write-Host "Watching Tailwind CSS for changes..." -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
.\tailwindcss.exe -i .\static\css\input.css -o .\static\css\output.css --watch
