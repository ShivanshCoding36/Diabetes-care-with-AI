# Build Tailwind CSS
# Run this script whenever you update templates or add new Tailwind classes

Write-Host "Building Tailwind CSS for production..." -ForegroundColor Cyan
.\tailwindcss.exe -i .\static\css\input.css -o .\static\css\output.css --minify
Write-Host "✓ Tailwind CSS built successfully!" -ForegroundColor Green
