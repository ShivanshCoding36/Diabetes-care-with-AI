# Tailwind CSS Setup

This project uses Tailwind CSS compiled for production to avoid the CDN warning.

## Files

- `tailwind.config.js` - Tailwind configuration
- `static/css/input.css` - Source CSS with Tailwind directives and custom styles
- `static/css/output.css` - Compiled production CSS (minified)
- `tailwindcss.exe` - Standalone Tailwind CLI (not committed to git)

## Building CSS

### Production Build (one-time)

```powershell
.\build-css.ps1
```

Or manually:

```powershell
.\tailwindcss.exe -i .\static\css\input.css -o .\static\css\output.css --minify
```

### Development Mode (watch for changes)

```powershell
.\watch-css.ps1
```

This will automatically rebuild CSS when you modify templates.

## First Time Setup

If `tailwindcss.exe` is not present, download it:

```powershell
Invoke-WebRequest -Uri "https://github.com/tailwindlabs/tailwindcss/releases/latest/download/tailwindcss-windows-x64.exe" -OutFile "tailwindcss.exe"
```

Then build the CSS:

```powershell
.\build-css.ps1
```

## Adding Custom Styles

Add your custom CSS to `static/css/input.css` and rebuild.

## Templates

Templates now use:

```html
<link
  rel="stylesheet"
  href="{{ url_for('static', filename='css/output.css') }}"
/>
```

Instead of:

```html
<script src="https://cdn.tailwindcss.com"></script>
```

This eliminates the production warning and improves performance.
