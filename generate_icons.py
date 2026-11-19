import os
import pymupdf

ICONS_DIR = os.path.join("frontend", "public", "icons")
os.makedirs(ICONS_DIR, exist_ok=True)

# 1. Standard App Icon (Rounded square with gradient background and book/AI symbol)
def create_standard_svg(size):
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">
  <defs>
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#1e3a8a" />
      <stop offset="50%" stop-color="#2563eb" />
      <stop offset="100%" stop-color="#0284c7" />
    </linearGradient>
    <linearGradient id="bookGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#ffffff" />
      <stop offset="100%" stop-color="#e0f2fe" />
    </linearGradient>
    <filter id="shadow" x="-10%" y="-10%" width="120%" height="120%">
      <feDropShadow dx="0" dy="{size * 0.03}" stdDeviation="{size * 0.04}" flood-color="#0f172a" flood-opacity="0.35"/>
    </filter>
  </defs>

  <!-- Background with rounded corners -->
  <rect width="{size}" height="{size}" rx="{size * 0.22}" fill="url(#bgGrad)" />

  <!-- Inner subtle border -->
  <rect x="{size * 0.02}" y="{size * 0.02}" width="{size * 0.96}" height="{size * 0.96}" rx="{size * 0.20}" fill="none" stroke="#ffffff" stroke-opacity="0.2" stroke-width="{size * 0.015}" />

  <!-- Central Graphic Group -->
  <g transform="translate({size * 0.16}, {size * 0.16}) scale({(size * 0.68) / 24})">
    <!-- Lucide BookOpen path with enhanced stroke -->
    <path d="M2 4h6a4 4 0 0 1 4 4v13a3 3 0 0 0-3-3H2z" fill="none" stroke="url(#bookGrad)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M22 4h-6a4 4 0 0 0-4 4v13a3 3 0 0 1 3-3h7z" fill="none" stroke="url(#bookGrad)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    <!-- Subtle pages lines inside book -->
    <line x1="5" y1="8" x2="9" y2="8" stroke="#ffffff" stroke-width="1.2" stroke-linecap="round" stroke-opacity="0.7"/>
    <line x1="5" y1="12" x2="9" y2="12" stroke="#ffffff" stroke-width="1.2" stroke-linecap="round" stroke-opacity="0.7"/>
    <line x1="15" y1="8" x2="19" y2="8" stroke="#ffffff" stroke-width="1.2" stroke-linecap="round" stroke-opacity="0.7"/>
    <line x1="15" y1="12" x2="19" y2="12" stroke="#ffffff" stroke-width="1.2" stroke-linecap="round" stroke-opacity="0.7"/>
    <!-- AI Sparkle at the top center -->
    <path d="M12 2l1 2 2 1-2 1-1 2-1-2-2-1 2-1z" fill="#facc15" />
  </g>
</svg>"""

# 2. Maskable Icon (Full bleed background, safe zone padding of 20%)
def create_maskable_svg(size):
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">
  <defs>
    <linearGradient id="maskGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0f172a" />
      <stop offset="30%" stop-color="#1e3a8a" />
      <stop offset="70%" stop-color="#2563eb" />
      <stop offset="100%" stop-color="#0284c7" />
    </linearGradient>
    <linearGradient id="bookGradM" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#ffffff" />
      <stop offset="100%" stop-color="#e0f2fe" />
    </linearGradient>
  </defs>

  <!-- Full Bleed Background for Maskable Icon -->
  <rect width="{size}" height="{size}" fill="url(#maskGrad)" />

  <!-- Inner Safe Zone Content (40% to 80% area) -->
  <g transform="translate({size * 0.22}, {size * 0.22}) scale({(size * 0.56) / 24})">
    <path d="M2 4h6a4 4 0 0 1 4 4v13a3 3 0 0 0-3-3H2z" fill="none" stroke="url(#bookGradM)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M22 4h-6a4 4 0 0 0-4 4v13a3 3 0 0 1 3-3h7z" fill="none" stroke="url(#bookGradM)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    <line x1="5" y1="8" x2="9" y2="8" stroke="#ffffff" stroke-width="1.2" stroke-linecap="round" stroke-opacity="0.7"/>
    <line x1="5" y1="12" x2="9" y2="12" stroke="#ffffff" stroke-width="1.2" stroke-linecap="round" stroke-opacity="0.7"/>
    <line x1="15" y1="8" x2="19" y2="8" stroke="#ffffff" stroke-width="1.2" stroke-linecap="round" stroke-opacity="0.7"/>
    <line x1="15" y1="12" x2="19" y2="12" stroke="#ffffff" stroke-width="1.2" stroke-linecap="round" stroke-opacity="0.7"/>
    <path d="M12 2l1 2 2 1-2 1-1 2-1-2-2-1 2-1z" fill="#facc15" />
  </g>
</svg>"""

def render_svg_to_png(svg_str, output_path, width, height):
    doc = pymupdf.open(stream=svg_str.encode("utf-8"), filetype="svg")
    page = doc[0]
    zoom_x = width / page.rect.width
    zoom_y = height / page.rect.height
    mat = pymupdf.Matrix(zoom_x, zoom_y)
    pix = page.get_pixmap(matrix=mat, alpha=True)
    pix.save(output_path)
    print(f"Generated: {output_path} ({pix.width}x{pix.height})")

def main():
    # 192x192 icon
    svg_192 = create_standard_svg(192)
    render_svg_to_png(svg_192, os.path.join(ICONS_DIR, "icon-192x192.png"), 192, 192)

    # 512x512 icon
    svg_512 = create_standard_svg(512)
    render_svg_to_png(svg_512, os.path.join(ICONS_DIR, "icon-512x512.png"), 512, 512)

    # 192x192 maskable icon
    svg_mask_192 = create_maskable_svg(192)
    render_svg_to_png(svg_mask_192, os.path.join(ICONS_DIR, "maskable-icon-192x192.png"), 192, 192)

    # 512x512 maskable icon
    svg_mask_512 = create_maskable_svg(512)
    render_svg_to_png(svg_mask_512, os.path.join(ICONS_DIR, "maskable-icon-512x512.png"), 512, 512)

    # Apple touch icon (180x180)
    svg_apple = create_standard_svg(180)
    render_svg_to_png(svg_apple, os.path.join("frontend", "public", "apple-touch-icon.png"), 180, 180)

    # Favicon 32x32 & 64x64
    render_svg_to_png(svg_192, os.path.join(ICONS_DIR, "favicon-32x32.png"), 32, 32)
    render_svg_to_png(svg_192, os.path.join(ICONS_DIR, "favicon-16x16.png"), 16, 16)

    # Also save clean brand favicon.svg in frontend/public
    with open(os.path.join("frontend", "public", "favicon.svg"), "w", encoding="utf-8") as f:
        f.write(create_standard_svg(64))
    print("Updated frontend/public/favicon.svg")

if __name__ == "__main__":
    main()
