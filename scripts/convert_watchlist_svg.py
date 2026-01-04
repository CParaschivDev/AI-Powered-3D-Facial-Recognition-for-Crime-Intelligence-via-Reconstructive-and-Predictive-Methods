"""Convert the watchlist SVG to PNG and @2x PNG."""
from pathlib import Path
import sys
try:
    import cairosvg
except Exception:
    print("Missing dependency 'cairosvg'. Install with: pip install cairosvg")
    sys.exit(2)
SVG = Path('docs/figures/watchlist_search_pipeline.svg')
PNG = SVG.with_suffix('.png')
if not SVG.exists():
    print('SVG not found at', SVG)
    sys.exit(1)
try:
    cairosvg.svg2png(url=str(SVG), write_to=str(PNG))
    print('Wrote', PNG)
    PNG2 = SVG.with_name(SVG.stem + '@2x.png')
    cairosvg.svg2png(url=str(SVG), write_to=str(PNG2), scale=2.0)
    print('Wrote', PNG2)
except Exception as e:
    print('Conversion failed:', e)
    sys.exit(3)
