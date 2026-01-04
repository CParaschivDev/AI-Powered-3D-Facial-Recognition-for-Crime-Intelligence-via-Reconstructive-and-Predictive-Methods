"""Generate a simple SVG facial analytics pipeline diagram.

Run this script to (re)create `docs/figures/facial_analytics_pipeline.svg`.
"""
from pathlib import Path

SVG = r'''
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="260" viewBox="0 0 1200 260">
  <style>
    .box { fill:#ffffff; stroke:#2b2b2b; stroke-width:2; rx:6; }
    .title { font: bold 14px sans-serif; fill:#111; }
    .sub { font: 12px sans-serif; fill:#444; }
    .arrow { stroke:#2b2b2b; stroke-width:2; fill:none; marker-end:url(#arrowhead);} 
  </style>
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#2b2b2b" />
    </marker>
  </defs>

  <!-- Input -->
  <rect x="20" y="40" width="140" height="80" class="box" />
  <text x="90" y="65" class="title" text-anchor="middle">Input frame</text>
  <text x="90" y="88" class="sub" text-anchor="middle">uploaded image</text>
  <text x="90" y="112" class="sub" text-anchor="middle">(raw)</text>

  <!-- Boxes -->
  <g transform="translate(200,20)">
    <rect x="0" y="20" width="160" height="80" class="box" />
    <text x="80" y="48" class="title" text-anchor="middle">YOLOv8s</text>
    <text x="80" y="68" class="sub" text-anchor="middle">yolo_detector.py</text>
    <text x="80" y="96" class="sub" text-anchor="middle">bboxes</text>
  </g>

  <g transform="translate(390,20)">
    <rect x="0" y="20" width="160" height="80" class="box" />
    <text x="80" y="48" class="title" text-anchor="middle">Crime-scene</text>
    <text x="80" y="68" class="sub" text-anchor="middle">processor.py</text>
    <text x="80" y="96" class="sub" text-anchor="middle">aligned crop</text>
  </g>

  <g transform="translate(580,20)">
    <rect x="0" y="20" width="160" height="80" class="box" />
    <text x="80" y="48" class="title" text-anchor="middle">LandmarkNet</text>
    <text x="80" y="68" class="sub" text-anchor="middle">training/landmark_train.py</text>
    <text x="80" y="96" class="sub" text-anchor="middle">landmarks</text>
  </g>

  <g transform="translate(770,20)">
    <rect x="0" y="20" width="120" height="80" class="box" />
    <text x="60" y="48" class="title" text-anchor="middle">Alignment</text>
    <text x="60" y="76" class="sub" text-anchor="middle">aligned crop</text>
  </g>

  <g transform="translate(920,20)">
    <rect x="0" y="20" width="160" height="80" class="box" />
    <text x="80" y="48" class="title" text-anchor="middle">ReconstructionNet</text>
    <text x="80" y="68" class="sub" text-anchor="middle">training/reconstruction_train.py</text>
    <text x="80" y="96" class="sub" text-anchor="middle">3-D params</text>
  </g>

  <!-- Lower row continuation for recognition and storage -->
  <g transform="translate(200,140)">
    <rect x="0" y="0" width="220" height="80" class="box" />
    <text x="110" y="28" class="title" text-anchor="middle">RecognitionNet</text>
    <text x="110" y="48" class="sub" text-anchor="middle">training/recognition_train.py</text>
    <text x="110" y="68" class="sub" text-anchor="middle">512-D embedding</text>
  </g>

  <g transform="translate(460,140)">
    <rect x="0" y="0" width="260" height="80" class="box" />
    <text x="130" y="28" class="title" text-anchor="middle">Embedding storage</text>
    <text x="130" y="48" class="sub" text-anchor="middle">police_face_db.sqlite / PostgreSQL</text>
    <text x="130" y="68" class="sub" text-anchor="middle">stored vectors</text>
  </g>

  <!-- Arrows -->
  <path class="arrow" d="M160 80 L200 80" />
  <path class="arrow" d="M360 80 L390 80" />
  <path class="arrow" d="M550 80 L580 80" />
  <path class="arrow" d="M740 80 L770 80" />
  <path class="arrow" d="M890 80 L920 80" />

  <path class="arrow" d="M1000 100 L1000 140 L540 140" />
  <path class="arrow" d="M680 140 L740 140" />

  <!-- Caption -->
  <text x="600" y="250" class="sub" text-anchor="middle">Figure: Facial analytics pipeline — detection → landmarks → alignment → reconstruction → recognition → storage</text>
</svg>
'''


def main():
    out = Path('docs/figures')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'facial_analytics_pipeline.svg').write_text(SVG, encoding='utf-8')
    print('Wrote', out / 'facial_analytics_pipeline.svg')


if __name__ == '__main__':
    main()
