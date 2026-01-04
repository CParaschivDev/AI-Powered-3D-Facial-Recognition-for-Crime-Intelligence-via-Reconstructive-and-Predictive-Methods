"""
Simple repository scanner to find potentially unbounded or risky patterns.
Generates `scan_report.json` in the repo root with entries: file, line, context, pattern, severity.

Run: python tools\repo_scan.py
"""
import re
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'scan_report.json'

# Patterns to search: (name, regex, severity)
PATTERNS = [
    ("infinite_while", r"while\s+True", "high"),
    ("read_whole_file", r"\.read\s*\(", "medium"),
    ("requests_call", r"requests\.(get|post|put|delete)\s*\(", "medium"),
    ("socket_recv", r"\.recv\s*\(", "high"),
    ("socket_accept", r"socket\.accept\s*\(", "high"),
    ("subprocess_call", r"subprocess\.(Popen|run|call|check_output)\s*\(", "medium"),
    ("queue_use", r"\bqueue\.Queue\s*\(", "medium"),
    ("threadpool_use", r"ThreadPoolExecutor\s*\(|ProcessPoolExecutor\s*\(", "medium"),
    ("append", r"\.append\s*\(", "low"),
    ("extend", r"\.extend\s*\(", "low"),
    ("torch_cat_accumulate", r"torch\.cat\s*\(", "medium"),
    ("np_concatenate", r"np\.concatenate\s*\(", "medium"),
    ("open_no_with", r"\bopen\s*\(.*\)\s*\.read\b", "low"),
    ("wide_glob", r"glob\.glob\s*\(", "low"),
    ("scipy_loadmat", r"scipy\.io\.loadmat\s*\(", "low"),
]

# File globs to include
INCLUDE_GLOBS = ['**/*.py']
EXCLUDE_DIRS = ['.git', '__pycache__', 'node_modules', 'reports', 'frontend', 'batch_debug_flame_v2']

results = []

for g in INCLUDE_GLOBS:
    for path in ROOT.glob(g):
        if any(p in path.parts for p in EXCLUDE_DIRS):
            continue
        try:
            text = path.read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            continue
        lines = text.splitlines()
        for i, line in enumerate(lines):
            for name, pat, severity in PATTERNS:
                for m in re.finditer(pat, line):
                    # capture small context
                    start = max(0, i-3)
                    end = min(len(lines), i+4)
                    context = "\n".join(lines[start:end])
                    results.append({
                        'file': str(path.relative_to(ROOT)),
                        'line_no': i+1,
                        'pattern': name,
                        'snippet': line.strip(),
                        'severity': severity,
                        'context': context
                    })

# Sort results by severity then file
sev_order = {'high': 0, 'medium': 1, 'low': 2}
results.sort(key=lambda r: (sev_order.get(r['severity'], 9), r['file'], r['line_no']))

report = {
    'root': str(ROOT),
    'total_matches': len(results),
    'matches': results
}

OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(f"Scan complete. Matches={len(results)}. Report written to {OUT}")
