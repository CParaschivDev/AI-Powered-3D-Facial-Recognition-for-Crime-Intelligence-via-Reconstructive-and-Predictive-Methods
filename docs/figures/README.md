Security & Governance Figure

This folder contains a script to generate the Security & Governance stack diagram used in the documentation.

Files
- `security_governance_stack.py`: Python script that generates an SVG (or PNG) diagram using the `graphviz` Python package.

Requirements
- Python 3.8+
- Graphviz system binaries (install via your OS package manager or from https://graphviz.org/download/)
- Python package `graphviz`

Quick start

Windows (PowerShell):

```powershell
pip install graphviz
# ensure Graphviz dot is on PATH; for Chocolatey: choco install graphviz
python security_governance_stack.py --output security_stack.svg
```

Linux/macOS:

```bash
pip install graphviz
# install graphviz via apt/brew: sudo apt install graphviz  OR brew install graphviz
python security_governance_stack.py --output security_stack.svg
```

Notes
- The script annotates arrows with protections such as `TLS + JWT`, `AES-256 at rest / envelope encryption`, and `Redacted JSON logs`.
- Adjust node labels and styling in `security_governance_stack.py` as needed to match your documentation style.
