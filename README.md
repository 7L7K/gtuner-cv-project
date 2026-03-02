# GTuner CV Project (GCV + GPC)

This repo contains:
- **GCV** Python script(s) for GTuner IV Computer Vision
- **GPC** script(s) for GTuner IV
- **Model weights** used by the CV pipeline

## Folder layout
- `gcv/` – Python GCV scripts
- `gpc/` – GPC scripts
- `models/` – model weights

## Notes
GTuner IV on macOS (Rosetta/x86_64) typically needs x86_64 Python deps.

## Quick Start
1. Clone and pull LFS model weights:
   - `git clone https://github.com/ziahziahziah/gtuner-cv-project.git`
   - `cd gtuner-cv-project`
   - `git lfs pull`
2. Create a virtualenv and install dependencies:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -r requirements.txt`
3. Run local regression checks:
   - `python3 -m unittest discover -s tests -p 'test_*.py' -v`
4. In GTuner IV, load:
   - `gcv/MY_CV.py` as the Computer Vision script
   - `gpc/gtuner_script.gpc` as the Titan Two script

## Runtime Behavior
- Model path resolution checks both `gcv/apex_8n.pt` and `models/apex_8n.pt`.
- Startup now raises a clear `FileNotFoundError` if no model is found in either location.
- Device selection uses MPS when available, otherwise CPU fallback.
