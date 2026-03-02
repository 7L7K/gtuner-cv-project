# Changelog

## Unreleased

### Fixed
- Resolve model weight path by checking both `gcv/apex_8n.pt` and `models/apex_8n.pt`, matching repository layout.
- Add runtime fallback from MPS to CPU when MPS initialization fails, and use the selected device for prediction.
- Guard target aspect-ratio filtering against non-positive box dimensions to prevent divide-by-zero runtime warnings.
- Add startup preflight validation that raises a clear `FileNotFoundError` when no model exists in either supported model path.
- Select `mps` only when available (`torch.backends.mps.is_available()`), otherwise initialize on CPU and keep one selected device for inference.

### Added
- Add pinned `requirements.txt` for reproducible local setup.
- Add quick-start and runtime behavior sections to `README.md`.
