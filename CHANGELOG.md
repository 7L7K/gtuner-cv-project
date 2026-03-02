# Changelog

## Unreleased

### Fixed
- Resolve model weight path by checking both `gcv/apex_8n.pt` and `models/apex_8n.pt`, matching repository layout.
- Add runtime fallback from MPS to CPU when MPS initialization fails, and use the selected device for prediction.
- Guard target aspect-ratio filtering against non-positive box dimensions to prevent divide-by-zero runtime warnings.
