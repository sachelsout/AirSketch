# AirSketch Quantitative Evaluation Results

Targets: MPJPE < 8 px | Jitter < 2 px² | Gesture Acc > 92%

Bold values indicate target met.

| Metric | Target | FreiHAND Test | Custom Gesture |
|--------|--------|---------------|----------------|
| MPJPE (px) ↓ | < 8.0 | 30.26 ✗ | **6.29** ✓ |
| MPJPE P50 (px) | — | 31.11 | 4.99 |
| MPJPE P95 (px) | — | 57.98 | 17.48 |
| Jitter Index (px²) ↓ | < 2.0 | 115.84 ✗ | 53.64 ✗ |
| Gesture Acc (%) ↑ | > 92 | **100.00** ✓ | 79.69 ✗ |

## Notes
- FreiHAND test = in-distribution benchmark (19,536 windows)
- Custom gesture = OOD benchmark (28,124 windows across 4 sessions)
- MPJPE computed in pixel space on 224×224 frames
- Jitter index = variance of frame-to-frame displacement magnitudes
- Gesture accuracy = binary draw/idle classification over all windows