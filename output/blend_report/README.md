# Held-out score-blending experiment

We tested whether averaging the geometry-only and GNN logits improves the
parking detector without retraining. Both heads shared the original frozen
MobileNetV2 encoder and the supplied checkpoints.

## Protocol

We tried GNN weights of 0, 0.25, 0.50, 0.75, and 1.0 on the official
validation split (524 frames, 18,647 labeled slots). We selected the weight
with the highest mean balanced accuracy across the two available conditions,
sunny and overcast. **The official validation split contains no rainy
examples.** The selected weight was 0.50, with 98.26% mean validation
balanced accuracy compared with 97.86% for GNN alone. We kept that weight
fixed for the 896 test frames, including the 363 rainy frames. Every
labeled slot was classified with argmax.

## Held-out results

| Condition | Slots | MobileNetV2 accuracy | Geometry-only accuracy | GNN accuracy | 50/50 blend accuracy | Fewer errors vs. GNN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sunny | 12,600 | 92.06% | 97.99% | 99.19% | 99.19% | 0 |
| Overcast | 6,365 | 96.12% | 99.45% | 99.48% | 99.61% | 8 |
| Rainy | 12,860 | 94.90% | 98.29% | 98.28% | 98.62% | 44 |
| Overall | 31,825 | 94.02% | 98.40% | 98.88% | 99.04% | 52 |

Overall, the blend reduced GNN errors from 356 to 304 and increased
occupied-class F1 from 99.03% to 99.17%. On rain, it reduced false occupied
calls from 142 to 103 and missed occupied calls from 79 to 74. Rainy
balanced accuracy increased from 97.82% to 98.33%. Sunny accuracy was
unchanged, with seven more false occupied calls and seven fewer missed
occupied calls.

The improvement is modest in absolute accuracy and measured on a single
test split. The test cameras also appear in training or validation.
This does not establish a causal weather effect or performance on new lots.

## Reproduce

See [metrics.csv](metrics.csv) for the full validation sweep and test
confusion matrices. The original three models reproduced every confusion
count in the [archived report](../held_out_report/report_summary.md).
The label, layout, and checkpoint hashes matched the archived manifest.

```bash
python evaluate_blend.py --encoder-device cpu --output output/blend_repeat.csv
```

The published run used PyTorch 2.15.0+cpu, torchvision 0.29.2+cpu,
OpenCV 5.1.0, and NumPy 2.5.4.
