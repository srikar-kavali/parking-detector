# Parking Detector

Parking Detector aims to improve parking occupancy detection when an
individual space is hard to classify. For example, part of a vehicle in a
neighboring space can appear inside a vacant space’s image crop and make
that space look occupied.

The project builds on a MobileNetV2 baseline that classifies each marked
space independently in fixed-camera images from CNR-EXT. A graph neural
network (GNN) refines those predictions using visual clues from nearby
spaces in the same frame. We compare the baseline, a refinement model
that uses slot geometry, and the GNN on the same held-out spaces to
measure how much neighbor information helps.

## Project layout

Run the commands below from the repository root. Set up the
[CNR-EXT dataset](https://cnrpark.it/) and the provided checkpoints like this:

```text
parking-detector/
├── README.md
├── main.py                      # Run the detector on one image
├── parking_dataset.py           # Load labeled parking-space crops
├── train.py                     # Train the MobileNetV2 baseline
├── slot_context.py              # Define the GNN and load its checkpoint
├── train_context.py             # Train or evaluate the refinement models
├── evaluate.py                  # Evaluate MobileNetV2 on pre-cropped images
├── evaluate_report.py           # Reproduce the held-out weather report
├── evaluate_blend.py            # Validate and test a geometry/GNN score blend
├── models/
│   ├── parking_classifier.pth   # Matching MobileNetV2 baseline
│   ├── slot_context.pth         # GNN refinement
│   └── slot_geometry.pth        # Refinement without neighbor messages
├── dataset/
│   ├── CNR-EXT-Patches-150x150/
│   │   ├── LABELS/
│   │   │   ├── train.txt
│   │   │   ├── val.txt
│   │   │   └── test.txt
│   │   └── PATCHES/             # Needed for train.py and evaluate.py
│   └── CNR-EXT_FULL_IMAGE_1000x750/
│       ├── camera1.csv … camera9.csv
│       └── FULL_IMAGE_1000x750/
│           ├── SUNNY/
│           ├── OVERCAST/
│           └── RAINY/
└── output/                      # Created by the scripts below
```

Keep the archive’s date and camera subfolders under each weather folder.
Install PyTorch, torchvision, Pillow, and the desktop OpenCV package in your
Python environment if they are missing:

```powershell
python -m pip install torch torchvision pillow opencv-python
```

**You can run the demo and weather evaluation without retraining.** Keep the
provided `parking_classifier.pth` paired with the two refinement checkpoints.
Each refinement checkpoint verifies the exact MobileNetV2 file it was trained
against. If you have `parking_classifier_fast.pth`, it is a separate baseline
experiment.

## Run the bundled demo

The demo works without downloading CNR-EXT.

```bash
python main.py --image dataset/demo/reference.png --camera-csv dataset/demo/parking_camera.csv --blend --no-display
```

Omit `--blend` to run the original GNN. The demo marks low-confidence
slots as uncertain. Benchmark results instead classify every labeled slot
with argmax.

The annotated image is saved to `output/parking_detection.jpg`. To select a
different frame, add `--image "path/to/frame.jpg" --camera-csv "path/to/camera1.csv"`
and choose the CSV for the frame’s camera. Add `--no-display` to save the result
without opening a window.

## Compare predictions side by side

From the repository root, run this example on rainy Camera 1 frame:

`python compare.py --image "dataset/CNR-EXT_FULL_IMAGE_1000x750/FULL_IMAGE_1000x750/RAINY/2016-01-08/camera1/2016-01-08_0919.jpg" --camera-csv "dataset/CNR-EXT_FULL_IMAGE_1000x750/camera1.csv" --labels "dataset/CNR-EXT-Patches-150x150/LABELS/train.txt"`

The left panel will show MobileNetV2 predictions, and the right panel will show MobileNetV2 + GNN predictions. The comparison image is saved to output/parking_comparison.jpg. Add `--no-display` to save it without opening a window. To try a different frame, pass the image path and the CSV for that frame's camera.

## Evaluate by weather

The committed [held-out report](output/held_out_report/report_summary.md)
contains the measured accuracy, occupied-class F1, and confusion counts for
all three models on 896 frames (31,825 labeled parking spaces). The GNN made
151 fewer errors than geometry-only on sunny frames, two fewer on overcast
frames, and one more on rainy frames. These comparisons are by weather, not
a controlled test of weather as a cause. The test cameras also appear in
training or validation.

To reproduce the comparison locally, first place the two CNR-EXT folders
inside `dataset/` as shown above. Keep the original date and camera
subfolders. The full-image archive includes the nine camera CSV files.
Only `LABELS/` is needed from the patch archive.

```bash
python -m pip install -r requirements.txt
python train_context.py --check-data
python evaluate_report.py --output output/held_out_repeat
```

On Apple Silicon, pass `--encoder-device mps` to the evaluation script if
PyTorch reports MPS is available. Use `--encoder-device cpu` to run on CPU.

## Test the score-blending improvement

```bash
python evaluate_blend.py --encoder-device cpu
```

On Apple Silicon, replace `cpu` with `mps` if available. This experiment uses
the existing checkpoints and does not retrain. It tries GNN weights of
0, 0.25, 0.5, 0.75, and 1 on the official validation split. A weight of 0 is
geometry-only and 1 is the original GNN. It selects one weight using mean
balanced accuracy on sunny and overcast validation frames. The official
validation split has no rainy examples, so rainy performance is assessed
only on the held-out test. The script then evaluates only the selected
weight on the held-out test alongside the unchanged baselines. The script
writes `output/blend_eval.csv` with the validation sweep and test results.
Use `--output output/blend_repeat.csv` to run it again without overwriting.

The [completed blend report](output/blend_report/README.md) records the
held-out result. The 50/50 blend reached **99.04% accuracy** versus
**98.88%** for the GNN alone, with **52 fewer errors**. On rainy frames,
false occupied calls fell from **142 to 103** while missed occupied
spaces fell from **79 to 74**. Sunny accuracy was unchanged.

## Training and full-test evaluation

`train.py` trains MobileNetV2. `train_context.py` uses a frozen MobileNetV2
checkpoint and caches full-frame crop features before training the refinement
head. Keep a new baseline and the heads trained against it together when
running a separate experiment.

To evaluate the provided checkpoints on the entire held-out test split:

```powershell
python train_context.py --evaluate --checkpoint models/slot_context.pth
python train_context.py --evaluate --checkpoint models/slot_geometry.pth
```


## Generate the final evaluation report

Use the official CNR-EXT patch archive for the three `LABELS` lists and
the full-image archive for the frames and camera CSVs. Both are available
from the [dataset archive release](https://github.com/fabiocarrara/deep-parking/releases/tag/archive). Place them in the
layout above. The final benchmark does not need the actual patch JPEGs.
[Official CNRPark+EXT dataset](http://cnrpark.it/)

```bash
python train_context.py --check-data
python evaluate_report.py --output output/held_out_repeat
```

The report command evaluates the same slots with the bundled MobileNetV2,
geometry-only head, and four-neighbor GNN. It reads the weights without
training and requires a new output directory for each run. On Apple
Silicon with PyTorch MPS available, add `--encoder-device mps` to run
MobileNetV2 on the GPU. The refinement heads use the detector's default
device.

- `results.md` contains the full-test and weather comparison.
- `metrics.csv` includes frame counts, slot counts, accuracy and occupied
  F1 as fractions, and TN, FP, FN, and TP for each model.
- `predictions.csv` records the ground truth and all three predictions
  for each labeled slot, with 0 for vacant and 1 for occupied.
- `gnn_changes.csv` counts corrected and new errors against each simpler
  model. A slot can count in both comparisons.
- `examples.csv` identifies the first matching slot for each observed
  category in frame-path and numeric slot order. The matching PNGs show
  the original frame and an annotated copy with the target and predictions.
  `gnn_neighbors.png` shows the incoming neighbors for the first saved case.
- `manifest.json` records the command, environment, and hashes of the
  checkpoints, label lists, and used camera layouts. Its presence marks
  a completed run.

The scores use argmax and include all labeled test slots. Treat examples
as illustrations of the measured errors. Claims about occlusion or
generalization to new cameras require a separate evaluation designed
for those questions.
