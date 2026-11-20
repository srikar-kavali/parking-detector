"""Select a geometry/GNN logit blend on validation, then evaluate it on test."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

WEATHERS = ("SUNNY", "OVERCAST", "RAINY")
GROUPS = ("FULL", *WEATHERS)
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
FIELDS = ("split", "weather", "model", "gnn_weight", "frames", "slots",
          "accuracy", "balanced_accuracy", "occupied_f1", "tn", "fp", "fn", "tp")


def metrics(counts):
    tn, fp, fn, tp = counts
    total = sum(counts)
    return {
        "accuracy": (tn + tp) / total if total else None,
        "balanced_accuracy": (tn / (tn + fp) + tp / (tp + fn)) / 2
        if (tn + fp) and (tp + fn) else None,
        "occupied_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def selection_weathers(counts):
    # Eligibility is determined by validation labels, never test labels.
    return tuple(weather for weather in WEATHERS
                 if metrics(counts[(weather, "Blend", 0.0)])["balanced_accuracy"]
                 is not None)


def select_weight(counts):
    groups = selection_weathers(counts)
    if not groups:
        raise ValueError("Validation has no weather group with both classes.")

    def score(weight):
        values = [metrics(counts[(weather, "Blend", weight)])["balanced_accuracy"]
                  for weather in groups]
        return mean(values), weight

    return max(WEIGHTS, key=score)


def collect(frames, detector, encoder, heads, encoder_device, candidates):
    import cv2
    import torch
    from slot_context import SlotContext, slot_geometry
    from train_context import full_frame_path

    counts = defaultdict(lambda: [0, 0, 0, 0])
    frame_counts = dict.fromkeys(GROUPS, 0)
    with torch.inference_mode():
        for i, (frame, labels) in enumerate(sorted(frames.items()), 1):
            weather = frame.parts[0]
            if weather not in WEATHERS:
                raise ValueError(f"Unexpected weather in {frame}")
            image_path = full_frame_path(detector.DATA_DIR, frame)
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not read {image_path}")
            csv_path = detector.DATA_DIR.parent / f"{frame.parent.name}.csv"
            crops, slots = detector.extract_frame_slots(image, csv_path)
            by_id = {int(slot[0]["id"]): index for index, slot in enumerate(slots)}
            missing = labels.keys() - by_id.keys()
            if missing:
                raise ValueError(f"{frame} is missing usable slots {sorted(missing)}")
            device = detector.device
            indexes = torch.tensor([by_id[slot] for slot in labels], device=device)
            truth = torch.tensor(list(labels.values()), device=device)
            geometry = slot_geometry(slots, image.shape).to(device)
            crops = crops.to(encoder_device)
            if encoder_device.type == "cpu":
                crops = crops.contiguous(memory_format=torch.channels_last)
            features, base_scores = SlotContext.encode(encoder, crops)
            features, base_scores = features.to(device), base_scores.to(device)
            geo, gnn = (head(features, base_scores, geometry) for head in heads)
            if not all(torch.isfinite(scores).all().item() for scores in (base_scores, geo, gnn)):
                raise ValueError(f"Non-finite model scores in {frame}")
            scores = {"MobileNetV2": base_scores, "Geometry-only": geo, "GNN": gnn}
            for group in ("FULL", weather):
                frame_counts[group] += 1
            for name, weight in candidates:
                logits = (geo if weight == 0 else gnn if weight == 1
                          else torch.lerp(geo, gnn, weight)) if name == "Blend" else scores[name]
                pred = logits.index_select(0, indexes).argmax(dim=1)
                bins = torch.bincount((truth * 2 + pred).cpu(), minlength=4).tolist()
                for group in ("FULL", weather):
                    row = counts[(group, name, weight)]
                    for j in range(4):
                        row[j] += bins[j]
            if i % 100 == 0 or i == len(frames):
                print(f"Scored {i}/{len(frames)} frames", flush=True)
    return counts, frame_counts


def rows(split, result, candidates):
    counts, frames = result
    for group in GROUPS:
        for name, weight in candidates:
            bins = counts[(group, name, weight)]
            yield dict(split=split, weather=group, model=name, gnn_weight=weight,
                       frames=frames[group], slots=sum(bins), **metrics(bins))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=Path("output/blend_eval.csv"))
    args = parser.parse_args()

    import torch
    import main as detector
    from slot_context import load_context
    from train_context import check_inputs, read_frames, verify_splits

    if args.output.exists():
        parser.error(f"{args.output} already exists. Use a different --output.")
    if args.encoder_device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is not available. Use --encoder-device cpu.")
    if args.encoder_device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available. Use --encoder-device cpu.")

    labels = detector.PROJECT_DIR / "dataset" / "CNR-EXT-Patches-150x150" / "LABELS"
    train, val, test = (read_frames(labels / f"{split}.txt") for split in ("train", "val", "test"))
    verify_splits((train, val, test))
    check_inputs(val, detector.DATA_DIR, detector.DATA_DIR.parent)
    check_inputs(test, detector.DATA_DIR, detector.DATA_DIR.parent)
    encoder, mapping = detector.load_model()
    if mapping != {"free": 0, "busy": 1}:
        raise ValueError(f"Unexpected model class mapping: {mapping}")
    encoder_device = torch.device(args.encoder_device)
    encoder = encoder.to(encoder_device).eval()
    if encoder_device.type == "cpu":
        encoder = encoder.to(memory_format=torch.channels_last)
    models = detector.MODEL_PATH.parent
    heads = [load_context(models / name, detector.MODEL_PATH, detector.device)
             for name in ("slot_geometry.pth", "slot_context.pth")]
    if tuple(head.neighbors for head in heads) != (0, 4):
        raise ValueError("Expected the supplied geometry-only and four-neighbor checkpoints.")

    val_candidates = [("Blend", weight) for weight in WEIGHTS]
    print("Validation sweep", flush=True)
    validation = collect(val, detector, encoder, heads, encoder_device, val_candidates)
    print("Validation selection groups: " + ", ".join(selection_weathers(validation[0])), flush=True)
    weight = select_weight(validation[0])
    print(f"Selected GNN weight on validation: {weight:g}", flush=True)

    test_candidates = [("MobileNetV2", None), ("Geometry-only", 0.0),
                       ("GNN", 1.0), ("Blend", weight)]
    print("Held-out test", flush=True)
    held_out = collect(test, detector, encoder, heads, encoder_device, test_candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows("validation", validation, val_candidates))
        writer.writerows(rows("test", held_out, test_candidates))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
