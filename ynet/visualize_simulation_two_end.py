#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_simulation_two_end import scenmap_for_collision
from train_simulation_two_end import (
    build_model,
    build_simulation_loader,
    decode_logits,
    forward_model,
    interpolate_key_coords,
    process_batch,
    waypoint_channels,
)


def map_to_rgb(map_img) -> np.ndarray:
    arr = map_img.detach().cpu().numpy() if torch.is_tensor(map_img) else np.asarray(map_img)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D map image, got {arr.shape}")
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    arr = arr.astype(np.float32)
    if arr.max() <= 1.0:
        arr = arr * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 3:
        arr = arr[..., ::-1]
    return arr


def normalized_to_pixels(coords: np.ndarray, h: int, w: int) -> list[tuple[float, float]]:
    pts = np.asarray(coords, dtype=np.float32)
    xs = np.clip(pts[:, 0] * float(w), 0.0, max(float(w - 1), 0.0))
    ys = np.clip(pts[:, 1] * float(h), 0.0, max(float(h - 1), 0.0))
    return [(float(x), float(y)) for x, y in zip(xs, ys)]


def compute_bad_points(pred_norm: torch.Tensor, eval_mask: torch.Tensor, scen_chw: torch.Tensor):
    h, w = int(scen_chw.shape[-2]), int(scen_chw.shape[-1])
    collision_mask = ((scen_chw[0] > 0) | (scen_chw[1] > 0)).bool()
    x = pred_norm[:, 0] * w
    y = pred_norm[:, 1] * h
    finite = torch.isfinite(x) & torch.isfinite(y)
    oob = finite & ((x < 0) | (x >= (w - 1e-6)) | (y < 0) | (y >= (h - 1e-6)))
    xi = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).long().clamp(0, w - 1)
    yi = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).long().clamp(0, h - 1)
    hit = collision_mask[yi, xi] & finite & (~oob)
    return (eval_mask.bool() & ((~finite) | oob | hit)).detach().cpu().numpy()


def draw_overlay(map_img, pred, target, observed, eval_mask, observed_mask, bad_points, title: str):
    rgb = map_to_rgb(map_img)
    h, w = rgb.shape[:2]
    canvas = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(canvas)

    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    observed_np = observed.detach().cpu().numpy()
    eval_np = eval_mask.detach().cpu().numpy().astype(bool)
    obs_np = observed_mask.detach().cpu().numpy().astype(bool)

    pred_xy = normalized_to_pixels(pred_np[eval_np | obs_np], h, w)
    target_xy = normalized_to_pixels(target_np[eval_np], h, w)
    obs_xy = normalized_to_pixels(observed_np[obs_np], h, w)
    bad_xy = normalized_to_pixels(pred_np[bad_points], h, w)

    if len(target_xy) >= 2:
        draw.line(target_xy, fill=(255, 0, 255), width=2)
    if len(pred_xy) >= 2:
        draw.line(pred_xy, fill=(255, 215, 0), width=3)

    for x, y in bad_xy:
        r = 3
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0))
    for x, y in obs_xy:
        r = 5
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(64, 224, 208))
    if target_xy:
        for x, y in (target_xy[0], target_xy[-1]):
            r = 4
            draw.rectangle((x - r, y - r, x + r, y + r), outline=(255, 255, 255), width=2)

    draw.rectangle((8, 8, min(w - 8, 8 + 8 * len(title)), 28), fill=(0, 0, 0))
    draw.text((12, 12), title, fill=(255, 255, 255))
    draw_legend(draw, w)
    return np.asarray(canvas)


def draw_legend(draw: ImageDraw.ImageDraw, w: int):
    items = [
        ("Pred", (255, 215, 0), "line"),
        ("GT target", (255, 0, 255), "line"),
        ("Start/End", (64, 224, 208), "dot"),
        ("Collision/OOB", (255, 0, 0), "dot"),
    ]
    x0 = max(8, w - 150)
    y0 = 8
    row_h = 18
    draw.rectangle((x0 - 6, y0 - 4, w - 8, y0 + row_h * len(items) + 4), fill=(0, 0, 0))
    for i, (label, color, kind) in enumerate(items):
        y = y0 + i * row_h + 8
        if kind == "line":
            draw.line((x0, y, x0 + 24, y), fill=color, width=3)
        else:
            r = 4
            draw.ellipse((x0 + 8 - r, y - r, x0 + 8 + r, y + r), fill=color)
        draw.text((x0 + 32, y - 6), label, fill=(255, 255, 255))


@torch.no_grad()
def collect_examples(model, loader, args, device, wp_channels):
    rng = random.Random(args.seed)
    examples = []
    seen = 0
    global_index = 0
    model.eval()

    for batch_no, batch in enumerate(tqdm(loader, desc=args.split, mininterval=5.0), start=1):
        if args.max_batches and batch_no > args.max_batches:
            break
        proc = process_batch(batch, args, device)
        _, traj_logits = forward_model(model, proc, wp_channels, teacher_forcing=False, temperature=args.temperature)
        key_coords = decode_logits(model, traj_logits, proc["valid_h"], proc["valid_w"])
        key_cond = ~proc["key_target"].bool()
        gt_key = proc["coords"][:, proc["key_idx"]]
        key_coords[key_cond] = gt_key[key_cond]
        full = interpolate_key_coords(key_coords, proc["key_idx"], proc["coords"].shape[1])

        scen = scenmap_for_collision(batch, device)
        map_source = batch.get("scen_map_raw", batch.get("scen_map"))
        obs_mask_time = (batch["gt_mask"].to(device).float().max(dim=-1).values > 0)
        eval_mask_time = proc["target_time"]
        scenario_id = batch.get("scenario_id", None)

        bsz = int(full.shape[0])
        for i in range(bsz):
            replace = None
            if len(examples) < args.num:
                replace = len(examples)
            else:
                j = rng.randint(0, seen)
                if j < args.num:
                    replace = j
            seen += 1
            if replace is None:
                global_index += 1
                continue

            bad = compute_bad_points(full[i], eval_mask_time[i], scen[i])
            ex = {
                "map": map_source[i].detach().cpu().clone() if torch.is_tensor(map_source) else np.array(map_source[i]),
                "pred": full[i].detach().cpu().clone(),
                "target": proc["coords"][i].detach().cpu().clone(),
                "observed": batch["observed_data"][i].detach().cpu().clone(),
                "eval_mask": eval_mask_time[i].detach().cpu().clone(),
                "observed_mask": obs_mask_time[i].detach().cpu().clone(),
                "bad": bad,
                "scenario_id": None if scenario_id is None else int(scenario_id[i]),
                "global_index": global_index,
                "bad_count": int(np.asarray(bad).sum()),
                "eval_count": int(eval_mask_time[i].sum().item()),
            }
            if replace == len(examples):
                examples.append(ex)
            else:
                examples[replace] = ex
            global_index += 1
    return examples


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Y-Net simulation two-end predictions")
    parser.add_argument("--csdi-root", default="/Users/qida0163/research/track_generation/CSDI_new")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num", type=int, default=20)
    parser.add_argument("--data-length", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-keypoints", type=int, default=50)
    parser.add_argument("--waypoint-channels", default="auto3")
    parser.add_argument("--sigma-pixels", type=float, default=1.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--map-downsample", type=float, default=8.0)
    parser.add_argument("--division-factor", type=int, default=32)
    parser.add_argument("--scen-map-variant", default="merged_last_empty_with_poi")
    parser.add_argument("--poi-radius", type=int, default=5)
    parser.add_argument("--scenarios", default="all")
    parser.add_argument("--max-batches", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    wp_channels = waypoint_channels(args.num_keypoints, args.waypoint_channels)

    loader = build_simulation_loader(args, args.split)
    model = build_model(args, wp_channels).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    examples = collect_examples(model, loader, args, device, wp_channels)
    manifest = []
    for rank, ex in enumerate(examples, start=1):
        sid = "NA" if ex["scenario_id"] is None else f"{ex['scenario_id']:02d}"
        title = f"sid={sid} idx={ex['global_index']} bad={ex['bad_count']}/{ex['eval_count']}"
        img = draw_overlay(
            ex["map"],
            ex["pred"],
            ex["target"],
            ex["observed"],
            ex["eval_mask"],
            ex["observed_mask"],
            ex["bad"],
            title,
        )
        name = f"{rank:02d}_sid{sid}_g{ex['global_index']:05d}_bad{ex['bad_count']:03d}.png"
        path = out_dir / name
        Image.fromarray(img, mode="RGB").save(path)
        manifest.append(
            {
                "file": str(path),
                "scenario_id": ex["scenario_id"],
                "global_index": ex["global_index"],
                "bad_count": ex["bad_count"],
                "eval_count": ex["eval_count"],
            }
        )

    with (out_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {len(manifest)} visualization(s) to {out_dir}")


if __name__ == "__main__":
    main()
