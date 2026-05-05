#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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


def parse_int_list(value: str) -> set[int]:
    if not value:
        return set()
    return {int(x.strip()) for x in value.split(",") if x.strip()}


def heatmap_to_uint8(heat: torch.Tensor, out_h: int, out_w: int) -> np.ndarray:
    heat = heat.detach().float().unsqueeze(0).unsqueeze(0)
    heat = F.interpolate(heat, size=(out_h, out_w), mode="bilinear", align_corners=False).squeeze()
    arr = heat.cpu().numpy()
    lo, hi = np.percentile(arr, [1.0, 99.5])
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    arr = (arr - lo) / max(hi - lo, 1e-8)
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def colorize_heatmap(gray: np.ndarray) -> np.ndarray:
    x = gray.astype(np.float32) / 255.0
    rgb = np.zeros((*gray.shape, 3), dtype=np.float32)
    rgb[..., 0] = np.clip(2.0 * x - 0.2, 0.0, 1.0)
    rgb[..., 1] = np.clip(2.0 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    rgb[..., 2] = np.clip(1.3 - 2.0 * x, 0.0, 1.0)
    return (rgb * 255.0).astype(np.uint8)


def blend_heatmap(map_img, heat: torch.Tensor) -> Image.Image:
    base = map_to_rgb(map_img)
    h, w = base.shape[:2]
    gray = heatmap_to_uint8(heat, h, w)
    color = colorize_heatmap(gray)
    alpha = (gray.astype(np.float32) / 255.0 * 0.75)[..., None]
    blended = base.astype(np.float32) * (1.0 - alpha) + color.astype(np.float32) * alpha
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")


def crop_valid_heatmap(heat: torch.Tensor, ex: dict) -> torch.Tensor:
    return heat[..., : int(ex["valid_h"]), : int(ex["valid_w"])]


def draw_point(draw: ImageDraw.ImageDraw, coord_norm: torch.Tensor, h: int, w: int, color, shape: str):
    xy = normalized_to_pixels(coord_norm.detach().cpu().numpy().reshape(1, 2), h, w)[0]
    x, y = xy
    r = 4
    if shape == "square":
        draw.rectangle((x - r, y - r, x + r, y + r), outline=color, width=2)
    elif shape == "cross":
        draw.line((x - r, y - r, x + r, y + r), fill=color, width=2)
        draw.line((x - r, y + r, x + r, y - r), fill=color, width=2)
    else:
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=2)


def choose_heatmap_channels(ex: dict, args) -> list[int]:
    explicit = parse_int_list(args.heatmap_channels)
    if explicit:
        return sorted(ch for ch in explicit if 0 <= ch < len(ex["key_idx"]))

    selected = {0, len(ex["key_idx"]) - 1, *ex["wp_channels"]}
    err = torch.linalg.norm(ex["key_coords"] - ex["gt_key"], dim=-1)
    target_mask = ex["key_target"].bool()
    err = torch.where(target_mask, err, torch.zeros_like(err))
    topk = min(int(args.heatmap_topk), int(err.numel()))
    if topk > 0:
        for ch in torch.topk(err, k=topk).indices.tolist():
            selected.add(int(ch))
            if ch > 0:
                selected.add(int(ch - 1))
            if ch + 1 < len(ex["key_idx"]):
                selected.add(int(ch + 1))
    return sorted(ch for ch in selected if 0 <= ch < len(ex["key_idx"]))


def compute_jump_mask(pred: torch.Tensor, eval_mask: torch.Tensor, observed_mask: torch.Tensor, topk: int = 4) -> np.ndarray:
    active = (eval_mask | observed_mask).bool()
    out = np.zeros(int(pred.shape[0]), dtype=bool)
    if int(active.sum().item()) < 3:
        return out
    active_idx = torch.where(active)[0]
    pts = pred[active_idx]
    step = torch.linalg.norm(pts[1:] - pts[:-1], dim=-1)
    if step.numel() == 0:
        return out
    count = min(int(topk), int(step.numel()))
    threshold = max(float(torch.quantile(step, 0.95).item()), float(step.mean().item() + 2.0 * step.std().item()))
    chosen = torch.where(step >= threshold)[0]
    if chosen.numel() == 0:
        chosen = torch.topk(step, k=count).indices
    for j in chosen[:count].tolist():
        out[int(active_idx[j].item())] = True
        out[int(active_idx[j + 1].item())] = True
    return out


def draw_heatmap_panel(ex: dict, args, title: str) -> np.ndarray:
    channels = choose_heatmap_channels(ex, args)
    if not channels:
        raise ValueError("No heatmap channels selected.")

    panels = []
    for ch in channels:
        traj_logits = crop_valid_heatmap(ex["traj_logits"][ch], ex)
        traj = torch.sigmoid(traj_logits / max(float(args.temperature), 1e-6))
        canvas = blend_heatmap(ex["map"], traj)
        draw = ImageDraw.Draw(canvas)
        h, w = canvas.size[1], canvas.size[0]

        draw_point(draw, ex["key_coords_raw"][ch], h, w, (255, 255, 0), "circle")
        draw_point(draw, ex["key_coords"][ch], h, w, (255, 255, 255), "square")
        draw_point(draw, ex["gt_key"][ch], h, w, (255, 0, 255), "cross")

        label = f"traj k={ch} t={int(ex['key_idx'][ch])} target={int(ex['key_target'][ch].item())}"
        draw.rectangle((6, 6, min(w - 6, 6 + 7 * len(label)), 24), fill=(0, 0, 0))
        draw.text((10, 10), label, fill=(255, 255, 255))
        panels.append(canvas)

        if ch in ex["wp_channels"]:
            wp_logits = crop_valid_heatmap(ex["waypoint_logits"][ch], ex)
            wp = torch.sigmoid(wp_logits / max(float(args.temperature), 1e-6))
            wp_canvas = blend_heatmap(ex["map"], wp)
            wp_draw = ImageDraw.Draw(wp_canvas)
            draw_point(wp_draw, ex["waypoint_coords"][ch], h, w, (255, 255, 0), "circle")
            draw_point(wp_draw, ex["gt_key"][ch], h, w, (255, 0, 255), "cross")
            wp_label = f"waypoint k={ch} t={int(ex['key_idx'][ch])}"
            wp_draw.rectangle((6, 6, min(w - 6, 6 + 7 * len(wp_label)), 24), fill=(0, 0, 0))
            wp_draw.text((10, 10), wp_label, fill=(255, 255, 255))
            panels.append(wp_canvas)

    base_w, base_h = panels[0].size
    cols = max(1, int(args.heatmap_cols))
    rows = int(np.ceil(len(panels) / cols))
    sheet = Image.new("RGB", (base_w * cols, base_h * rows + 28), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    legend = f"{title}  yellow=raw softargmax white=used keypoint magenta=GT"
    draw.text((10, 8), legend, fill=(255, 255, 255))
    for n, panel in enumerate(panels):
        x = (n % cols) * base_w
        y = 28 + (n // cols) * base_h
        sheet.paste(panel, (x, y))
    return np.asarray(sheet)


def draw_overlay(
    map_img,
    pred,
    target,
    observed,
    eval_mask,
    observed_mask,
    bad_points,
    title: str,
    key_idx: torch.Tensor | None = None,
    key_coords: torch.Tensor | None = None,
    jump_mask: np.ndarray | None = None,
):
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
    jump_xy = normalized_to_pixels(pred_np[jump_mask], h, w) if jump_mask is not None else []
    jump_indices = np.where(jump_mask)[0].tolist() if jump_mask is not None else []

    if len(target_xy) >= 2:
        draw.line(target_xy, fill=(255, 0, 255), width=2)
    if len(pred_xy) >= 2:
        draw.line(pred_xy, fill=(255, 215, 0), width=3)

    if key_idx is not None and key_coords is not None:
        key_np = key_coords.detach().cpu().numpy()
        key_xy = normalized_to_pixels(key_np, h, w)
        for x, y in key_xy:
            r = 2
            draw.ellipse((x - r, y - r, x + r, y + r), outline=(255, 255, 0), width=1)

    for x, y in bad_xy:
        r = 3
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0))
    for point_idx, (x, y) in zip(jump_indices, jump_xy):
        r = 5
        draw.rectangle((x - r, y - r, x + r, y + r), outline=(255, 255, 255), width=2)
        label = f"t={point_idx}"
        if key_idx is not None:
            key_np = key_idx.detach().cpu().numpy().astype(np.int64)
            exact = np.where(key_np == int(point_idx))[0]
            if exact.size:
                label = f"k={int(exact[0])} t={point_idx}"
            else:
                right = int(np.searchsorted(key_np, int(point_idx), side="right"))
                left = max(0, right - 1)
                right = min(len(key_np) - 1, right)
                label = f"k{left}->{right} t={point_idx}"
        tx = min(max(x + 7, 0), max(w - 64, 0))
        ty = min(max(y - 14, 0), max(h - 16, 0))
        draw.rectangle((tx - 2, ty - 2, tx + 7 * len(label) + 2, ty + 12), fill=(0, 0, 0))
        draw.text((tx, ty), label, fill=(255, 255, 255))
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
        ("Large jump", (255, 255, 255), "box"),
    ]
    x0 = max(8, w - 150)
    y0 = 8
    row_h = 18
    draw.rectangle((x0 - 6, y0 - 4, w - 8, y0 + row_h * len(items) + 4), fill=(0, 0, 0))
    for i, (label, color, kind) in enumerate(items):
        y = y0 + i * row_h + 8
        if kind == "line":
            draw.line((x0, y, x0 + 24, y), fill=color, width=3)
        elif kind == "box":
            draw.rectangle((x0 + 4, y - 4, x0 + 12, y + 4), outline=color, width=2)
        else:
            r = 4
            draw.ellipse((x0 + 8 - r, y - r, x0 + 8 + r, y + r), fill=color)
        draw.text((x0 + 32, y - 6), label, fill=(255, 255, 255))


@torch.no_grad()
def collect_examples(model, loader, args, device, wp_channels):
    rng = random.Random(args.seed)
    only_indices = parse_int_list(args.only_indices)
    examples = []
    seen = 0
    global_index = 0
    model.eval()

    for batch_no, batch in enumerate(tqdm(loader, desc=args.split, mininterval=5.0), start=1):
        if args.max_batches and batch_no > args.max_batches:
            break
        proc = process_batch(batch, args, device)
        waypoint_logits, traj_logits = forward_model(
            model, proc, wp_channels, teacher_forcing=False, temperature=args.temperature
        )
        waypoint_coords = decode_logits(model, waypoint_logits, proc["valid_h"], proc["valid_w"])
        key_coords_raw = decode_logits(model, traj_logits, proc["valid_h"], proc["valid_w"])
        key_coords = key_coords_raw.clone()
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
            if only_indices:
                replace = len(examples) if global_index in only_indices else None
            else:
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
            jump = compute_jump_mask(full[i], eval_mask_time[i], obs_mask_time[i])
            ex = {
                "map": map_source[i].detach().cpu().clone() if torch.is_tensor(map_source) else np.array(map_source[i]),
                "pred": full[i].detach().cpu().clone(),
                "target": proc["coords"][i].detach().cpu().clone(),
                "observed": batch["observed_data"][i].detach().cpu().clone(),
                "waypoint_logits": waypoint_logits[i].detach().cpu().clone(),
                "traj_logits": traj_logits[i].detach().cpu().clone(),
                "waypoint_coords": waypoint_coords[i].detach().cpu().clone(),
                "key_coords_raw": key_coords_raw[i].detach().cpu().clone(),
                "key_coords": key_coords[i].detach().cpu().clone(),
                "gt_key": gt_key[i].detach().cpu().clone(),
                "key_target": proc["key_target"][i].detach().cpu().clone(),
                "key_idx": proc["key_idx"].detach().cpu().clone(),
                "valid_h": int(proc["valid_h"]),
                "valid_w": int(proc["valid_w"]),
                "wp_channels": list(wp_channels),
                "eval_mask": eval_mask_time[i].detach().cpu().clone(),
                "observed_mask": obs_mask_time[i].detach().cpu().clone(),
                "bad": bad,
                "jump": jump,
                "scenario_id": None if scenario_id is None else int(scenario_id[i]),
                "global_index": global_index,
                "bad_count": int(np.asarray(bad).sum()),
                "eval_count": int(eval_mask_time[i].sum().item()),
                "jump_count": int(np.asarray(jump).sum()),
            }
            if replace == len(examples):
                examples.append(ex)
            else:
                examples[replace] = ex
            global_index += 1
            if only_indices and only_indices.issubset({int(e["global_index"]) for e in examples}):
                return sorted(examples, key=lambda item: item["global_index"])
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
    parser.add_argument("--only-indices", default="", help="Comma-separated global_index values to visualize exactly.")
    parser.add_argument("--save-heatmaps", action="store_true", help="Also save waypoint/traj heatmap diagnostic panels.")
    parser.add_argument("--heatmap-channels", default="", help="Comma-separated keypoint channels to draw; default selects high-error channels.")
    parser.add_argument("--heatmap-topk", type=int, default=6, help="Number of largest keypoint-error traj heatmaps to include.")
    parser.add_argument("--heatmap-cols", type=int, default=2)
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
            key_idx=ex["key_idx"],
            key_coords=ex["key_coords"],
            jump_mask=ex["jump"],
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
                "jump_count": ex["jump_count"],
            }
        )
        if args.save_heatmaps:
            heat = draw_heatmap_panel(ex, args, title)
            heat_name = f"{rank:02d}_sid{sid}_g{ex['global_index']:05d}_heatmaps.png"
            heat_path = out_dir / heat_name
            Image.fromarray(heat, mode="RGB").save(heat_path)
            manifest[-1]["heatmaps"] = str(heat_path)

    with (out_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {len(manifest)} visualization(s) to {out_dir}")


if __name__ == "__main__":
    main()
