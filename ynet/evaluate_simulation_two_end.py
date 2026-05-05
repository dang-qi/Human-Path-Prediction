#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_simulation_two_end import (
    build_model,
    build_simulation_loader,
    decode_logits,
    forward_model,
    interpolate_key_coords,
    process_batch,
    waypoint_channels,
)


def add_csdi_to_path(csdi_root: str):
    root = Path(csdi_root).expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.append(str(root))
    return root


def load_collision_evaluator(csdi_root: str):
    root = add_csdi_to_path(csdi_root)
    utils_path = root / "utils.py"
    spec = importlib.util.spec_from_file_location("csdi_eval_utils", utils_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load CSDI utils from {utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CollisionEvaluator


def scenmap_for_collision(batch: dict, device: torch.device) -> torch.Tensor:
    scen = batch.get("scen_map_raw", batch.get("scen_map"))
    if scen is None:
        raise ValueError("Batch must contain scen_map or scen_map_raw for collision evaluation.")
    scen = scen.to(device).float() if torch.is_tensor(scen) else torch.as_tensor(scen, device=device).float()
    if scen.dim() == 4 and scen.shape[-1] == 3:
        scen = scen.permute(0, 3, 1, 2)
    return scen


def scale_for_collision(batch: dict, device: torch.device) -> torch.Tensor:
    scale = batch.get("scen_map_scale", None)
    if scale is None:
        raise ValueError("Batch must contain scen_map_scale for collision evaluation.")
    return scale.to(device).float() if torch.is_tensor(scale) else torch.as_tensor(scale, device=device).float()


@torch.no_grad()
def evaluate_checkpoint(model, loader, args, device, wp_channels):
    CollisionEvaluator = load_collision_evaluator(args.csdi_root)

    model.eval()
    collision = CollisionEvaluator()
    mse_total = 0.0
    mae_total = 0.0
    eval_total = 0.0

    all_samples = []
    all_target = []
    all_eval = []
    all_observed = []
    all_time = []

    for batch_no, batch in enumerate(tqdm(loader, desc=args.split, mininterval=5.0), start=1):
        if args.max_batches and batch_no > args.max_batches:
            break
        proc = process_batch(batch, args, device)
        _, traj_logits = forward_model(model, proc, wp_channels, teacher_forcing=False, temperature=args.temperature)
        key_coords = decode_logits(model, traj_logits, proc["valid_h"], proc["valid_w"])

        key_cond = proc["key_cond"].bool()
        gt_key = proc["coords"][:, proc["key_idx"]]
        key_coords[key_cond] = gt_key[key_cond]
        full = interpolate_key_coords(key_coords, proc["key_idx"], proc["coords"].shape[1])

        eval_mask_time = proc["target_time"].unsqueeze(-1)
        resid = (full - proc["coords"]) * eval_mask_time
        mse_total += float(resid.pow(2).sum())
        mae_total += float(resid.abs().sum())
        eval_total += float(eval_mask_time.sum() * 2)

        samples_batch = full
        target_batch = proc["coords"]
        eval_points = eval_mask_time.expand_as(target_batch).float()
        observed_points = batch["observed_data"].to(device).float()
        observed_time = batch["timepoints"].to(device).float()

        collision.update(
            samples_batch,
            scenmap_for_collision(batch, device),
            eval_points,
            scale_for_collision(batch, device),
            mode="normalized",
            coord_range="zero_one",
        )
        collision.update_sample_set(
            samples_batch.unsqueeze(1),
            scenmap_for_collision(batch, device),
            eval_points,
            scale_for_collision(batch, device),
            mode="normalized",
            coord_range="zero_one",
        )

        if args.save_outputs:
            all_samples.append(samples_batch.unsqueeze(1).cpu())
            all_target.append(target_batch.cpu())
            all_eval.append(eval_points.cpu())
            all_observed.append(observed_points.cpu())
            all_time.append(observed_time.cpu())

    metrics = {
        "RMSE": (mse_total / max(eval_total, 1.0)) ** 0.5,
        "MAE": mae_total / max(eval_total, 1.0),
    }
    metrics.update(collision.compute_metrics())

    outputs = None
    if args.save_outputs:
        outputs = [
            torch.cat(all_samples, dim=0),
            torch.cat(all_target, dim=0),
            torch.cat(all_eval, dim=0),
            torch.cat(all_observed, dim=0),
            torch.cat(all_time, dim=0),
            1,
            0,
        ]
    return metrics, outputs


def write_metrics_csv(path: Path, metrics: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for key, value in metrics.items():
            writer.writerow([key, value])


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Y-Net style simulation two-end baseline")
    parser.add_argument("--csdi-root", default="/Users/qida0163/research/track_generation/CSDI_new")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1)
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
    parser.add_argument("--save-outputs", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    wp_channels = waypoint_channels(args.num_keypoints, args.waypoint_channels)

    loader = build_simulation_loader(args, args.split)
    model = build_model(args, wp_channels).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)

    metrics, outputs = evaluate_checkpoint(model, loader, args, device, wp_channels)
    print(json.dumps(metrics, indent=2), flush=True)

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.split}_nsample1_normalized"
    write_metrics_csv(out_dir / f"result_{tag}.csv", metrics)
    with (out_dir / f"summary_{tag}.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    if args.save_outputs and outputs is not None:
        import pickle

        with (out_dir / f"generated_outputs_{tag}.pk").open("wb") as f:
            pickle.dump(outputs, f)


if __name__ == "__main__":
    main()
