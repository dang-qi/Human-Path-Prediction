#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import YNetTorch


def add_csdi_to_path(csdi_root: str):
    root = Path(csdi_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"CSDI root does not exist: {root}")
    if str(root) not in sys.path:
        sys.path.append(str(root))
    return root


def build_simulation_loader(args, split: str):
    csdi_root = add_csdi_to_path(args.csdi_root)
    from dataset_augmented_simulation import get_nonaugmented_dataloader_with_scenario_batches

    data_folder = args.data_root
    if data_folder is None:
        data_folder = str(csdi_root / "data" / "simulation_data")

    return get_nonaugmented_dataloader_with_scenario_batches(
        data_length=args.data_length,
        seed=args.seed,
        scenarios=None if args.scenarios in ("", "all", "null", "None") else args.scenarios.split(","),
        batch_size=args.batch_size,
        load_scenario_map=True,
        zero_based_position=True,
        debug=False,
        part=split,
        num_workers=args.num_workers,
        pin_memory=(args.device.startswith("cuda")),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2,
        preprocess_for_resnet=False,
        scen_map_variant=args.scen_map_variant,
        poi_radius=args.poi_radius,
        gen_sdf=False,
        missing_strategy="all_but_two_end",
        missing_ratio=0.1,
        coord_range="zero_one",
        data_folder=data_folder,
    )


def key_indices(length: int, num_keypoints: int, device: torch.device) -> torch.Tensor:
    idx = torch.linspace(0, length - 1, num_keypoints, device=device).round().long().unique()
    if idx.numel() != num_keypoints:
        idx = torch.linspace(0, length - 1, num_keypoints, device=device).long()
    return idx


def waypoint_channels(num_keypoints: int, spec: str) -> list[int]:
    if spec == "auto3":
        vals = torch.linspace(1, num_keypoints - 2, 3).round().long().tolist()
    elif spec == "auto5":
        vals = torch.linspace(1, num_keypoints - 2, 5).round().long().tolist()
    else:
        vals = [int(x) for x in spec.split(",") if x.strip()]
    vals = sorted(set(v for v in vals if 0 <= v < num_keypoints))
    if not vals:
        raise ValueError("At least one waypoint channel is required.")
    return vals


def pad_to_divisor(x: torch.Tensor, divisor: int) -> torch.Tensor:
    h, w = x.shape[-2:]
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h))


def prepare_scenemap(batch: dict, args, device: torch.device):
    scen = batch.get("scen_map_raw", batch.get("scen_map"))
    if scen is None:
        raise ValueError("Batch must contain scen_map or scen_map_raw.")
    scen = scen.to(device).float() if torch.is_tensor(scen) else torch.as_tensor(scen, device=device).float()
    if scen.dim() != 4:
        raise ValueError(f"Expected scenemap batch with 4 dims, got {tuple(scen.shape)}")
    if scen.shape[-1] == 3:
        scen = scen.permute(0, 3, 1, 2)
    if scen.max() > 2.0:
        scen = scen / 255.0

    raw_h, raw_w = int(scen.shape[-2]), int(scen.shape[-1])
    valid_h = max(1, int(round(raw_h / float(args.map_downsample))))
    valid_w = max(1, int(round(raw_w / float(args.map_downsample))))
    scen = F.interpolate(scen, size=(valid_h, valid_w), mode="nearest")
    scen = pad_to_divisor(scen, args.division_factor)
    return scen.clamp(0.0, 1.0), valid_h, valid_w


def coords_to_heatmaps(
    coords: torch.Tensor,
    mask: torch.Tensor,
    out_h: int,
    out_w: int,
    valid_h: int,
    valid_w: int,
    sigma: float,
) -> torch.Tensor:
    coords = coords.clamp(0.0, 1.0)
    bsz, steps, _ = coords.shape
    dtype = coords.dtype
    x_px = coords[..., 0] * max(valid_w - 1, 1)
    y_px = coords[..., 1] * max(valid_h - 1, 1)
    xs = torch.arange(out_w, device=coords.device, dtype=dtype).view(1, 1, 1, out_w)
    ys = torch.arange(out_h, device=coords.device, dtype=dtype).view(1, 1, out_h, 1)
    dx = xs - x_px.view(bsz, steps, 1, 1)
    dy = ys - y_px.view(bsz, steps, 1, 1)
    heat = torch.exp(-0.5 * (dx.pow(2) + dy.pow(2)) / max(float(sigma), 1e-6) ** 2)
    return heat * mask.view(bsz, steps, 1, 1).to(dtype=dtype)


def process_batch(batch: dict, args, device: torch.device):
    scen, valid_h, valid_w = prepare_scenemap(batch, args, device)
    observed_data = batch["observed_data"].to(device).float()
    observed_mask = batch["observed_mask"].to(device).float()
    gt_mask = batch["gt_mask"].to(device).float()

    coords = observed_data
    valid_time = (observed_mask.max(dim=-1).values > 0).float()
    cond_time = (gt_mask.max(dim=-1).values > 0).float()
    target_time = ((observed_mask - gt_mask).max(dim=-1).values > 0).float()

    bsz, length, _ = coords.shape
    idx = key_indices(length, args.num_keypoints, device)
    key_coords = coords[:, idx]
    key_target = target_time[:, idx]
    key_valid = valid_time[:, idx]
    out_h, out_w = scen.shape[-2:]

    target_heatmaps = coords_to_heatmaps(
        key_coords, key_valid, out_h, out_w, valid_h, valid_w, args.sigma_pixels
    ) * key_target.view(bsz, -1, 1, 1)

    time_ids = torch.arange(length, device=device).view(1, length)
    first_idx = torch.where(cond_time > 0, time_ids, torch.full_like(time_ids, length)).min(dim=1).values
    last_idx = torch.where(cond_time > 0, time_ids, torch.full_like(time_ids, -1)).max(dim=1).values
    first_idx = first_idx.clamp(0, length - 1)
    last_idx = last_idx.clamp(0, length - 1)
    batch_idx = torch.arange(bsz, device=device)
    cond_coords = torch.stack([coords[batch_idx, first_idx], coords[batch_idx, last_idx]], dim=1)
    cond_maps = coords_to_heatmaps(
        cond_coords,
        torch.ones(bsz, 2, device=device),
        out_h,
        out_w,
        valid_h,
        valid_w,
        args.sigma_pixels,
    )

    model_input = torch.cat([scen, cond_maps], dim=1)
    return {
        "input": model_input,
        "target_heatmaps": target_heatmaps,
        "key_target": key_target,
        "key_idx": idx,
        "coords": coords,
        "target_time": target_time.bool(),
        "gt_mask": gt_mask.permute(0, 2, 1),
        "observed_data_bcl": observed_data.permute(0, 2, 1),
        "valid_h": valid_h,
        "valid_w": valid_w,
    }


def masked_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    mask = key_mask.view(key_mask.shape[0], key_mask.shape[1], 1, 1).to(dtype=loss.dtype)
    denom = (mask.sum() * logits.shape[-2] * logits.shape[-1]).clamp_min(1.0)
    return (loss * mask).sum() / denom


def downsample_condition_maps(maps: torch.Tensor, num_features: int):
    outs = [maps]
    for i in range(1, num_features):
        outs.append(F.avg_pool2d(maps, kernel_size=2**i, stride=2**i))
    return outs


def interpolate_key_coords(key_coords: torch.Tensor, key_idx: torch.Tensor, length: int) -> torch.Tensor:
    bsz = key_coords.shape[0]
    out = key_coords.new_zeros(bsz, length, 2)
    for j in range(key_idx.numel() - 1):
        start = int(key_idx[j].item())
        end = int(key_idx[j + 1].item())
        weights = torch.linspace(0.0, 1.0, end - start + 1, device=key_coords.device, dtype=key_coords.dtype)
        seg = key_coords[:, j : j + 1] * (1.0 - weights.view(1, -1, 1))
        seg = seg + key_coords[:, j + 1 : j + 2] * weights.view(1, -1, 1)
        out[:, start : end + 1] = seg
    return out


def decode_logits(model: YNetTorch, logits: torch.Tensor, valid_h: int, valid_w: int) -> torch.Tensor:
    logits = logits[..., :valid_h, :valid_w].contiguous()
    coords_px = model.softargmax(logits)
    denom = coords_px.new_tensor([max(valid_w - 1, 1), max(valid_h - 1, 1)]).view(1, 1, 2)
    return (coords_px / denom).clamp(0.0, 1.0)


def forward_model(model: YNetTorch, proc: dict, wp_channels: list[int], teacher_forcing: bool, temperature: float):
    features = model.pred_features(proc["input"])
    waypoint_logits = model.pred_goal(features)
    if teacher_forcing:
        waypoint_maps = proc["target_heatmaps"][:, wp_channels]
    else:
        waypoint_maps = torch.sigmoid(waypoint_logits[:, wp_channels] / max(float(temperature), 1e-6))
    waypoint_down = downsample_condition_maps(waypoint_maps, len(features))
    traj_input = [torch.cat([feat, wp], dim=1) for feat, wp in zip(features, waypoint_down)]
    traj_logits = model.pred_traj(traj_input)
    return waypoint_logits, traj_logits


def train_one_epoch(model, loader, optimizer, args, device, wp_channels, epoch: int):
    model.train()
    total_loss = 0.0
    total_goal = 0.0
    total_traj = 0.0
    count = 0
    iterator = tqdm(loader, desc=f"train {epoch}", mininterval=5.0)
    for batch_no, batch in enumerate(iterator, start=1):
        if args.max_train_batches and batch_no > args.max_train_batches:
            break
        proc = process_batch(batch, args, device)
        waypoint_logits, traj_logits = forward_model(
            model, proc, wp_channels, teacher_forcing=True, temperature=args.temperature
        )
        goal_loss = masked_bce_with_logits(waypoint_logits, proc["target_heatmaps"], proc["key_target"])
        traj_loss = masked_bce_with_logits(traj_logits, proc["target_heatmaps"], proc["key_target"])
        loss = (goal_loss + traj_loss) * args.loss_scale

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        total_loss += float(loss.detach())
        total_goal += float(goal_loss.detach())
        total_traj += float(traj_loss.detach())
        count += 1
        iterator.set_postfix(loss=total_loss / count, goal=total_goal / count, traj=total_traj / count)
    return {
        "loss": total_loss / max(count, 1),
        "goal_bce": total_goal / max(count, 1),
        "traj_bce": total_traj / max(count, 1),
    }


@torch.no_grad()
def evaluate(model, loader, args, device, wp_channels, split: str):
    model.eval()
    mse_total = 0.0
    mae_total = 0.0
    eval_total = 0.0
    count = 0
    iterator = tqdm(loader, desc=split, mininterval=5.0)
    for batch_no, batch in enumerate(iterator, start=1):
        if args.max_valid_batches and batch_no > args.max_valid_batches:
            break
        proc = process_batch(batch, args, device)
        _, traj_logits = forward_model(model, proc, wp_channels, teacher_forcing=False, temperature=args.temperature)
        key_coords = decode_logits(model, traj_logits, proc["valid_h"], proc["valid_w"])

        key_cond = ~proc["key_target"].bool()
        gt_key = proc["coords"][:, proc["key_idx"]]
        key_coords[key_cond] = gt_key[key_cond]
        full = interpolate_key_coords(key_coords, proc["key_idx"], proc["coords"].shape[1])
        mask = proc["target_time"].unsqueeze(-1)
        resid = (full - proc["coords"]) * mask
        mse_total += float((resid.pow(2)).sum())
        mae_total += float(resid.abs().sum())
        eval_total += float(mask.sum() * 2)
        count += 1
        rmse = (mse_total / max(eval_total, 1.0)) ** 0.5
        iterator.set_postfix(rmse=rmse, mae=mae_total / max(eval_total, 1.0))
    return {
        "RMSE": (mse_total / max(eval_total, 1.0)) ** 0.5,
        "MAE": mae_total / max(eval_total, 1.0),
        "batches": count,
    }


def build_model(args, wp_channels):
    return YNetTorch(
        obs_len=2,
        pred_len=args.num_keypoints,
        segmentation_model_fp=None,
        use_features_only=False,
        semantic_classes=3,
        encoder_channels=[32, 32, 64, 64, 64],
        decoder_channels=[64, 64, 64, 32, 32],
        waypoints=len(wp_channels),
    )


def save_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Y-Net style two-end simulation baseline")
    parser.add_argument("--csdi-root", default="/Users/qida0163/research/track_generation/CSDI_new")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--scenarios", default="all")
    parser.add_argument("--output-dir", default="./save/simulation_two_end_ynet")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--data-length", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--loss-scale", type=float, default=1000.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-keypoints", type=int, default=50)
    parser.add_argument("--waypoint-channels", default="auto3")
    parser.add_argument("--sigma-pixels", type=float, default=1.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--map-downsample", type=float, default=8.0)
    parser.add_argument("--division-factor", type=int, default=32)
    parser.add_argument("--scen-map-variant", default="merged_last_empty_with_poi")
    parser.add_argument("--poi-radius", type=int, default=5)
    parser.add_argument("--valid-every", type=int, default=1)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-valid-batches", type=int, default=0)
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument("--final-eval-split", default="test", choices=["valid", "test"])
    parser.add_argument("--final-viz-num", type=int, default=40)
    parser.add_argument("--skip-final-viz", action="store_true")
    parser.add_argument("--save-final-outputs", action="store_true")
    return parser.parse_args()


def run_final_artifacts(args, checkpoint: Path, out_dir: Path):
    common = [
        "--csdi-root",
        args.csdi_root,
        "--checkpoint",
        str(checkpoint),
        "--split",
        args.final_eval_split,
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--data-length",
        str(args.data_length),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--num-keypoints",
        str(args.num_keypoints),
        "--waypoint-channels",
        args.waypoint_channels,
        "--sigma-pixels",
        str(args.sigma_pixels),
        "--temperature",
        str(args.temperature),
        "--map-downsample",
        str(args.map_downsample),
        "--division-factor",
        str(args.division_factor),
        "--scen-map-variant",
        args.scen_map_variant,
        "--poi-radius",
        str(args.poi_radius),
        "--scenarios",
        args.scenarios,
    ]
    if args.data_root is not None:
        common.extend(["--data-root", args.data_root])

    eval_cmd = [
        sys.executable,
        str(ROOT / "evaluate_simulation_two_end.py"),
        *common,
        "--output-dir",
        str(out_dir),
    ]
    if args.save_final_outputs:
        eval_cmd.append("--save-outputs")
    print("Running final collision evaluation:", " ".join(eval_cmd), flush=True)
    subprocess.run(eval_cmd, check=True)

    if not args.skip_final_viz and args.final_viz_num > 0:
        viz_dir = out_dir / f"{args.final_eval_split}_viz"
        viz_cmd = [
            sys.executable,
            str(ROOT / "visualize_simulation_two_end.py"),
            *common,
            "--output-dir",
            str(viz_dir),
            "--num",
            str(args.final_viz_num),
        ]
        print("Running final visualization:", " ".join(viz_cmd), flush=True)
        subprocess.run(viz_cmd, check=True)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wp_channels = waypoint_channels(args.num_keypoints, args.waypoint_channels)
    train_loader = build_simulation_loader(args, "train")
    valid_loader = build_simulation_loader(args, "valid")

    model = build_model(args, wp_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    config = vars(args).copy()
    config["waypoint_channels_resolved"] = wp_channels
    save_json(out_dir / "config.json", config)

    best_rmse = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, args, device, wp_channels, epoch)
        row = {"epoch": epoch, "train": train_metrics, "seconds": time.time() - t0}

        if epoch % args.valid_every == 0:
            valid_metrics = evaluate(model, valid_loader, args, device, wp_channels, "valid")
            row["valid"] = valid_metrics
            if valid_metrics["RMSE"] < best_rmse:
                best_rmse = valid_metrics["RMSE"]
                torch.save(model.state_dict(), out_dir / "model_best.pt")
                row["best"] = True
            else:
                row["best"] = False
        print(json.dumps(row, indent=2), flush=True)
        history.append(row)
        save_json(out_dir / "history.json", history)

    torch.save(model.state_dict(), out_dir / "model_last.pt")
    save_json(out_dir / "summary.json", {"best_valid_RMSE": best_rmse, "history": history[-5:]})
    if not args.skip_final_eval:
        checkpoint = out_dir / "model_best.pt"
        if not checkpoint.exists():
            checkpoint = out_dir / "model_last.pt"
        run_final_artifacts(args, checkpoint, out_dir)


if __name__ == "__main__":
    main()
