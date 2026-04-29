# Simulation Two-End Y-Net Baseline

This is a simulation-specific entry point for using the Y-Net heatmap training setup on the CSDI simulation data.
It leaves the original Y-Net `model.py`, `train.py`, and `test.py` unchanged.

## What It Trains

The input is:

```text
simulation scenemap RGB channels + start heatmap + end heatmap
```

The output is:

```text
K key-time trajectory heatmaps
```

The loss follows the Y-Net style:

```text
BCEWithLogitsLoss(goal/keypoint heatmaps) + BCEWithLogitsLoss(trajectory heatmaps)
```

The trajectory decoder is conditioned on a small set of GT waypoint heatmaps during training, and on predicted
waypoint probabilities during validation.

## Smoke Test

Run from the repository root:

```bash
conda run -n lab1 python ynet/train_simulation_two_end.py \
  --device cpu \
  --epochs 1 \
  --batch-size 2 \
  --num-workers 0 \
  --max-train-batches 1 \
  --max-valid-batches 1 \
  --output-dir /tmp/ynet_sim_smoke
```

## Full Run Example

```bash
conda run -n lab1 python ynet/train_simulation_two_end.py \
  --device cuda:0 \
  --epochs 50 \
  --batch-size 32 \
  --num-workers 8 \
  --output-dir ./save/simulation_two_end_ynet
```

Useful options:

```text
--map-downsample 8          Downsample scenemap/heatmap resolution before Y-Net.
--num-keypoints 50          Number of key-time heatmaps.
--waypoint-channels auto3   Waypoint channels used to condition the trajectory decoder.
--sigma-pixels 1.5          Gaussian target heatmap width at downsampled resolution.
--loss-scale 1000           Matches the original Y-Net convention.
```

By default, the script reads simulation data from:

```text
/Users/qida0163/research/track_generation/CSDI_new/data/simulation_data
```

Override with `--csdi-root` or `--data-root` if needed.

## Alvis SLURM Run

Copy the public-safe template to a local script and fill in your own project ID and cluster paths:

```bash
cp ynet/sbatch_simulation_two_end_alvis.template.sh ynet/sbatch_simulation_two_end_alvis.local.sh
$EDITOR ynet/sbatch_simulation_two_end_alvis.local.sh
```

The template uses placeholders:

```text
#SBATCH -A <PROJECT_ID>
CONTAINER=<PATH_TO_CONTAINER_SIF>
YNET_DIR=<PATH_TO_HUMAN_PATH_PREDICTION>
CSDI_DIR=<PATH_TO_CSDI_NEW>
DATA_DIR=${CSDI_DIR}/data/simulation_data
```

`*.local.sh` is ignored by git so private cluster details are not committed.

It copies `DATA_DIR` to local SSD as:

```text
$TMPDIR/simulation_data
```

and passes it to training as:

```text
--data-root /data/simulation_data
```

You can override common settings without editing the script:

```bash
RUN_NAME=simulation_two_end_ynet_smoke \
EPOCHS=1 \
BATCH_SIZE=8 \
NUM_WORKERS=4 \
sbatch ynet/sbatch_simulation_two_end_alvis.local.sh
```

Outputs are saved by default to:

```text
$YNET_DIR/save/$RUN_NAME
```
