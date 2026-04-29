#!/bin/bash
#SBATCH -A <PROJECT_ID>
#SBATCH -p alvis
#SBATCH --gpus-per-node=A100:1
#SBATCH -t 24:00:00
#SBATCH -o ./srun_logs/output_%j.log

set -euo pipefail

# Copy this file to sbatch_simulation_two_end_alvis.local.sh and fill in
# the paths for your own cluster account. Do not commit the local file.
CONTAINER=<PATH_TO_CONTAINER_SIF>
YNET_DIR=<PATH_TO_HUMAN_PATH_PREDICTION>
CSDI_DIR=<PATH_TO_CSDI_NEW>
DATA_DIR=${CSDI_DIR}/data/simulation_data

RUN_NAME=${RUN_NAME:-simulation_two_end_ynet_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-${YNET_DIR}/save/${RUN_NAME}}

EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-8}
DATA_LENGTH=${DATA_LENGTH:-200}
NUM_KEYPOINTS=${NUM_KEYPOINTS:-50}
MAP_DOWNSAMPLE=${MAP_DOWNSAMPLE:-8}
SIGMA_PIXELS=${SIGMA_PIXELS:-1.5}
WAYPOINT_CHANNELS=${WAYPOINT_CHANNELS:-auto3}

mkdir -p "${YNET_DIR}/srun_logs"
mkdir -p "${OUTPUT_DIR}"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Run name: ${RUN_NAME}"
echo "Y-Net dir: ${YNET_DIR}"
echo "CSDI dir: ${CSDI_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Copying simulation data to local SSD..."

rm -rf "${TMPDIR}/simulation_data"
cp -r "${DATA_DIR}" "${TMPDIR}/simulation_data"

cd "${YNET_DIR}"

echo "Starting Y-Net simulation two-end baseline..."
apptainer exec --nv \
    --bind "${TMPDIR}:/data" \
    --bind "${YNET_DIR}:${YNET_DIR}" \
    --bind "${CSDI_DIR}:${CSDI_DIR}" \
    "${CONTAINER}" \
    python ynet/train_simulation_two_end.py \
        --csdi-root "${CSDI_DIR}" \
        --data-root /data/simulation_data \
        --device cuda:0 \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}" \
        --data-length "${DATA_LENGTH}" \
        --num-keypoints "${NUM_KEYPOINTS}" \
        --map-downsample "${MAP_DOWNSAMPLE}" \
        --sigma-pixels "${SIGMA_PIXELS}" \
        --waypoint-channels "${WAYPOINT_CHANNELS}" \
        --output-dir "${OUTPUT_DIR}"

echo "Finished Y-Net simulation two-end baseline."
echo "Run folder: ${OUTPUT_DIR}"
