#!/bin/bash
set -e

# ============================================================================
# LIBERO Dataset Download to S3
# ============================================================================
# Downloads LIBERO datasets (original HDF5 + OpenVLA modified RLDS) and uploads to S3
#
# Usage:
#   ./scripts/download_libero_to_s3.sh
#
# Requirements:
#   - AWS credentials configured
#   - huggingface-cli installed (pip install huggingface-hub)
#   - git-lfs installed (for HuggingFace datasets)
# ============================================================================

S3_BUCKET="ml-research-bys"
S3_PREFIX="libero"
LOCAL_DATA_DIR="/workspace/data/libero"
LIBERO_REPO_DIR="/workspace/third_party/openpi/third_party/libero"

echo "============================================"
echo "LIBERO Dataset Download to S3"
echo "============================================"
echo "S3 Bucket: s3://${S3_BUCKET}/${S3_PREFIX}"
echo "Local Dir: ${LOCAL_DATA_DIR}"
echo "============================================"
echo ""

# Create directories
mkdir -p "${LOCAL_DATA_DIR}/datasets"
mkdir -p "${LOCAL_DATA_DIR}/openvla_rlds"

# ============================================================================
# 1. Download LIBERO Original Datasets (HDF5)
# ============================================================================
echo "=== [1/3] Downloading LIBERO Original Datasets (HDF5) ==="

# Clone or update LIBERO repository if needed
if [ ! -d "${LIBERO_REPO_DIR}" ]; then
    echo "LIBERO repository not found at ${LIBERO_REPO_DIR}"
    echo "Cloning LIBERO repository..."
    mkdir -p "$(dirname ${LIBERO_REPO_DIR})"
    cd "$(dirname ${LIBERO_REPO_DIR})"
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git libero
    cd libero
    pip install -e . --quiet
else
    echo "LIBERO repository found at ${LIBERO_REPO_DIR}"
fi

cd "${LIBERO_REPO_DIR}"

# Download datasets
DATASETS=("libero_spatial" "libero_object" "libero_goal" "libero_10")

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "--- Downloading ${DATASET} ---"

    # Check if already exists in S3
    if aws s3 ls "s3://${S3_BUCKET}/${S3_PREFIX}/datasets/${DATASET}/" 2>/dev/null; then
        echo "✓ ${DATASET} already exists in S3, skipping download"
        continue
    fi

    # Download dataset
    python3 benchmark_scripts/download_libero_datasets.py \
        --datasets "${DATASET}" \
        --folder "${LOCAL_DATA_DIR}/datasets" || {
        echo "Warning: Failed to download ${DATASET}, continuing..."
        continue
    }

    # Upload to S3
    echo "Uploading ${DATASET} to S3..."
    aws s3 sync "${LOCAL_DATA_DIR}/datasets/${DATASET}" \
        "s3://${S3_BUCKET}/${S3_PREFIX}/datasets/${DATASET}/" \
        --quiet

    echo "✓ ${DATASET} uploaded to S3"

    # Clean up local copy to save space
    echo "Cleaning up local copy..."
    rm -rf "${LOCAL_DATA_DIR}/datasets/${DATASET}"
done

echo ""
echo "✓ LIBERO Original Datasets Complete"

# ============================================================================
# 2. Download OpenVLA Modified LIBERO (RLDS)
# ============================================================================
echo ""
echo "=== [2/3] Downloading OpenVLA Modified LIBERO (RLDS) ==="

# Check if already exists in S3
if aws s3 ls "s3://${S3_BUCKET}/${S3_PREFIX}/openvla_rlds/" 2>/dev/null | grep -q "."; then
    echo "✓ OpenVLA RLDS dataset already exists in S3, skipping download"
else
    echo "Downloading from HuggingFace: openvla/modified_libero_rlds"

    # Install git-lfs if not available
    if ! command -v git-lfs &> /dev/null; then
        echo "git-lfs not found, installing..."
        sudo apt-get update -qq
        sudo apt-get install -y git-lfs
        git lfs install
    fi

    # Clean up existing directory if present
    if [ -d "${LOCAL_DATA_DIR}/openvla_rlds" ]; then
        echo "Cleaning up existing openvla_rlds directory..."
        rm -rf "${LOCAL_DATA_DIR}/openvla_rlds"
    fi
    mkdir -p "${LOCAL_DATA_DIR}/openvla_rlds"

    # Clone HuggingFace dataset repository
    cd "${LOCAL_DATA_DIR}/openvla_rlds"

    # Use huggingface-cli to download
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download openvla/modified_libero_rlds \
            --repo-type dataset \
            --local-dir . \
            --quiet || {
            echo "Warning: huggingface-cli download failed, trying git clone..."
            rm -rf .git  # Clean any partial git data
            git clone https://huggingface.co/datasets/openvla/modified_libero_rlds .
        }
    else
        echo "huggingface-cli not found, using git clone..."
        git clone https://huggingface.co/datasets/openvla/modified_libero_rlds .
    fi

    # Upload to S3
    echo "Uploading OpenVLA RLDS to S3..."
    aws s3 sync . "s3://${S3_BUCKET}/${S3_PREFIX}/openvla_rlds/" \
        --exclude ".git/*" \
        --quiet

    echo "✓ OpenVLA RLDS uploaded to S3"

    # Clean up local copy
    cd ..
    rm -rf openvla_rlds
fi

echo ""
echo "✓ OpenVLA Modified RLDS Complete"

# ============================================================================
# 3. Summary
# ============================================================================
echo ""
echo "============================================"
echo "Download Complete!"
echo "============================================"
echo ""
echo "S3 Structure:"
echo "  s3://${S3_BUCKET}/${S3_PREFIX}/"
echo "  ├── datasets/"
echo "  │   ├── libero_spatial/  (HDF5)"
echo "  │   ├── libero_object/   (HDF5)"
echo "  │   ├── libero_goal/     (HDF5)"
echo "  │   └── libero_10/       (HDF5)"
echo "  └── openvla_rlds/        (RLDS)"
echo ""
echo "Total Size:"
aws s3 ls "s3://${S3_BUCKET}/${S3_PREFIX}/" --recursive --summarize | tail -2
echo ""
echo "============================================"
