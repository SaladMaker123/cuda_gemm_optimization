#!/usr/bin/env bash
set -euo pipefail

# GEMM project environment: GeForce RTX 50 series + CUDA 13.0 + Nsight Compute 2025.2.0
CUDA_PKG="cuda-toolkit-13-0"
CUDA_VERSION="13.0.3-1"
NCU_PKG="nsight-compute-2025.2.0"
NCU_VERSION="2025.2.0.11-1"
CUDA_DIR="/usr/local/cuda-13.0/bin"
NCU_DIR="/opt/nvidia/nsight-compute/2025.2.0"

if [[ "${EUID}" -ne 0 ]]; then
    echo "ERROR: Run this script as root (or with sudo)."
    exit 1
fi

# Target environment: Ubuntu 22.04/24.04 x86-64 with GeForce RTX 50 series.
source /etc/os-release
if [[ "${ID}" != "ubuntu" || "${VERSION_ID}" != "22.04" && "${VERSION_ID}" != "24.04" ]]; then
    echo "ERROR: Ubuntu 22.04 or 24.04 is required."
    exit 1
fi
if [[ "$(uname -m)" != "x86_64" ]]; then
    echo "ERROR: This script requires x86_64."
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi is unavailable. Check the GPU driver/VM first."
    exit 1
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
echo "GPU: ${GPU_NAME}"
echo "Driver: ${DRIVER_VERSION}"
if [[ "${GPU_NAME}" != *"RTX 5050"* &&
      "${GPU_NAME}" != *"RTX 5060"* &&
      "${GPU_NAME}" != *"RTX 5070"* &&
      "${GPU_NAME}" != *"RTX 5080"* &&
      "${GPU_NAME}" != *"RTX 5090"* ]]; then
    echo "ERROR: This project requires GeForce RTX 50 series."
    exit 1
fi
if [[ "${DRIVER_VERSION%%.*}" -lt 580 ]]; then
    echo "ERROR: CUDA 13.x needs NVIDIA driver 580 or newer."
    echo "The script will NOT replace the cloud GPU driver."
    exit 1
fi

# Refuse silent version changes: benchmark environments must be reproducible.
for entry in "${CUDA_PKG}:${CUDA_VERSION}" "${NCU_PKG}:${NCU_VERSION}"; do
    pkg="${entry%%:*}"
    expected="${entry#*:}"
    installed="$(dpkg-query -W -f='${Version}' "${pkg}" 2>/dev/null || true)"
    if [[ -n "${installed}" && "${installed}" != "${expected}" ]]; then
        echo "ERROR: ${pkg} is ${installed}, expected ${expected}."
        echo "Review the version mismatch instead of silently changing it."
        exit 1
    fi
done

CUDA_INSTALLED="$(dpkg-query -W -f='${Status}' "${CUDA_PKG}" 2>/dev/null || true)"
NCU_INSTALLED="$(dpkg-query -W -f='${Status}' "${NCU_PKG}" 2>/dev/null || true)"
if [[ "${CUDA_INSTALLED}" != "install ok installed" || "${NCU_INSTALLED}" != "install ok installed" ]]; then
    apt-get update

    # Add NVIDIA's official CUDA repository only if the pinned packages are absent.
    if ! apt-cache madison "${CUDA_PKG}" | awk '{print $3}' | grep -Fxq "${CUDA_VERSION}" || \
       ! apt-cache madison "${NCU_PKG}" | awk '{print $3}' | grep -Fxq "${NCU_VERSION}"; then
        apt-get install -y ca-certificates wget
        distro="ubuntu${VERSION_ID//./}"
        keyring="$(mktemp --suffix=.deb)"
        wget -q -O "${keyring}" \
            "https://developer.download.nvidia.com/compute/cuda/repos/${distro}/x86_64/cuda-keyring_1.1-1_all.deb"
        dpkg -i "${keyring}"
        rm -f "${keyring}"
        apt-get update
    fi

    apt-get install -y \
        "${CUDA_PKG}=${CUDA_VERSION}" \
        "${NCU_PKG}=${NCU_VERSION}"
else
    echo "Pinned CUDA Toolkit and Nsight Compute are already installed."
fi

# Enable commands now and in future root SSH sessions.
export PATH="${NCU_DIR}:${CUDA_DIR}:${PATH}"
ENV_LINE='export PATH=/opt/nvidia/nsight-compute/2025.2.0:/usr/local/cuda-13.0/bin:$PATH'
touch /root/.bashrc
if ! grep -Fxq "${ENV_LINE}" /root/.bashrc; then
    printf '\n%s\n' "${ENV_LINE}" >> /root/.bashrc
fi

nvcc --version
ncu --version

if ! nvcc --version | grep -Fq "release 13.0"; then
    echo "ERROR: CUDA Toolkit version mismatch."
    exit 1
fi

if ! ncu --version | grep -Fq "Version 2025.2.0.0"; then
    echo "ERROR: Nsight Compute version mismatch."
    exit 1
fi
# A real profiling run is needed: ncu --version alone cannot confirm permission.
TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TEMP_DIR}"' EXIT
cat > "${TEMP_DIR}/ncu_test.cu" <<'CUDA_EOF'
#include <cuda_runtime.h>
#include <cstdio>

__global__ void test(float* data)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    data[i] = i * 2.0f;
}

int main()
{
    float* data = nullptr;
    if (cudaMalloc(&data, 1024 * sizeof(float)) != cudaSuccess) return 1;
    test<<<4, 256>>>(data);
    if (cudaGetLastError() != cudaSuccess) return 2;
    if (cudaDeviceSynchronize() != cudaSuccess) return 3;
    if (cudaFree(data) != cudaSuccess) return 4;
    return 0;
}
CUDA_EOF

nvcc -std=c++17 -arch=sm_120 \
    "${TEMP_DIR}/ncu_test.cu" -o "${TEMP_DIR}/ncu_test"

echo "Testing Nsight Compute performance-counter permission..."
if ncu --set basic --launch-count 1 "${TEMP_DIR}/ncu_test" > "${TEMP_DIR}/ncu.log" 2>&1; then
    if grep -q 'Section:' "${TEMP_DIR}/ncu.log"; then
        echo "PASS: Nsight Compute collected GPU performance metrics."
    else
        cat "${TEMP_DIR}/ncu.log"
        echo "ERROR: ncu exited successfully but no metric section was found."
        exit 1
    fi
else
    cat "${TEMP_DIR}/ncu.log"
    if grep -q 'ERR_NVGPUCTRPERM' "${TEMP_DIR}/ncu.log"; then
        echo "ERROR: The VM/provider did not grant GPU performance-counter permission."
    else
        echo "ERROR: Nsight Compute profiling failed for another reason."
    fi
    exit 1
fi
echo "Environment ready: ${GPU_NAME} + CUDA 13.0 + Nsight Compute 2025.2.0."
