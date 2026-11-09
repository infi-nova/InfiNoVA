# InfiNoVA: splatting (InstantSplatPP, Python 3.10 + CUDA extensions) and
# dataset/training (LeRobot, Python >=3.12) live in two separate conda
# environments inside this image, matching the two-environment setup
# documented in README.md. Use `conda run -n <env> ...` to invoke each half.
#
# Build (from the infinova repo root, with submodules checked out):
#   docker build -t infinova .
#
# Run with GPU access:
#   docker run --gpus all -it -v $(pwd):/workspace/infinova infinova
#
# Requires the NVIDIA Container Toolkit on the host for --gpus all to work.

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-compute,utility,graphics}

RUN apt-get update -y && apt-get install -y --no-install-recommends \
        git wget curl ca-certificates \
        libgl1-mesa-dev libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Miniconda
# ---------------------------------------------------------------------------
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}
RUN curl -fsSL -o /tmp/miniconda.sh \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniconda.sh \
    && conda clean -afy

WORKDIR /workspace/infinova
COPY . .

# ---------------------------------------------------------------------------
# InstantSplatPP environment
# (mirrors InstantSplatPP/README.md Installation steps 1-5)
# ---------------------------------------------------------------------------
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN conda create -n instantsplatPP python=3.10.13 cmake=3.14.0 -y

RUN mkdir -p InstantSplatPP/mast3r/checkpoints/ \
    && wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
        -P InstantSplatPP/mast3r/checkpoints/

# Pin mkl alongside pytorch: conda's solver otherwise picks whatever MKL is
# newest at build time, which can be ABI-incompatible with PyTorch 2.1.2's
# bundled Intel ITT symbols (undefined symbol: iJIT_NotifyEvent at import).
RUN conda run -n instantsplatPP conda install pytorch==2.1.2 torchvision pytorch-cuda=12.1 "mkl<2024" -c pytorch -c nvidia -y \
    && conda run -n instantsplatPP python -m pip install -U pip

# Constrain torch/torchvision so pip's resolver can't upgrade away the
# conda-installed, CUDA-12.1-matched build while satisfying requirements.txt
# (e.g. huggingface-hub[torch] can otherwise pull in a newer torch with a
# bundled CUDA 13 toolkit, breaking the CUDA extension builds below).
RUN cd InstantSplatPP \
    && conda run -n instantsplatPP bash -c 'printf "torch==2.1.2\ntorchvision==0.16.2\n" > /tmp/torch-constraints.txt' \
    && conda run -n instantsplatPP pip install -r requirements.txt -c /tmp/torch-constraints.txt \
    && conda run -n instantsplatPP python -m pip uninstall -y numpy opencv-python \
    && conda run -n instantsplatPP python -m pip install --no-cache-dir "numpy<2" "opencv-python<4.12"

ARG TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV FORCE_CUDA=1

# torch.utils.cpp_extension (used below) imports pkg_resources, which
# setuptools>=81 no longer provides. Pin below that removal.
RUN conda run -n instantsplatPP python -m pip install --no-cache-dir "setuptools<81" wheel

RUN cd InstantSplatPP \
    && conda run -n instantsplatPP pip install -v --no-build-isolation ./submodules/simple-knn \
    && conda run -n instantsplatPP pip install -v --no-build-isolation ./submodules/diff-gaussian-rasterization \
    && conda run -n instantsplatPP pip install -v --no-build-isolation ./submodules/fused-ssim

# (Optional, skipped) InstantSplatPP/README.md step 5 compiles CroCo v2's
# RoPE CUDA kernels from croco/models/curope/, but that directory isn't
# present at the pinned submodule commit. Not required for the splatting
# pipeline in this repo.

# (Optional) MapAnything prior is not installed by default; see
# InstantSplatPP/README.md step 6 if you need PRIOR_MODEL_TYPE=mapanything.

# ---------------------------------------------------------------------------
# LeRobot environment
# ---------------------------------------------------------------------------
RUN conda create -n infinova-lerobot python=3.12 -y \
    && conda run -n infinova-lerobot pip install -e ./lerobot \
    && conda run -n infinova-lerobot pip install -e './lerobot[smolvla]'

CMD ["/bin/bash"]
