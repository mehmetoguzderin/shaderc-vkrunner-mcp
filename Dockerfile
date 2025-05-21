FROM ubuntu:25.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    bc \
    build-essential \
    clang \
    clang-format \
    clang-tidy \
    clang-tools \
    clangd \
    cmake \
    curl \
    ffmpeg \
    git \
    glslang-tools \
    glslc \
    jq \
    libshaderc-dev \
    libshaderc1 \
    imagemagick \
    libgl1-mesa-dri \
    libvulkan-dev \
    libvulkan1 \
    mesa-utils \
    mesa-vulkan-drivers \
    ninja-build \
    npm \
    pipx \
    python3 \
    python3-pip \
    python3-venv \
    rustup \
    sudo \
    unzip \
    vulkan-tools \
    wget \
    x11-utils \
    xvfb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN rustup install 1.86.0 && \
    rustup default 1.86.0

RUN rustup component add rustfmt && \
    rustup component add clippy && \
    rustup component add rust-src && \
    rustup component add rust-analyzer

WORKDIR /app
COPY . .

RUN cargo build --release

WORKDIR /vkrunner_build
RUN git clone https://gitlab.freedesktop.org/mesa/vkrunner.git . && \
    cargo build --release


FROM ubuntu:25.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    bc \
    glslang-tools \
    glslc \
    imagemagick \
    jq \
    libgl1-mesa-dri \
    libshaderc1 \
    libvulkan1 \
    mesa-utils \
    mesa-vulkan-drivers \
    vulkan-tools \
    x11-utils \
    xvfb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN echo '#!/bin/bash \n\
export VK_ICD_FILES=$(find /usr/share/vulkan/icd.d/ -name "lvp_icd*.json") \n\
export VK_ICD_FILENAMES=$VK_ICD_FILES \n\
export VK_DRIVER_FILES=$VK_ICD_FILES \n\
export LIBGL_ALWAYS_SOFTWARE=1 \n\
export GALLIUM_DRIVER=llvmpipe \n\
if ! ps aux | grep -v grep | grep "Xvfb :99" > /dev/null; then \n\
    rm -f /tmp/.X11-unix/X99 \n\
    rm -f /tmp/.X99-lock \n\
    Xvfb :99 -screen 0 960x540x24 > /dev/null 2>&1 & \n\
fi \n\
export DISPLAY=:99 \n\
export XDG_RUNTIME_DIR=/tmp/xdg-runtime-dir \n\
mkdir -p $XDG_RUNTIME_DIR && chmod 700 $XDG_RUNTIME_DIR \n\
' > /usr/local/bin/setup-vulkan-env.sh && chmod +x /usr/local/bin/setup-vulkan-env.sh

RUN echo '#!/bin/bash \n\
source /usr/local/bin/setup-vulkan-env.sh \n\
if [[ "${1}" == --* ]]; then \n\
    /usr/local/bin/shaderc-vkrunner-mcp "$@" \n\
else \n\
    exec "$@" \n\
fi \n\
' > /entrypoint.sh && chmod +x /entrypoint.sh



COPY --from=builder /app/target/release/shaderc-vkrunner-mcp /usr/local/bin/
COPY --from=builder /vkrunner_build/target/release/vkrunner /usr/local/bin/

WORKDIR /work

ENTRYPOINT ["/entrypoint.sh"]
