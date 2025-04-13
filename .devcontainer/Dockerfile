FROM ubuntu:25.04

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

RUN sh -c "pipx ensurepath" && \
    bash -c "pipx ensurepath"

RUN pipx install uv \
    && pipx install ruff \
    && pipx install pre-commit

RUN echo '#!/bin/bash \n\
export VK_ICD_FILES=$(find /usr/share/vulkan/icd.d/ -name "lvp_icd*.json") \n\
export VK_ICD_FILENAMES=$VK_ICD_FILES \n\
export VK_DRIVER_FILES=$VK_ICD_FILES \n\
export LIBGL_ALWAYS_SOFTWARE=1 \n\
export GALLIUM_DRIVER=llvmpipe \n\
if ! DISPLAY=:99 xdpyinfo >/dev/null 2>&1; then  \n\
    rm -f /tmp/.X11-unix/X99 \n\
    rm -f /tmp/.X99-lock \n\
    Xvfb :99 -screen 0 960x540x24 & \n\
fi \n\
export DISPLAY=:99 \n\
export XDG_RUNTIME_DIR=/tmp/xdg-runtime-dir \n\
mkdir -p $XDG_RUNTIME_DIR && chmod 700 $XDG_RUNTIME_DIR \n\
' > /usr/local/bin/setup-vulkan-env.sh && chmod +x /usr/local/bin/setup-vulkan-env.sh

RUN echo '#!/bin/bash \n\
source /usr/local/bin/setup-vulkan-env.sh \n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

RUN echo '. /usr/local/bin/setup-vulkan-env.sh' >> /etc/bash.bashrc

RUN echo '#!/bin/bash \n\
vulkaninfo --summary \n\
vkcube --width 256 --height 256 & \n\
for attempt in $(seq 1 64); do \n\
    import -window root /setup-vulkan-env.png \n\
    mean=$(identify -format "%[fx:mean]" /setup-vulkan-env.png) \n\
    if (( $(echo "$mean > 0.01" | bc -l) )); then \n\
        break \n\
    fi \n\
    sleep 0.1 \n\
done \n\
' | ./entrypoint.sh bash

COPY vkrunner /vkrunner

WORKDIR /vkrunner

RUN cargo build --release && \
    cp /vkrunner/target/release/vkrunner /usr/local/bin/ && \
    chmod +x /usr/local/bin/vkrunner

WORKDIR /

ENTRYPOINT ["/entrypoint.sh"]

CMD ["bash"]
