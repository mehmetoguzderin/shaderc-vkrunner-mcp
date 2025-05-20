#!/usr/bin/env bash
#
# Copyright 2025 Intel Corporation
# SPDX-License-Identifier: MIT

set -e

if [ ! -f "Cargo.toml" ]; then
    echo "Run the script from the root of the repository."
    exit 1
fi

SKIP_DOWNLOAD=

for arg in "$@"; do
    case $arg in
        --skip-download)
            SKIP_DOWNLOAD=1
            shift
            ;;
        *)
            shift
            ;;
    esac
done

if [ -z "$SKIP_DOWNLOAD" ]; then
    VULKAN_HEADERS=$(mktemp -d)
    git clone https://github.com/KhronosGroup/Vulkan-Headers.git "$VULKAN_HEADERS"
    git -C "$VULKAN_HEADERS" log -1

    cp -f $VULKAN_HEADERS/include/vk_video/*.h include/vk_video/
    cp -f $VULKAN_HEADERS/include/vulkan/{vulkan.h,vulkan_core.h,vk_platform.h} include/vulkan/
fi

# TODO: Most of these scripts should be using the registry/vk.xml instead of
# parsing the C headers.

echo | gcc -include "./include/vulkan/vulkan.h" -E - | vkrunner/make-enums.py > vkrunner/enum_table.rs
vkrunner/make-features.py < include/vulkan/vulkan_core.h > vkrunner/features.rs
vkrunner/make-formats.py < include/vulkan/vulkan_core.h > vkrunner/format_table.rs
vkrunner/make-pipeline-key-data.py > vkrunner/pipeline_key_data.rs
vkrunner/make-vulkan-funcs-data.py > vkrunner/vulkan_funcs_data.rs
