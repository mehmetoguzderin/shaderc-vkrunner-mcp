#!/bin/bash

set -eu

src_dir="$(cd $(dirname "$0") && pwd)"
build_dir="$src_dir/tmp-build"
install_dir="$build_dir/install"
device_id=""

if [ $# -gt 0 ] && [ "$1" = "--device-id" ]; then
    if [ -z "${2:-}" ]; then
        echo "--device-id must be followed by a number"
        exit 1
    fi
    device_id="--device-id $2"
fi

rm -fr -- "$build_dir"

cargo test --target-dir "$build_dir"
cargo install --target-dir "$build_dir" --path . --root "$install_dir"

# Run the built executable with all of the examples and enable the
# validation layer. Verify that nothing was written to the output.
VKRUNNER_ALWAYS_FLUSH_MEMORY=true \
VK_LOADER_LAYERS_ENABLE="*validation" \
                  "$install_dir/bin/vkrunner" \
                  -q \
                  "$src_dir/examples"/*.shader_test \
                  2>&1 \
    | tee "$build_dir/output.txt"

if grep -q --invert-match '^/tmp' "$build_dir/output.txt"; then
    echo "FAIL VkRunner had output with quiet flag"
    exit 1;
fi

# Try again with precompiled scripts
"$src_dir"/precompile-script.py -o "$build_dir/precompiled-examples" \
          "$src_dir/examples"/*.shader_test
"$install_dir/bin/vkrunner" $device_id \
    "$build_dir/precompiled-examples/"*.shader_test

echo
echo "Test build succeeded."
