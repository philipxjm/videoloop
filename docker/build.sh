#!/bin/bash
# Build script for video-understanding-agent Docker images
#
# Usage:
#   ./build.sh          - Build only the tools layer (fast, ~5 seconds)
#   ./build.sh base     - Build the base layer (slow, ~10-15 minutes)
#   ./build.sh all      - Build both base and tools layers
#   ./build.sh rebuild  - Force rebuild base layer (no cache)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BASE_IMAGE="video-understanding-base:latest"
TOOLS_IMAGE="video-understanding-sandbox:latest"

build_base() {
    echo "=================================================="
    echo "Building BASE layer (this takes ~10-15 minutes)..."
    echo "=================================================="
    local cache_flag=""
    if [ "$1" == "--no-cache" ]; then
        cache_flag="--no-cache"
    fi
    docker build $cache_flag -t "$BASE_IMAGE" -f Dockerfile.base --progress=plain .
    echo ""
    echo "✅ Base layer built: $BASE_IMAGE"
}

build_tools() {
    echo "=================================================="
    echo "Building TOOLS layer (this takes ~5 seconds)..."
    echo "=================================================="
    
    # Check if base image exists
    if ! docker image inspect "$BASE_IMAGE" &>/dev/null; then
        echo "❌ Base image not found: $BASE_IMAGE"
        echo "   Run './build.sh base' first to build the base layer."
        exit 1
    fi
    
    docker build --build-arg BASE_IMAGE="$BASE_IMAGE" -t "$TOOLS_IMAGE" -f Dockerfile.tools --progress=plain .
    echo ""
    echo "✅ Tools layer built: $TOOLS_IMAGE"
}

show_info() {
    echo ""
    echo "=================================================="
    echo "Build complete!"
    echo "=================================================="
    echo ""
    echo "Images:"
    docker images | grep -E "video-understanding|REPOSITORY" | head -5
    echo ""
    echo "To test the container:"
    echo "  docker run --rm -it --gpus all $TOOLS_IMAGE nvidia-smi"
    echo ""
    echo "To run with video mounting:"
    echo "  docker run --rm -it --gpus all -v /path/to/videos:/testbed/videos $TOOLS_IMAGE"
}

case "${1:-tools}" in
    base)
        build_base
        ;;
    all)
        build_base
        build_tools
        show_info
        ;;
    rebuild)
        build_base --no-cache
        build_tools
        show_info
        ;;
    tools|"")
        build_tools
        show_info
        ;;
    *)
        echo "Usage: ./build.sh [base|tools|all|rebuild]"
        echo ""
        echo "  base    - Build only the base layer (slow, ~10-15 minutes)"
        echo "  tools   - Build only the tools layer (fast, ~5 seconds) [default]"
        echo "  all     - Build both base and tools layers"
        echo "  rebuild - Force rebuild base with no cache, then tools"
        exit 1
        ;;
esac
