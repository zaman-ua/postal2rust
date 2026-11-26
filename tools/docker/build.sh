#!/bin/bash
# Builds in the p2-builder Docker image (Debian trixie, Rust stable, aarch64 cross toolchain).
#   tools/docker/build.sh [arm|x86] [package]       default: arm p2gles
# Output: target-docker/aarch64-unknown-linux-gnu/release/<package> (arm) or
# target-docker/release/<package> (x86).
# ARM builds of p2game get the `arm-board` feature (see crates/p2game/Cargo.toml); P2_FEATURES
# overrides the feature list for either target (P2_FEATURES= for none).
# P2_PROFILE=dist builds with fat LTO (slow; the paths above then say dist instead of release).
# The image is built from tools/docker/Dockerfile on first use; P2_BUILDER_IMAGE names another one.
set -e
ARCH=${1:-arm}; PKG=${2:-p2gles}; PROFILE=${P2_PROFILE:-release}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
IMG=${P2_BUILDER_IMAGE:-p2-builder:latest}
docker image inspect "$IMG" >/dev/null 2>&1 || docker build -t "$IMG" "$ROOT/tools/docker"
case $ARCH in
  arm) TGT="--target aarch64-unknown-linux-gnu"; ENVS="-e PKG_CONFIG_ALLOW_CROSS=1 -e PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/pkgconfig -e PKG_CONFIG_SYSROOT_DIR=/ -e RUSTFLAGS=-Ctarget-cpu=cortex-a55" ;;
  x86) TGT=""; ENVS="" ;;
  *) echo "first argument: arm or x86"; exit 1 ;;
esac
DEFAULT_FEATURES=""
[ "$ARCH" = arm ] && [ "$PKG" = p2game ] && DEFAULT_FEATURES=p2game/arm-board
FEATURES=${P2_FEATURES-$DEFAULT_FEATURES}
# --offline when the image already carries the crate registry; otherwise cargo downloads.
OFF=$(docker run --rm "$IMG" sh -c 'ls /opt/cargo/registry/cache 2>/dev/null | grep -q . && echo --offline' || true)
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -e CARGO_HOME=/opt/cargo $ENVS -e CARGO_TARGET_DIR=/w/target-docker \
  -v "$ROOT":/w -w /w "$IMG" cargo build $OFF --profile "$PROFILE" -p "$PKG" ${FEATURES:+--features "$FEATURES"} $TGT
