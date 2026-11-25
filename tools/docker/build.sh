#!/bin/bash
# Сборка в Docker-образе p2-builder (Debian trixie, Rust stable, кросс-компилятор aarch64).
#   tools/docker/build.sh [arm|x86] [пакет]       по умолчанию: arm p2gles
# Результат: target-docker/aarch64-unknown-linux-gnu/release/<пакет> (arm) или target-docker/release/<пакет> (x86).
# Образ: собирается из tools/docker/Dockerfile при первом запуске, или загружается заранее: docker load -i p2-builder.tar
set -e
ARCH=${1:-arm}; PKG=${2:-p2gles}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
IMG=${P2_BUILDER_IMAGE:-p2-builder:latest}
docker image inspect "$IMG" >/dev/null 2>&1 || docker build -t "$IMG" "$ROOT/tools/docker"
case $ARCH in
  arm) TGT="--target aarch64-unknown-linux-gnu"; ENVS="-e PKG_CONFIG_ALLOW_CROSS=1 -e PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/pkgconfig -e PKG_CONFIG_SYSROOT_DIR=/ -e RUSTFLAGS=-Ctarget-cpu=cortex-a55" ;;
  x86) TGT=""; ENVS="" ;;
  *) echo "arm или x86"; exit 1 ;;
esac
# --offline, если зависимости уже есть в образе (p2-builder с флешки); иначе скачает.
OFF=$(docker run --rm "$IMG" sh -c 'ls /opt/cargo/registry/cache 2>/dev/null | grep -q . && echo --offline' || true)
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -e CARGO_HOME=/opt/cargo $ENVS -e CARGO_TARGET_DIR=/w/target-docker \
  -v "$ROOT":/w -w /w "$IMG" cargo build $OFF --release -p "$PKG" $TGT
