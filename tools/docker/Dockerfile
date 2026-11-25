# Сборщик postal2rust / p2gles: кросс-сборка под aarch64 (Debian trixie, приставка) и сборка под x86_64.
FROM debian:trixie
ENV DEBIAN_FRONTEND=noninteractive
RUN dpkg --add-architecture arm64 && apt-get update -qq && apt-get install -y -qq --no-install-recommends \
      ca-certificates curl gcc g++ libc6-dev pkg-config git \
      gcc-aarch64-linux-gnu libc6-dev-arm64-cross \
      libasound2-dev libudev-dev libxkbcommon-dev libwayland-dev \
      libasound2-dev:arm64 libudev-dev:arm64 libxkbcommon-dev:arm64 libwayland-dev:arm64 \
    && rm -rf /var/lib/apt/lists/*
ENV CARGO_HOME=/opt/cargo RUSTUP_HOME=/opt/rustup PATH=/opt/cargo/bin:$PATH
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal --no-modify-path \
    && rustup target add aarch64-unknown-linux-gnu
ENV CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
    CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc
RUN chmod -R a+rwX /opt/cargo /opt/rustup
