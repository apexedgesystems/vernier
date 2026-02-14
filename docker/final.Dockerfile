# ==============================================================================
# final.Dockerfile - Artifact packaging and extraction stage
#
# Collects install trees from all builder images into a single lightweight
# image for easy extraction. Based on busybox for minimal size.
#
# Library artifacts (per-platform):
#   /output/vernier-{VER}-x86_64-linux.tar.gz          - x86_64 Linux (no CUDA)
#   /output/vernier-{VER}-x86_64-linux-cuda.tar.gz     - x86_64 Linux + CUDA
#   /output/vernier-{VER}-aarch64-jetson.tar.gz        - Jetson (aarch64 + CUDA)
#   /output/vernier-{VER}-aarch64-rpi.tar.gz           - Raspberry Pi (aarch64)
#   /output/vernier-{VER}-riscv64-linux.tar.gz         - RISC-V 64-bit
#
# Tool artifacts (native x86_64 only):
#   /output/vernier-tools-{VER}-x86_64-linux.tar.gz    - Rust CLI (bench binary)
#   /output/vernier_py_tools-{VER}-py3-none-any.whl    - Python CLI (pip install)
#
# Library tarballs contain:
#   lib/              - Shared libraries (.so)
#   include/          - Public headers
#   lib/cmake/vernier - CMake find_package() config
#   share/doc/        - Documentation
#   .env              - Environment setup (LD_LIBRARY_PATH)
#
# Usage:
#   docker compose build final
#   make artifacts
#
# Extract artifacts:
#   docker create --name tmp vernier.final
#   docker cp tmp:/output/vernier-1.0.0-x86_64-linux.tar.gz .
#   docker rm tmp
# ==============================================================================
FROM busybox:latest

ARG USER
ARG VERSION=1.0.0

LABEL org.opencontainers.image.title="vernier.final" \
      org.opencontainers.image.description="Packaged build artifacts for distribution"

WORKDIR /output

# ==============================================================================
# Collect Library Install Trees from Builders
# ==============================================================================
COPY --from=vernier.builder.cpu:latest      /home/${USER}/workspace/build/native-linux-release/install/      ./cpu/
COPY --from=vernier.builder.cuda:latest     /home/${USER}/workspace/build/native-linux-release/install/      ./cuda/
COPY --from=vernier.builder.jetson:latest   /home/${USER}/workspace/build/jetson-aarch64-release/install/    ./jetson/
COPY --from=vernier.builder.rpi:latest      /home/${USER}/workspace/build/rpi-aarch64-release/install/       ./rpi/
COPY --from=vernier.builder.riscv64:latest  /home/${USER}/workspace/build/riscv64-linux-release/install/     ./riscv64/

# ==============================================================================
# Collect Tool Artifacts (native builders only)
# ==============================================================================
COPY --from=vernier.builder.cuda:latest     /home/${USER}/workspace/build/native-linux-release/install-tools/rust/  ./tools-rust/
COPY --from=vernier.builder.cuda:latest     /home/${USER}/workspace/build/native-linux-release/install-tools/py/    ./tools-py-staging/

# ==============================================================================
# Create Distribution Packages
# ==============================================================================
RUN tar -czf vernier-${VERSION}-x86_64-linux.tar.gz       -C ./cpu . && \
    tar -czf vernier-${VERSION}-x86_64-linux-cuda.tar.gz  -C ./cuda . && \
    tar -czf vernier-${VERSION}-aarch64-jetson.tar.gz     -C ./jetson . && \
    tar -czf vernier-${VERSION}-aarch64-rpi.tar.gz        -C ./rpi . && \
    tar -czf vernier-${VERSION}-riscv64-linux.tar.gz      -C ./riscv64 . && \
    tar -czf vernier-tools-${VERSION}-x86_64-linux.tar.gz -C ./tools-rust . && \
    cp ./tools-py-staging/*.whl .

# ==============================================================================
# Default: List Available Artifacts
# ==============================================================================
CMD ["ls", "-la", "/output"]
