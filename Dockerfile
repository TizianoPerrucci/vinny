FROM python:3.12.11-bookworm

ARG DEBIAN_FRONTEND=noninteractive

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=2.1.4 \
    POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    VENV_PATH="/app/.venv" \
    JUPYTER_PORT=8888

EXPOSE $JUPYTER_PORT

# Set PATH to include Poetry and custom venv
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

# Install just
RUN curl -sSf https://just.systems/install.sh | bash -s -- --to /usr/bin

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python - --version $POETRY_VERSION

# Install OpenGL, Cuda compiler https://developer.nvidia.com/cuda-downloads
RUN --mount=type=cache,target=/var/cache/apt \
    wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libegl-dev cuda-compiler-12-9 cuda-libraries-dev-12-9

# See https://github.com/NVlabs/nvdiffrast/blob/main/docker/Dockerfile
# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES='all'
ENV NVIDIA_DRIVER_CAPABILITIES='compute,utility,graphics'

# Default pyopengl to EGL for good headless rendering support
ENV PYOPENGL_PLATFORM='egl'

COPY docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Create and prepare the virtual environment
RUN python -m venv $VENV_PATH && \
    python -m pip install --upgrade pip && \
    pip cache purge && \
    rm -rf /root/.cache/pip/http

WORKDIR /app

# Copy dependency files to the app directory
COPY poetry.lock pyproject.toml /app/

# Install dependencies with Poetry, caching downloaded packages
RUN --mount=type=cache,target=/root/.cache/pypoetry/cache \
    --mount=type=cache,target=/root/.cache/pypoetry/artifacts \
    poetry install --only main

# Copy the entire project code to the container
COPY . .

RUN just install-3d-libs

ENTRYPOINT [ "/bin/bash" ]
