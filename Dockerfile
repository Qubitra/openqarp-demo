# The dashboard as a container, served on 7860 (the Hugging Face Spaces port).
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Spaces run the container as uid 1000; the app, its venv and marimo's state belong to it.
RUN useradd --create-home --uid 1000 app
USER app
ENV HOME=/home/app \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /home/app/demo

COPY --chown=app pyproject.toml uv.lock README.md LICENSE NOTICE ./
RUN uv sync --locked --no-dev --no-install-project
COPY --chown=app src ./src
RUN uv sync --locked --no-dev

EXPOSE 7860
CMD [".venv/bin/qubitra-openqarp-demo", "--host", "0.0.0.0", "--port", "7860", "--headless"]
