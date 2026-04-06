FROM python:3.11-slim-bookworm AS builder

WORKDIR /app

ENV POETRY_HOME="/root/.poetry" \
    VIRTUAL_ENV=/venv \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/venv/bin:/root/.poetry/bin:$PATH"

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl build-essential ca-certificates \
 && update-ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 -
RUN python -m venv /venv && /venv/bin/pip install --upgrade pip

COPY ./pyproject.toml ./pyproject.toml
COPY ./poetry.lock ./poetry.lock

RUN poetry lock
RUN poetry config virtualenvs.create false \
 && poetry install --only main --no-root --no-interaction --no-ansi

COPY ./common ./common/
COPY ./generated ./generated/
COPY ./services ./services/
COPY ./main.py ./main.py
COPY ./healthcheck.py ./healthcheck.py
COPY ./settings.py ./settings.py

FROM python:3.11-slim-bookworm AS runtime

WORKDIR /app

ENV VIRTUAL_ENV=/venv \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/appuser/.cache/huggingface \
    PATH="/venv/bin:$PATH"

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && update-ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /venv /venv
COPY --from=builder /app /app

CMD ["python", "main.py"]