FROM python:3.11-slim

WORKDIR /app

COPY ./schedulers /app/schedulers/.

COPY ../svaeva_redux /app/svaeva_redux

WORKDIR /app/schedulers

RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --only main