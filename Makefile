.PHONY: run test lint format build up down logs

# ── Local dev (no Docker) ────────────────────────────────────────────────────

run:
	cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8080

test:
	cd backend && pytest tests/ -v

lint:
	cd backend && ruff check app/ tests/

format:
	cd backend && ruff format app/ tests/

typecheck:
	cd backend && mypy app/

# ── Docker Compose ───────────────────────────────────────────────────────────

build:
	docker compose build

up:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f api
