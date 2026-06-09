"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from .config import ServiceConfig
from .pipeline import Pipeline
from .schemas import CompleteResponse, HealthResponse

_STATIC_DIR = Path(__file__).parent / "static"
_RESERVED_LOOKUP_PARAMS = {"n", "limit"}


def create_app(pipeline: Pipeline | None = None, config: ServiceConfig | None = None) -> FastAPI:
    """Build the app from an existing pipeline (tests) or a config (deferred load)."""
    if pipeline is None and config is None:
        raise ValueError("either pipeline or config is required")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = pipeline if pipeline is not None else Pipeline.from_config(config)
        yield

    app = FastAPI(title="Deep-Spell-9", version="2", lifespan=lifespan)

    @app.get("/healthz", response_model=HealthResponse)
    def healthz(request: Request) -> HealthResponse:
        p: Pipeline = request.app.state.pipeline
        return HealthResponse(with_corrector=p.corrector is not None, with_lookup=p.lookup_db is not None)

    @app.get("/api/complete", response_model=CompleteResponse)
    def complete(request: Request, q: str = "") -> CompleteResponse:
        p: Pipeline = request.app.state.pipeline
        return p.complete(q)

    @app.get("/api/lookup")
    def lookup(request: Request, n: int = 10) -> list[dict]:
        p: Pipeline = request.app.state.pipeline
        criteria = {
            key: value
            for key, value in request.query_params.items()
            if key not in _RESERVED_LOOKUP_PARAMS
        }
        try:
            return p.lookup(criteria, limit=min(max(n, 1), 100))
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    return app
