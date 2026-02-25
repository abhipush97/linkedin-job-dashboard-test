from __future__ import annotations

import os
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from job_sources import ALL_SOURCES, DEFAULT_UI_SOURCES, SearchConfig, clear_search_cache, run_multi_source_search

from .auth import create_session, destroy_session, validate_credentials, validate_session
from .schemas import (
    LoginRequest,
    LoginResponse,
    SearchRequest,
    SearchResponse,
    SourceReportResponse,
    UserResponse,
)

app = FastAPI(
    title="Jobs Metasearch API",
    version="1.0.0",
    description="Modern API for multi-source job search.",
)

origins_raw = os.getenv("APP_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
allowed_origins = [origin.strip() for origin in origins_raw.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _extract_bearer_token(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization header")

    parts = authorization.strip().split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Authorization header")
    return parts[1].strip()


def require_user(
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> str:
    token = _extract_bearer_token(authorization)
    session = validate_session(token)
    if not session:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired session")
    return session.username


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/auth/login", response_model=LoginResponse)
def login(payload: LoginRequest) -> LoginResponse:
    if not validate_credentials(payload.username, payload.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")

    token, ttl_seconds = create_session(payload.username)
    return LoginResponse(
        access_token=token,
        expires_in=ttl_seconds,
        username=payload.username.strip(),
    )


@app.post("/api/auth/logout")
def logout(
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> dict[str, str]:
    token = _extract_bearer_token(authorization)
    destroy_session(token)
    return {"status": "signed_out"}


@app.get("/api/auth/me", response_model=UserResponse)
def me(current_user: Annotated[str, Depends(require_user)]) -> UserResponse:
    return UserResponse(username=current_user)


@app.get("/api/sources")
def list_sources(current_user: Annotated[str, Depends(require_user)]) -> dict[str, object]:
    return {
        "sources": ALL_SOURCES,
        "default_sources": DEFAULT_UI_SOURCES,
        "user": current_user,
    }


@app.post("/api/search", response_model=SearchResponse)
def search_jobs(
    payload: SearchRequest,
    current_user: Annotated[str, Depends(require_user)],
) -> SearchResponse:
    selected_sources = payload.sources or DEFAULT_UI_SOURCES
    config = SearchConfig(
        keywords=payload.keywords.strip(),
        location=payload.location.strip(),
        sources=selected_sources,
        limit_per_source=payload.limit_per_source,
        linkedin_detail_delay=payload.linkedin_detail_delay,
        linkedin_skip_apply_url=payload.linkedin_skip_apply_url,
        web_result_pages=payload.web_result_pages,
        web_max_sites=payload.web_max_sites,
        web_follow_links_per_site=payload.web_follow_links_per_site,
        web_links_per_site=payload.web_links_per_site,
    )

    events: list[str] = [f"Search requested by {current_user}"]

    def on_progress(step: int, total: int, message: str) -> None:
        events.append(f"[{step}/{total}] {message}")

    jobs, reports = run_multi_source_search(config, progress_callback=on_progress)
    return SearchResponse(
        jobs=[job.to_dict() for job in jobs],
        reports=[SourceReportResponse(**report.to_dict()) for report in reports],
        events=events,
    )


@app.post("/api/cache/clear")
def clear_cache(current_user: Annotated[str, Depends(require_user)]) -> dict[str, str]:
    clear_search_cache()
    return {"status": "cache_cleared", "user": current_user}
