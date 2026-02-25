from __future__ import annotations

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=1, max_length=120)


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "Bearer"
    expires_in: int
    username: str


class UserResponse(BaseModel):
    username: str


class SearchRequest(BaseModel):
    keywords: str = Field(min_length=1, max_length=200)
    location: str = Field(default="United States", max_length=200)
    sources: list[str] = Field(default_factory=list)
    limit_per_source: int = Field(default=30, ge=5, le=120)
    linkedin_detail_delay: float = Field(default=0.2, ge=0.0, le=1.5)
    linkedin_skip_apply_url: bool = True
    web_result_pages: int = Field(default=1, ge=1, le=3)
    web_max_sites: int = Field(default=8, ge=4, le=36)
    web_follow_links_per_site: int = Field(default=0, ge=0, le=4)
    web_links_per_site: int = Field(default=4, ge=2, le=20)


class JobResponse(BaseModel):
    source: str
    keywords: str
    title: str
    company: str
    location: str
    listed_at: str
    listed_date: str
    job_id: str
    job_url: str
    apply_url: str
    employment_type: str
    salary: str
    is_remote: bool
    description_snippet: str


class SourceReportResponse(BaseModel):
    source: str
    status: str
    jobs_fetched: int
    elapsed_seconds: float
    message: str


class SearchResponse(BaseModel):
    jobs: list[JobResponse]
    reports: list[SourceReportResponse]
    events: list[str]
