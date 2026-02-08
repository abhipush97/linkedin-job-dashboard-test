#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, unquote, urlparse

import requests
from bs4 import BeautifulSoup


SEARCH_ENDPOINT = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
JOB_POSTING_ENDPOINT = "https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass(frozen=True)
class JobListing:
    keywords: str
    title: str
    company: str
    location: str
    listed_at: str
    job_id: str
    job_url: str
    apply_url: str


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def canonicalize_job_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return url
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"


def extract_job_id(entity_urn: str) -> str:
    # entity_urn format: urn:li:jobPosting:1234567890
    parts = entity_urn.split(":")
    return parts[-1] if parts else ""


def parse_apply_url(raw_apply_url: str) -> str:
    parsed = urlparse(raw_apply_url)
    query = parse_qs(parsed.query)
    for key in ("url", "session_redirect"):
        candidate = query.get(key, [])
        if candidate:
            return unquote(candidate[0])
    return raw_apply_url


def is_external_url(url: str) -> bool:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return False
    host = parsed.netloc.lower()
    return "linkedin.com" not in host


class LinkedInPublicJobsClient:
    def __init__(self, *, timeout_seconds: int = 20, max_retries: int = 3) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)

    def _get(self, url: str, params: dict[str, object] | None = None) -> str:
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout_seconds)
                if response.status_code in {429, 999}:
                    if attempt == self.max_retries:
                        response.raise_for_status()
                    backoff = min(20.0, 2.0 * attempt) + random.uniform(0.1, 0.8)
                    time.sleep(backoff)
                    continue
                response.raise_for_status()
                return response.text
            except requests.RequestException:
                if attempt == self.max_retries:
                    raise
                backoff = min(20.0, 1.5 * attempt) + random.uniform(0.1, 0.8)
                time.sleep(backoff)

        raise RuntimeError("Request failed after retries")

    def search_jobs(
        self,
        *,
        keywords: str,
        location: str,
        limit: int,
        include_apply_url: bool,
        detail_delay_seconds: float,
    ) -> list[JobListing]:
        collected: list[JobListing] = []
        seen_job_ids: set[str] = set()
        start = 0

        while len(collected) < limit:
            page_html = self._get(
                SEARCH_ENDPOINT,
                params={
                    "keywords": keywords,
                    "location": location,
                    "start": start,
                },
            )
            soup = BeautifulSoup(page_html, "html.parser")
            cards = soup.select("div.base-card[data-entity-urn]")
            if not cards:
                break

            for card in cards:
                job_id = extract_job_id(card.get("data-entity-urn", ""))
                if not job_id or job_id in seen_job_ids:
                    continue
                seen_job_ids.add(job_id)

                title_tag = card.select_one("h3.base-search-card__title")
                company_tag = card.select_one("h4.base-search-card__subtitle")
                location_tag = card.select_one("span.job-search-card__location")
                time_tag = card.select_one("time")

                title = clean_text(title_tag.get_text(" ", strip=True) if title_tag else "")
                company = clean_text(company_tag.get_text(" ", strip=True) if company_tag else "")
                job_location = clean_text(location_tag.get_text(" ", strip=True) if location_tag else "")

                listed_at = ""
                if time_tag:
                    listed_at = clean_text(time_tag.get("datetime") or time_tag.get_text(" ", strip=True))

                link_tag = card.select_one("a.base-card__full-link")
                raw_job_url = html.unescape(link_tag.get("href", "")) if link_tag else ""
                job_url = canonicalize_job_url(raw_job_url)
                apply_url = job_url

                if include_apply_url:
                    apply_url = self.fetch_apply_url(job_id=job_id, job_url=job_url, fallback_url=job_url)
                    time.sleep(max(0.0, detail_delay_seconds) + random.uniform(0.05, 0.25))

                collected.append(
                    JobListing(
                        keywords=keywords,
                        title=title,
                        company=company,
                        location=job_location,
                        listed_at=listed_at,
                        job_id=job_id,
                        job_url=job_url,
                        apply_url=apply_url,
                    )
                )
                if len(collected) >= limit:
                    break

            start += len(cards)
            time.sleep(0.25 + random.uniform(0.05, 0.2))

        return collected

    def _extract_apply_url_candidates(self, html_text: str) -> list[str]:
        candidates: list[str] = []
        soup = BeautifulSoup(html_text, "html.parser")

        code_tag = soup.select_one("code#applyUrl")
        if code_tag:
            raw = code_tag.decode_contents().strip()
            raw = raw.removeprefix("<!--").removesuffix("-->").strip()
            raw = raw.strip('"')
            raw = html.unescape(raw)
            if raw:
                candidates.append(raw)

        for match in re.finditer(
            r"https://www\.linkedin\.com/jobs/view/externalApply/\d+\?url=[^\"'<> ]+",
            html_text,
        ):
            candidates.append(html.unescape(match.group(0)))

        selectors = [
            "a.apply-button[href]",
            "a[data-tracking-control-name*='apply-link'][href]",
            "a.top-card-layout__cta--primary[href]",
        ]
        for selector in selectors:
            for anchor in soup.select(selector):
                href = clean_text(anchor.get("href"))
                if href:
                    candidates.append(html.unescape(href))

        deduped: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            if item and item not in seen:
                seen.add(item)
                deduped.append(item)
        return deduped

    def fetch_apply_url(self, *, job_id: str, job_url: str, fallback_url: str) -> str:
        for target in (JOB_POSTING_ENDPOINT.format(job_id=job_id), job_url):
            if not target:
                continue
            try:
                detail_html = self._get(target)
            except requests.RequestException:
                continue

            candidates = self._extract_apply_url_candidates(detail_html)
            if not candidates:
                continue

            normalized: list[str] = []
            for candidate in candidates:
                resolved = parse_apply_url(candidate)
                if resolved.startswith("/"):
                    resolved = f"https://www.linkedin.com{resolved}"
                if resolved and resolved not in normalized:
                    normalized.append(resolved)

            for url in normalized:
                if is_external_url(url):
                    return url
            for url in normalized:
                if url:
                    return url

        return fallback_url


def prompt_for_missing_args(keywords: str | None, location: str | None) -> tuple[str, str]:
    entered_keywords = (keywords or "").strip()
    entered_location = (location or "").strip()

    if not entered_keywords:
        entered_keywords = input("Keywords to search jobs for: ").strip()
    if not entered_location:
        entered_location = input("Location [United States]: ").strip() or "United States"

    return entered_keywords, entered_location


def write_csv(path: Path, jobs: Iterable[JobListing]) -> int:
    rows = [asdict(job) for job in jobs]
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "keywords",
                    "title",
                    "company",
                    "location",
                    "listed_at",
                    "job_id",
                    "job_url",
                    "apply_url",
                ]
            )
        return 0

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch LinkedIn public job search results and export them to CSV."
    )
    parser.add_argument("--keywords", help="Keyword search query, e.g. 'data scientist'.")
    parser.add_argument("--location", help="Location filter, e.g. 'United States'.")
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Maximum number of jobs to fetch (default: 25).",
    )
    parser.add_argument(
        "--output",
        default="linkedin_jobs.csv",
        help="CSV output path (default: linkedin_jobs.csv).",
    )
    parser.add_argument(
        "--detail-delay",
        type=float,
        default=0.35,
        help="Base delay in seconds between detail page requests (default: 0.35).",
    )
    parser.add_argument(
        "--skip-apply-url",
        action="store_true",
        help="Skip job detail calls and use the LinkedIn job page as apply_url.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    keywords, location = prompt_for_missing_args(args.keywords, args.location)

    if not keywords:
        print("Error: keywords are required.", file=sys.stderr)
        return 1
    if args.limit <= 0:
        print("Error: --limit must be greater than 0.", file=sys.stderr)
        return 1

    client = LinkedInPublicJobsClient()

    print(f"Searching LinkedIn jobs for '{keywords}' in '{location}'...")
    jobs = client.search_jobs(
        keywords=keywords,
        location=location,
        limit=args.limit,
        include_apply_url=not args.skip_apply_url,
        detail_delay_seconds=args.detail_delay,
    )

    output_path = Path(args.output).expanduser().resolve()
    count = write_csv(output_path, jobs)

    print(f"Wrote {count} job(s) to: {output_path}")
    if count == 0:
        print("No jobs were returned. Try broader keywords/location or run again later.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
