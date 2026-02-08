from __future__ import annotations

import json
import hashlib
import html
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from time import perf_counter
from threading import Lock
from typing import Callable
from urllib.parse import parse_qs, unquote, urlencode, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from linkedin_jobs_to_csv import LinkedInPublicJobsClient


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

LINKEDIN = "LinkedIn"
INDEED = "Indeed"
NAUKRI = "Naukri"
REMOTEOK = "RemoteOK"
REMOTIVE = "Remotive"
ARBEITNOW = "Arbeitnow"
WEB_COMPANY_CAREERS = "Web Search (Company Careers)"
DUCKDUCKGO_HTML_ENDPOINT = "https://duckduckgo.com/html/"
YAHOO_SEARCH_ENDPOINT = "https://search.yahoo.com/search"

ALL_SOURCES = [LINKEDIN, INDEED, NAUKRI, REMOTEOK, REMOTIVE, ARBEITNOW, WEB_COMPANY_CAREERS]
DEFAULT_UI_SOURCES = [LINKEDIN, REMOTEOK, REMOTIVE, ARBEITNOW]

REMOTE_HINTS = (
    "remote",
    "work from home",
    "wfh",
    "distributed",
    "anywhere",
    "worldwide",
    "telecommute",
)
CAREER_LINK_HINTS = (
    "career",
    "careers",
    "jobs",
    "join-us",
    "joinus",
    "vacancies",
    "openings",
    "work-with-us",
    "opportunities",
)
JOB_LINK_HINTS = (
    "job",
    "jobs",
    "opening",
    "openings",
    "position",
    "vacancy",
    "opportunity",
    "hiring",
    "requisition",
    "reqid",
    "jobid",
    "department",
)
IGNORED_WEB_HOST_MARKERS = (
    "search.yahoo.com",
    "r.search.yahoo.com",
    "duckduckgo.com",
    "linkedin.com",
    "indeed.com",
    "naukri.com",
    "glassdoor.com",
    "ziprecruiter.com",
    "monster.com",
    "simplyhired.com",
    "youtube.com",
    "facebook.com",
    "instagram.com",
    "x.com",
    "twitter.com",
    "wikipedia.org",
)
SEARCH_CACHE_TTL_SECONDS = 300.0
SEARCH_CACHE_MAX_ENTRIES = 20
SEARCH_WORKERS = 6
CAREER_CRAWL_WORKERS = 8
_SEARCH_CACHE: dict[
    tuple[object, ...],
    tuple[float, list[UnifiedJob], list[SourceRunReport]],
] = {}
_SEARCH_CACHE_LOCK = Lock()

ProgressCallback = Callable[[int, int, str], None]


class SourceBlockedError(RuntimeError):
    """Raised when a source explicitly blocks automation."""


@dataclass(frozen=True)
class UnifiedJob:
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

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SourceRunReport:
    source: str
    status: str
    jobs_fetched: int
    elapsed_seconds: float
    message: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["elapsed_seconds"] = round(self.elapsed_seconds, 2)
        return payload


@dataclass(frozen=True)
class SearchConfig:
    keywords: str
    location: str
    sources: list[str]
    limit_per_source: int = 30
    linkedin_detail_delay: float = 0.35
    linkedin_skip_apply_url: bool = False
    web_result_pages: int = 1
    web_max_sites: int = 8
    web_follow_links_per_site: int = 0
    web_links_per_site: int = 4


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def strip_html(value: str | None) -> str:
    if not value:
        return ""
    soup = BeautifulSoup(value, "html.parser")
    return clean_text(soup.get_text(" ", strip=True))


def clean_index_title(value: str) -> str:
    cleaned = clean_text(value)
    cleaned = re.sub(r"^naukri\.com\s+www\.naukri\.com\s+›\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*-\s*naukri\.com\s*$", "", cleaned, flags=re.IGNORECASE)
    prefix_match = re.match(r"^[a-z0-9-]{8,}\s+(.+)$", cleaned)
    if prefix_match:
        cleaned = prefix_match.group(1).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def normalize_date(raw: object) -> str:
    if raw is None:
        return ""

    if isinstance(raw, (int, float)):
        try:
            return datetime.fromtimestamp(float(raw), tz=timezone.utc).date().isoformat()
        except (OverflowError, OSError, ValueError):
            return ""

    value = clean_text(str(raw))
    if not value:
        return ""

    direct = re.search(r"\d{4}-\d{2}-\d{2}", value)
    if direct:
        return direct.group(0)

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.date().isoformat()
    except ValueError:
        return value


def is_remote_label(location: str, title: str, description: str = "") -> bool:
    blob = f"{location} {title} {description}".lower()
    return any(term in blob for term in REMOTE_HINTS)


def keyword_match(keywords: str, text_blob: str) -> bool:
    query = clean_text(keywords).lower()
    if not query:
        return True

    blob = clean_text(text_blob).lower()
    if query in blob:
        return True

    terms = [term for term in re.findall(r"[a-z0-9]+", query) if len(term) > 2]
    if not terms:
        return True

    score = sum(term in blob for term in terms)
    threshold = 1 if len(terms) <= 2 else 2
    return score >= threshold


def location_match(target_location: str, text_blob: str) -> bool:
    wanted = clean_text(target_location).lower()
    if not wanted:
        return True
    if wanted in {"any", "anywhere", "global", "worldwide", "remote"}:
        return True

    blob = clean_text(text_blob).lower()
    if wanted in blob:
        return True

    wanted_terms = [term for term in re.findall(r"[a-z0-9]+", wanted) if len(term) > 2]
    if not wanted_terms:
        return True
    return all(term in blob for term in wanted_terms[:3])


def to_absolute_url(url: str, base: str) -> str:
    cleaned = clean_text(url)
    if not cleaned:
        return ""
    return urljoin(base, html.unescape(cleaned))


def extract_duckduckgo_target(raw_url: str) -> str:
    resolved = html.unescape(clean_text(raw_url))
    if not resolved:
        return ""
    if resolved.startswith("//"):
        resolved = f"https:{resolved}"

    parsed = urlparse(resolved)
    if "duckduckgo.com" in parsed.netloc.lower():
        target = parse_qs(parsed.query).get("uddg", [])
        if target:
            return clean_text(unquote(target[0]))
    return resolved


def extract_yahoo_target(raw_url: str) -> str:
    resolved = html.unescape(clean_text(raw_url))
    if not resolved:
        return ""

    parsed = urlparse(resolved)
    query_params = parse_qs(parsed.query)
    for key in ("RU", "ru", "u", "url"):
        values = query_params.get(key, [])
        if values:
            return clean_text(unquote(values[0]))

    marker_start = "/RU="
    marker_end = "/RK="
    if marker_start in parsed.path and marker_end in parsed.path:
        encoded = parsed.path.split(marker_start, 1)[1].split(marker_end, 1)[0]
        return clean_text(unquote(encoded))
    if marker_start in resolved and marker_end in resolved:
        encoded = resolved.split(marker_start, 1)[1].split(marker_end, 1)[0]
        return clean_text(unquote(encoded))
    return resolved


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    return session


def canonicalize_web_url(raw_url: str) -> str:
    parsed = urlparse(clean_text(raw_url))
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path or '/'}"
    return normalized.rstrip("/")


def root_domain(hostname: str) -> str:
    host = clean_text(hostname).lower().split(":")[0]
    parts = [part for part in host.split(".") if part]
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host


def infer_company_name_from_url(target_url: str) -> str:
    host = urlparse(target_url).netloc.lower()
    host = host.removeprefix("www.")
    core = host.split(":")[0]
    if not core:
        return ""
    parts = [part for part in core.split(".") if part]
    if len(parts) >= 2:
        core = parts[-2]
    return clean_text(core.replace("-", " ").replace("_", " ").title())


def should_skip_web_result(target_url: str) -> bool:
    parsed = urlparse(target_url)
    host = parsed.netloc.lower()
    if not host:
        return True
    if parsed.scheme not in {"http", "https"}:
        return True
    if any(marker in host for marker in IGNORED_WEB_HOST_MARKERS):
        return True
    if target_url.lower().endswith((".pdf", ".doc", ".docx")):
        return True
    return False


def iter_jsonld_nodes(payload: object) -> list[dict[str, object]]:
    nodes: list[dict[str, object]] = []
    if isinstance(payload, dict):
        nodes.append(payload)
        graph = payload.get("@graph")
        if isinstance(graph, list):
            for item in graph:
                nodes.extend(iter_jsonld_nodes(item))
        elif isinstance(graph, dict):
            nodes.extend(iter_jsonld_nodes(graph))
        main_entity = payload.get("mainEntity")
        if isinstance(main_entity, (list, dict)):
            nodes.extend(iter_jsonld_nodes(main_entity))
    elif isinstance(payload, list):
        for item in payload:
            nodes.extend(iter_jsonld_nodes(item))
    return nodes


def extract_jsonld_location(value: object) -> str:
    if isinstance(value, str):
        return clean_text(value)
    if isinstance(value, list):
        parts = [extract_jsonld_location(item) for item in value]
        unique_parts = [part for part in parts if part]
        return clean_text(" / ".join(unique_parts))
    if not isinstance(value, dict):
        return ""

    if clean_text(str(value.get("jobLocationType"))).lower() == "telecommute":
        return "Remote"

    address = value.get("address")
    if isinstance(address, str):
        return clean_text(address)
    if isinstance(address, dict):
        fragments = [
            clean_text(str(address.get("addressLocality") or "")),
            clean_text(str(address.get("addressRegion") or "")),
            clean_text(str(address.get("addressCountry") or "")),
        ]
        joined = ", ".join(fragment for fragment in fragments if fragment)
        if joined:
            return joined

    name = clean_text(str(value.get("name") or ""))
    if name:
        return name
    return ""


def extract_jsonld_salary(value: object) -> str:
    if isinstance(value, (int, float)):
        return clean_text(str(value))
    if isinstance(value, str):
        return clean_text(value)
    if not isinstance(value, dict):
        return ""

    currency = clean_text(str(value.get("currency") or ""))
    amount = value.get("value")
    unit = ""
    amount_text = ""
    if isinstance(amount, dict):
        min_value = clean_text(str(amount.get("minValue") or ""))
        max_value = clean_text(str(amount.get("maxValue") or ""))
        exact_value = clean_text(str(amount.get("value") or ""))
        if min_value or max_value:
            amount_text = f"{min_value}-{max_value}".strip("-")
        else:
            amount_text = exact_value
        unit = clean_text(str(amount.get("unitText") or ""))
    elif isinstance(amount, (int, float, str)):
        amount_text = clean_text(str(amount))
    parts = [part for part in (currency, amount_text, unit) if part]
    return clean_text(" ".join(parts))


def jsonld_type_contains_job_posting(value: object) -> bool:
    if isinstance(value, str):
        return clean_text(value).lower() == "jobposting"
    if isinstance(value, list):
        return any(jsonld_type_contains_job_posting(item) for item in value)
    return False


def parse_jobs_from_jsonld(
    soup: BeautifulSoup,
    *,
    page_url: str,
    fallback_company: str,
    config: SearchConfig,
    max_jobs: int,
) -> list[UnifiedJob]:
    collected: list[UnifiedJob] = []
    seen: set[str] = set()

    for script in soup.select("script[type='application/ld+json']"):
        raw_json = script.string if script.string else script.get_text(" ", strip=True)
        raw_json = raw_json.strip() if raw_json else ""
        if not raw_json:
            continue

        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError:
            continue

        for node in iter_jsonld_nodes(payload):
            if not jsonld_type_contains_job_posting(node.get("@type")):
                continue

            title = clean_text(str(node.get("title") or node.get("name") or ""))
            description = strip_html(str(node.get("description") or ""))
            hiring = node.get("hiringOrganization")
            company = fallback_company
            if isinstance(hiring, str):
                company = clean_text(hiring) or company
            elif isinstance(hiring, dict):
                company = clean_text(str(hiring.get("name") or "")) or company

            location = extract_jsonld_location(node.get("jobLocation"))
            if not location and clean_text(str(node.get("jobLocationType") or "")).lower() == "telecommute":
                location = "Remote"

            listed_at = clean_text(str(node.get("datePosted") or node.get("validThrough") or ""))
            listed_date = normalize_date(listed_at)
            salary = extract_jsonld_salary(node.get("baseSalary"))
            employment_type_raw = node.get("employmentType")
            if isinstance(employment_type_raw, list):
                employment_type = ", ".join(
                    clean_text(str(item))
                    for item in employment_type_raw
                    if clean_text(str(item))
                )
            else:
                employment_type = clean_text(str(employment_type_raw or ""))

            job_url = to_absolute_url(clean_text(str(node.get("url") or "")), page_url)
            if not job_url:
                job_url = clean_text(page_url)
            dedupe_key = canonicalize_web_url(job_url) or f"{title}|{company}|{location}"
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            blob = f"{title} {company} {location} {description}"
            if not keyword_match(config.keywords, blob):
                continue
            if config.location and location and not location_match(config.location, blob):
                continue

            job_id = "career-" + hashlib.sha1(dedupe_key.encode("utf-8")).hexdigest()[:14]
            collected.append(
                UnifiedJob(
                    source=WEB_COMPANY_CAREERS,
                    keywords=config.keywords,
                    title=title or "Job Opening",
                    company=company,
                    location=location or config.location,
                    listed_at=listed_at,
                    listed_date=listed_date,
                    job_id=job_id,
                    job_url=job_url,
                    apply_url=job_url,
                    employment_type=employment_type,
                    salary=salary,
                    is_remote=is_remote_label(location, title, description),
                    description_snippet=description[:260],
                )
            )
            if len(collected) >= max_jobs:
                return collected

    return collected


def score_link_for_hints(anchor_text: str, target_url: str, hints: tuple[str, ...]) -> int:
    parsed = urlparse(target_url)
    signal_blob = f"{anchor_text} {parsed.path} {parsed.query}".lower()
    return sum(hint in signal_blob for hint in hints)


def extract_candidate_links(
    soup: BeautifulSoup,
    *,
    page_url: str,
    hints: tuple[str, ...],
    max_links: int,
) -> list[tuple[str, str, str]]:
    base_host = urlparse(page_url).netloc.lower()
    base_root = root_domain(base_host)
    candidates: list[tuple[int, str, str, str]] = []
    seen_urls: set[str] = set()

    for anchor in soup.select("a[href]"):
        href = to_absolute_url(clean_text(anchor.get("href")), page_url)
        if not href:
            continue
        parsed = urlparse(href)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        if base_root and base_root not in root_domain(parsed.netloc.lower()):
            continue

        canonical = canonicalize_web_url(href)
        if not canonical or canonical in seen_urls:
            continue
        seen_urls.add(canonical)

        anchor_text = clean_text(anchor.get_text(" ", strip=True))
        if not anchor_text and not parsed.path:
            continue

        score = score_link_for_hints(anchor_text, href, hints)
        if score <= 0:
            continue

        context_text = clean_text(anchor.parent.get_text(" ", strip=True) if anchor.parent else "")
        candidates.append((score, anchor_text, href, context_text))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [(title, url, snippet) for _, title, url, snippet in candidates[:max_links]]


def fetch_html_page(session: requests.Session, target_url: str) -> tuple[str, str]:
    response = session.get(target_url, timeout=7, allow_redirects=True)
    response.raise_for_status()
    content_type = clean_text(response.headers.get("Content-Type", "")).lower()
    if content_type and "html" not in content_type:
        return "", clean_text(response.url or target_url)
    return response.text, clean_text(response.url or target_url)


def crawl_single_career_site(seed_url: str, config: SearchConfig) -> list[UnifiedJob]:
    session = make_session()
    visited_pages: set[str] = set()
    collected: list[UnifiedJob] = []
    seen_jobs: set[str] = set()
    pages_to_visit: list[str] = [seed_url]

    try:
        while pages_to_visit and len(visited_pages) <= config.web_follow_links_per_site:
            current_url = pages_to_visit.pop(0)
            canonical_page = canonicalize_web_url(current_url)
            if not canonical_page or canonical_page in visited_pages:
                continue
            visited_pages.add(canonical_page)

            try:
                page_html, resolved_url = fetch_html_page(session, current_url)
            except requests.RequestException:
                continue
            if not page_html:
                continue

            soup = BeautifulSoup(page_html, "html.parser")
            company = infer_company_name_from_url(resolved_url)
            structured = parse_jobs_from_jsonld(
                soup,
                page_url=resolved_url,
                fallback_company=company,
                config=config,
                max_jobs=config.web_links_per_site,
            )

            for job in structured:
                dedupe_key = canonicalize_web_url(job.job_url) or job.job_id
                if dedupe_key in seen_jobs:
                    continue
                seen_jobs.add(dedupe_key)
                collected.append(job)
                if len(collected) >= config.limit_per_source:
                    return collected

            if len(visited_pages) == 1:
                career_links = extract_candidate_links(
                    soup,
                    page_url=resolved_url,
                    hints=CAREER_LINK_HINTS,
                    max_links=config.web_follow_links_per_site,
                )
                for _, career_url, _ in career_links:
                    canonical_career = canonicalize_web_url(career_url)
                    if canonical_career and canonical_career not in visited_pages:
                        pages_to_visit.append(career_url)

            if structured:
                continue

            fallback_links = extract_candidate_links(
                soup,
                page_url=resolved_url,
                hints=JOB_LINK_HINTS,
                max_links=config.web_links_per_site,
            )
            for title, job_url, snippet in fallback_links:
                snippet_text = clean_text(snippet)[:260]
                clean_title = clean_text(title) or "Job Opening"
                fallback_company = company or infer_company_name_from_url(job_url)
                blob = f"{clean_title} {fallback_company} {snippet_text}"
                if not keyword_match(config.keywords, blob):
                    continue

                dedupe_key = canonicalize_web_url(job_url) or f"{clean_title}|{fallback_company}"
                if dedupe_key in seen_jobs:
                    continue
                seen_jobs.add(dedupe_key)
                job_id = "career-" + hashlib.sha1(dedupe_key.encode("utf-8")).hexdigest()[:14]
                collected.append(
                    UnifiedJob(
                        source=WEB_COMPANY_CAREERS,
                        keywords=config.keywords,
                        title=clean_title,
                        company=fallback_company,
                        location=config.location or "",
                        listed_at="",
                        listed_date="",
                        job_id=job_id,
                        job_url=job_url,
                        apply_url=job_url,
                        employment_type="",
                        salary="",
                        is_remote=is_remote_label(config.location, clean_title, snippet_text),
                        description_snippet=snippet_text,
                    )
                )
                if len(collected) >= config.limit_per_source:
                    return collected
    finally:
        session.close()

    return collected


def fetch_web_seed_urls(
    config: SearchConfig,
    session: requests.Session,
) -> list[tuple[str, str, str]]:
    pages = max(1, min(4, int(config.web_result_pages or 1)))
    max_sites = max(4, int(config.web_max_sites or 16))
    query = clean_text(f'{config.keywords} {config.location} "careers" "job openings"')

    candidates: list[tuple[str, str, str]] = []
    seen_urls: set[str] = set()

    def add_candidate(url: str, title: str = "", snippet: str = "") -> bool:
        canonical = canonicalize_web_url(url)
        if not canonical or canonical in seen_urls:
            return False
        if should_skip_web_result(url):
            return False
        seen_urls.add(canonical)
        candidates.append((url, clean_text(title), clean_text(snippet)))
        return len(candidates) >= max_sites

    for page_index in range(pages):
        try:
            response = session.get(
                YAHOO_SEARCH_ENDPOINT,
                params={"p": query, "b": (page_index * 10) + 1},
                timeout=14,
                headers=DEFAULT_HEADERS,
            )
            response.raise_for_status()
        except requests.RequestException:
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        nodes = soup.select("div#web ol li") or soup.select("div#web li")
        for node in nodes:
            anchor = node.select_one("a")
            if not anchor:
                continue
            target_url = extract_yahoo_target(clean_text(anchor.get("href")))
            title = clean_text(anchor.get_text(" ", strip=True))
            snippet_tag = node.select_one("p")
            snippet = clean_text(snippet_tag.get_text(" ", strip=True) if snippet_tag else "")
            if add_candidate(target_url, title, snippet):
                return candidates
        time.sleep(0.10)

    for page_index in range(pages):
        try:
            response = session.get(
                DUCKDUCKGO_HTML_ENDPOINT,
                params={"q": query, "s": page_index * 30},
                timeout=14,
                headers=DEFAULT_HEADERS,
            )
            response.raise_for_status()
        except requests.RequestException:
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        for node in soup.select("div.result"):
            anchor = node.select_one("a.result__a")
            if not anchor:
                continue
            target_url = extract_duckduckgo_target(clean_text(anchor.get("href")))
            title = clean_text(anchor.get_text(" ", strip=True))
            snippet_tag = node.select_one(".result__snippet")
            snippet = clean_text(snippet_tag.get_text(" ", strip=True) if snippet_tag else "")
            if add_candidate(target_url, title, snippet):
                return candidates
        time.sleep(0.10)

    return candidates


def fetch_web_company_jobs(config: SearchConfig, session: requests.Session) -> list[UnifiedJob]:
    seed_rows = fetch_web_seed_urls(config, session)
    if not seed_rows:
        return []
    seed_urls = [url for url, _, _ in seed_rows]
    crawl_site_cap = max(4, min(len(seed_urls), int(config.limit_per_source) + 2))
    seed_urls = seed_urls[:crawl_site_cap]

    collected: list[UnifiedJob] = []
    seen_jobs: set[str] = set()
    workers = max(1, min(CAREER_CRAWL_WORKERS, len(seed_urls), max(2, int(config.limit_per_source))))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(crawl_single_career_site, url, config) for url in seed_urls]
        for future in as_completed(futures):
            try:
                jobs = future.result()
            except Exception:  # pylint: disable=broad-exception-caught
                continue

            for job in jobs:
                dedupe_key = canonicalize_web_url(job.job_url) or job.job_id
                if dedupe_key in seen_jobs:
                    continue
                seen_jobs.add(dedupe_key)
                collected.append(job)
                if len(collected) >= config.limit_per_source:
                    return collected

    # If direct crawling yields sparse data, preserve relevant search hits.
    if len(collected) < config.limit_per_source:
        for seed_url, seed_title, seed_snippet in seed_rows:
            dedupe_key = canonicalize_web_url(seed_url) or seed_url
            if dedupe_key in seen_jobs:
                continue

            company = infer_company_name_from_url(seed_url)
            title = seed_title or "Career Page"
            blob = f"{title} {company} {seed_snippet}"
            if not keyword_match(config.keywords, blob):
                continue

            seen_jobs.add(dedupe_key)
            job_id = "career-" + hashlib.sha1(dedupe_key.encode("utf-8")).hexdigest()[:14]
            collected.append(
                UnifiedJob(
                    source=WEB_COMPANY_CAREERS,
                    keywords=config.keywords,
                    title=title,
                    company=company,
                    location=config.location or "",
                    listed_at="",
                    listed_date="",
                    job_id=job_id,
                    job_url=seed_url,
                    apply_url=seed_url,
                    employment_type="",
                    salary="",
                    is_remote=is_remote_label(config.location, title, seed_snippet),
                    description_snippet=seed_snippet[:260],
                )
            )
            if len(collected) >= config.limit_per_source:
                break

    return collected


def fetch_linkedin_jobs(config: SearchConfig) -> list[UnifiedJob]:
    client = LinkedInPublicJobsClient(timeout_seconds=20, max_retries=3)
    jobs = client.search_jobs(
        keywords=config.keywords,
        location=config.location,
        limit=config.limit_per_source,
        include_apply_url=not config.linkedin_skip_apply_url,
        detail_delay_seconds=config.linkedin_detail_delay,
    )
    converted: list[UnifiedJob] = []
    for job in jobs:
        listed_date = normalize_date(job.listed_at)
        converted.append(
            UnifiedJob(
                source=LINKEDIN,
                keywords=config.keywords,
                title=clean_text(job.title),
                company=clean_text(job.company),
                location=clean_text(job.location),
                listed_at=clean_text(job.listed_at),
                listed_date=listed_date,
                job_id=clean_text(job.job_id),
                job_url=clean_text(job.job_url),
                apply_url=clean_text(job.apply_url or job.job_url),
                employment_type="",
                salary="",
                is_remote=is_remote_label(job.location, job.title),
                description_snippet="",
            )
        )
    return converted


def detect_indeed_block(response: requests.Response) -> None:
    text = response.text.lower()
    if response.status_code in {401, 403, 429}:
        raise SourceBlockedError(
            "Indeed is blocking this request from your current network/session."
        )
    blocked_markers = (
        "attention required",
        "sorry, you have been blocked",
        "cf-error-details",
        "cloudflare",
    )
    if any(marker in text for marker in blocked_markers):
        raise SourceBlockedError(
            "Indeed returned an anti-bot challenge (Cloudflare)."
        )


def fetch_indeed_jobs(config: SearchConfig, session: requests.Session) -> list[UnifiedJob]:
    collected: list[UnifiedJob] = []
    seen_ids: set[str] = set()
    page_size = 10
    start = 0

    while len(collected) < config.limit_per_source:
        response = session.get(
            "https://www.indeed.com/jobs",
            params={
                "q": config.keywords,
                "l": config.location,
                "start": start,
            },
            timeout=25,
        )
        detect_indeed_block(response)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        cards = soup.select("a.tapItem") or soup.select("div.job_seen_beacon")
        if not cards:
            break

        for card in cards:
            root = card
            if root.name != "a":
                anchor = root.select_one("a.jcs-JobTitle")
                if anchor:
                    root = anchor

            raw_id = clean_text(card.get("data-jk")) or clean_text(root.get("data-jk"))
            raw_href = clean_text(root.get("href"))
            job_url = to_absolute_url(raw_href, "https://www.indeed.com")
            job_id = raw_id or clean_text(job_url.split("jk=")[-1] if "jk=" in job_url else "")
            if not job_id:
                job_id = clean_text(job_url)
            if job_id in seen_ids:
                continue
            seen_ids.add(job_id)

            title_tag = card.select_one("span[title]") or card.select_one("h2.jobTitle span")
            title = clean_text(title_tag.get("title") if title_tag and title_tag.get("title") else "")
            if not title and title_tag:
                title = clean_text(title_tag.get_text(" ", strip=True))
            if not title:
                title = clean_text(root.get("aria-label") or "")

            company = clean_text(
                card.select_one("span.companyName").get_text(" ", strip=True)
                if card.select_one("span.companyName")
                else ""
            )
            location = clean_text(
                card.select_one("div.companyLocation").get_text(" ", strip=True)
                if card.select_one("div.companyLocation")
                else ""
            )
            listed_at = clean_text(
                card.select_one("span.date").get_text(" ", strip=True)
                if card.select_one("span.date")
                else ""
            )

            if not title and not company:
                continue

            collected.append(
                UnifiedJob(
                    source=INDEED,
                    keywords=config.keywords,
                    title=title,
                    company=company,
                    location=location,
                    listed_at=listed_at,
                    listed_date=normalize_date(listed_at),
                    job_id=job_id,
                    job_url=job_url,
                    apply_url=job_url,
                    employment_type="",
                    salary="",
                    is_remote=is_remote_label(location, title),
                    description_snippet="",
                )
            )
            if len(collected) >= config.limit_per_source:
                break

        start += page_size
        time.sleep(0.35)

    if not collected:
        raise SourceBlockedError(
            "Indeed returned no parseable cards; this usually means anti-bot protection."
        )
    return collected


def fetch_naukri_jobs(config: SearchConfig, session: requests.Session) -> list[UnifiedJob]:
    endpoint = "https://www.naukri.com/jobapi/v3/search"
    page = 1
    page_size = 20
    collected: list[UnifiedJob] = []
    seen_ids: set[str] = set()

    while len(collected) < config.limit_per_source:
        params = {
            "noOfResults": page_size,
            "keyword": config.keywords,
            "location": config.location,
            "pageNo": page,
            "urlType": "search_by_keyword",
            "searchType": "adv",
        }
        headers = {
            **DEFAULT_HEADERS,
            "Accept": "application/json, text/plain, */*",
            "appid": "109",
            "systemid": "109",
            "referer": (
                "https://www.naukri.com/software-engineer-jobs?"
                + urlencode({"k": config.keywords})
            ),
        }

        response = session.get(endpoint, params=params, headers=headers, timeout=25)
        lowered = response.text.lower()
        if response.status_code == 406 and "recaptcha required" in lowered:
            fallback_jobs = fetch_naukri_jobs_via_search_index(config, session)
            if fallback_jobs:
                return fallback_jobs
            raise SourceBlockedError(
                "Naukri API requires captcha/session verification for this request."
            )
        if response.status_code in {401, 403, 429}:
            fallback_jobs = fetch_naukri_jobs_via_search_index(config, session)
            if fallback_jobs:
                return fallback_jobs
            raise SourceBlockedError(
                f"Naukri blocked the request with HTTP {response.status_code}."
            )
        response.raise_for_status()

        payload = response.json()
        details = payload.get("jobDetails") or payload.get("jobdetails") or []
        if not details:
            break

        for detail in details:
            job_id = clean_text(str(detail.get("jobId") or detail.get("id") or ""))
            if not job_id or job_id in seen_ids:
                continue
            seen_ids.add(job_id)

            title = clean_text(detail.get("title") or detail.get("jobTitle") or "")
            company = clean_text(detail.get("companyName") or detail.get("company") or "")
            location = clean_text(detail.get("location") or "")
            salary = clean_text(detail.get("salary") or "")
            employment_type = clean_text(detail.get("experienceText") or "")

            for placeholder in detail.get("placeholders") or []:
                p_type = clean_text(str(placeholder.get("type") or "")).lower()
                label = clean_text(placeholder.get("label") or placeholder.get("value") or "")
                if not label:
                    continue
                if p_type in {"location", "loc"} and not location:
                    location = label
                elif p_type in {"salary", "sal"} and not salary:
                    salary = label
                elif p_type in {"experience", "exp"} and not employment_type:
                    employment_type = label

            listed_at = clean_text(
                detail.get("displayDate")
                or detail.get("postedDate")
                or detail.get("createdDate")
                or ""
            )
            job_url = clean_text(detail.get("jdURL") or detail.get("jobUrl") or detail.get("url") or "")
            if job_url.startswith("/"):
                job_url = f"https://www.naukri.com{job_url}"
            apply_url = job_url

            text_blob = f"{title} {company} {location}"
            if not keyword_match(config.keywords, text_blob):
                continue
            if not location_match(config.location, f"{location} {title}"):
                continue

            collected.append(
                UnifiedJob(
                    source=NAUKRI,
                    keywords=config.keywords,
                    title=title,
                    company=company,
                    location=location,
                    listed_at=listed_at,
                    listed_date=normalize_date(listed_at),
                    job_id=job_id,
                    job_url=job_url,
                    apply_url=apply_url,
                    employment_type=employment_type,
                    salary=salary,
                    is_remote=is_remote_label(location, title),
                    description_snippet="",
                )
            )
            if len(collected) >= config.limit_per_source:
                break

        page += 1
        time.sleep(0.4)

    if not collected:
        fallback_jobs = fetch_naukri_jobs_via_search_index(config, session)
        if fallback_jobs:
            return fallback_jobs
        raise SourceBlockedError(
            "Naukri returned no accessible jobs for this automated request."
        )
    return collected


def fetch_naukri_jobs_via_search_index(
    config: SearchConfig, session: requests.Session
) -> list[UnifiedJob]:
    query_parts = ["site:naukri.com", "job-listings", config.keywords, config.location]
    query = " ".join(clean_text(part) for part in query_parts if clean_text(part))
    if not query:
        return []

    candidates: list[tuple[str, str, str]] = []

    # Yahoo tends to be the most stable fallback index in this environment.
    try:
        yahoo_response = session.get(
            YAHOO_SEARCH_ENDPOINT,
            params={"p": query},
            timeout=20,
            headers=DEFAULT_HEADERS,
        )
        yahoo_response.raise_for_status()
        yahoo_soup = BeautifulSoup(yahoo_response.text, "html.parser")
        yahoo_nodes = yahoo_soup.select("div#web ol li") or yahoo_soup.select("div#web li")
        for node in yahoo_nodes:
            anchor = node.select_one("a")
            if not anchor:
                continue
            title = clean_index_title(anchor.get_text(" ", strip=True))
            target_url = extract_yahoo_target(clean_text(anchor.get("href")))
            snippet_tag = node.select_one("p")
            snippet = clean_text(snippet_tag.get_text(" ", strip=True) if snippet_tag else "")
            if target_url:
                candidates.append((title, snippet, target_url))
    except requests.RequestException:
        pass

    # Secondary engine if Yahoo produced nothing.
    if not candidates:
        try:
            duck_response = session.get(
                DUCKDUCKGO_HTML_ENDPOINT,
                params={"q": query},
                timeout=15,
                headers=DEFAULT_HEADERS,
            )
            duck_response.raise_for_status()
            duck_soup = BeautifulSoup(duck_response.text, "html.parser")
            for node in duck_soup.select("div.result"):
                link = node.select_one("a.result__a")
                if not link:
                    continue
                title = clean_index_title(link.get_text(" ", strip=True))
                target_url = extract_duckduckgo_target(clean_text(link.get("href")))
                snippet_tag = node.select_one(".result__snippet")
                snippet = clean_text(snippet_tag.get_text(" ", strip=True) if snippet_tag else "")
                if target_url:
                    candidates.append((title, snippet, target_url))
        except requests.RequestException:
            return []

    collected: list[UnifiedJob] = []
    seen_urls: set[str] = set()

    for title, snippet, resolved_url in candidates:
        if not resolved_url:
            continue

        parsed = urlparse(resolved_url)
        host = parsed.netloc.lower()
        if "naukri.com" not in host:
            continue
        if "/job-listings" not in parsed.path and not parsed.path.endswith("-jobs"):
            continue
        if resolved_url in seen_urls:
            continue
        seen_urls.add(resolved_url)

        if not keyword_match(config.keywords, f"{title} {snippet}"):
            continue

        slug = parsed.path.strip("/").split("/")[-1]
        match = re.search(r"(\d{10,})$", slug)
        base_id = match.group(1) if match else hashlib.sha1(resolved_url.encode("utf-8")).hexdigest()[:12]
        job_id = f"fallback-{base_id}"

        collected.append(
            UnifiedJob(
                source=NAUKRI,
                keywords=config.keywords,
                title=title,
                company="",
                location=config.location,
                listed_at="",
                listed_date="",
                job_id=job_id,
                job_url=resolved_url,
                apply_url=resolved_url,
                employment_type="",
                salary="",
                is_remote=is_remote_label(config.location, title, snippet),
                description_snippet=snippet,
            )
        )
        if len(collected) >= config.limit_per_source:
            break

    return collected


def fetch_remoteok_jobs(config: SearchConfig, session: requests.Session) -> list[UnifiedJob]:
    response = session.get(
        "https://remoteok.com/api",
        timeout=25,
        headers={**DEFAULT_HEADERS, "Accept": "application/json"},
    )
    response.raise_for_status()
    payload = response.json()
    records = payload if isinstance(payload, list) else []

    collected: list[UnifiedJob] = []
    for item in records:
        if not isinstance(item, dict):
            continue
        if not item.get("id"):
            continue

        title = clean_text(item.get("position") or item.get("title") or "")
        company = clean_text(item.get("company") or "")
        location = clean_text(item.get("location") or "Remote")
        description = strip_html(item.get("description") or "")
        tags = item.get("tags") or []
        tags_text = " ".join(str(tag) for tag in tags)
        blob = f"{title} {company} {location} {tags_text} {description}"
        if not keyword_match(config.keywords, blob):
            continue
        if config.location and not location_match(config.location, blob):
            continue

        min_salary = item.get("salary_min")
        max_salary = item.get("salary_max")
        salary = ""
        if min_salary or max_salary:
            salary = f"{min_salary or ''}-{max_salary or ''}".strip("-")
        job_url = to_absolute_url(item.get("url") or "", "https://remoteok.com")
        apply_url = to_absolute_url(item.get("apply_url") or item.get("url") or "", "https://remoteok.com")

        collected.append(
            UnifiedJob(
                source=REMOTEOK,
                keywords=config.keywords,
                title=title,
                company=company,
                location=location or "Remote",
                listed_at=clean_text(item.get("date") or ""),
                listed_date=normalize_date(item.get("date") or item.get("epoch")),
                job_id=clean_text(str(item.get("id") or "")),
                job_url=job_url,
                apply_url=apply_url or job_url,
                employment_type="",
                salary=salary,
                is_remote=True,
                description_snippet=description[:260],
            )
        )
        if len(collected) >= config.limit_per_source:
            break

    return collected


def fetch_remotive_jobs(config: SearchConfig, session: requests.Session) -> list[UnifiedJob]:
    response = session.get(
        "https://remotive.com/api/remote-jobs",
        params={"search": config.keywords},
        timeout=25,
    )
    response.raise_for_status()
    payload = response.json()
    records = payload.get("jobs") if isinstance(payload, dict) else []
    collected: list[UnifiedJob] = []

    for item in records or []:
        title = clean_text(item.get("title") or "")
        company = clean_text(item.get("company_name") or "")
        location = clean_text(item.get("candidate_required_location") or "Remote")
        description = strip_html(item.get("description") or "")
        tags = item.get("tags") or []
        blob = f"{title} {company} {location} {' '.join(tags)} {description}"
        if not keyword_match(config.keywords, blob):
            continue
        if config.location and not location_match(config.location, blob):
            continue

        job_url = clean_text(item.get("url") or "")
        collected.append(
            UnifiedJob(
                source=REMOTIVE,
                keywords=config.keywords,
                title=title,
                company=company,
                location=location,
                listed_at=clean_text(item.get("publication_date") or ""),
                listed_date=normalize_date(item.get("publication_date")),
                job_id=clean_text(str(item.get("id") or "")),
                job_url=job_url,
                apply_url=job_url,
                employment_type=clean_text(item.get("job_type") or ""),
                salary=clean_text(item.get("salary") or ""),
                is_remote=True,
                description_snippet=description[:260],
            )
        )
        if len(collected) >= config.limit_per_source:
            break

    return collected


def fetch_arbeitnow_jobs(config: SearchConfig, session: requests.Session) -> list[UnifiedJob]:
    next_url = "https://www.arbeitnow.com/api/job-board-api?limit=200&page=1"
    collected: list[UnifiedJob] = []
    visited_pages = 0
    max_pages = 6

    while next_url and len(collected) < config.limit_per_source and visited_pages < max_pages:
        visited_pages += 1
        response = session.get(next_url, timeout=25)
        response.raise_for_status()
        payload = response.json()
        records = payload.get("data") or []

        for item in records:
            title = clean_text(item.get("title") or "")
            company = clean_text(item.get("company_name") or "")
            location = clean_text(item.get("location") or "")
            description = strip_html(item.get("description") or "")
            tags = item.get("tags") or []
            tags_text = " ".join(tags) if isinstance(tags, list) else clean_text(str(tags))
            blob = f"{title} {company} {location} {description} {tags_text}"
            if not keyword_match(config.keywords, blob):
                continue
            if config.location and not location_match(config.location, blob):
                continue

            job_url = clean_text(item.get("url") or "")
            job_id = clean_text(item.get("slug") or "")
            salary = ""
            job_types = item.get("job_types") or []
            employment_type = ", ".join(clean_text(str(job_type)) for job_type in job_types if clean_text(str(job_type)))

            collected.append(
                UnifiedJob(
                    source=ARBEITNOW,
                    keywords=config.keywords,
                    title=title,
                    company=company,
                    location=location,
                    listed_at=clean_text(str(item.get("created_at") or "")),
                    listed_date=normalize_date(item.get("created_at")),
                    job_id=job_id,
                    job_url=job_url,
                    apply_url=job_url,
                    employment_type=employment_type,
                    salary=salary,
                    is_remote=bool(item.get("remote")) or is_remote_label(location, title, description),
                    description_snippet=description[:260],
                )
            )
            if len(collected) >= config.limit_per_source:
                break

        next_url = (payload.get("links") or {}).get("next")
        time.sleep(0.2)

    return collected


FETCHERS = {
    LINKEDIN: lambda cfg, sess: fetch_linkedin_jobs(cfg),
    INDEED: fetch_indeed_jobs,
    NAUKRI: fetch_naukri_jobs,
    REMOTEOK: fetch_remoteok_jobs,
    REMOTIVE: fetch_remotive_jobs,
    ARBEITNOW: fetch_arbeitnow_jobs,
    WEB_COMPANY_CAREERS: fetch_web_company_jobs,
}


def make_search_cache_key(config: SearchConfig, selected_sources: list[str]) -> tuple[object, ...]:
    return (
        clean_text(config.keywords).lower(),
        clean_text(config.location).lower(),
        tuple(selected_sources),
        int(config.limit_per_source),
        round(float(config.linkedin_detail_delay), 2),
        bool(config.linkedin_skip_apply_url),
        int(config.web_result_pages),
        int(config.web_max_sites),
        int(config.web_follow_links_per_site),
        int(config.web_links_per_site),
    )


def get_cached_search_result(
    cache_key: tuple[object, ...],
) -> tuple[list[UnifiedJob], list[SourceRunReport]] | None:
    with _SEARCH_CACHE_LOCK:
        cached = _SEARCH_CACHE.get(cache_key)
        if not cached:
            return None
        captured_at, jobs, reports = cached
        if (time.time() - captured_at) > SEARCH_CACHE_TTL_SECONDS:
            _SEARCH_CACHE.pop(cache_key, None)
            return None
        return list(jobs), list(reports)


def set_cached_search_result(
    cache_key: tuple[object, ...],
    jobs: list[UnifiedJob],
    reports: list[SourceRunReport],
) -> None:
    with _SEARCH_CACHE_LOCK:
        _SEARCH_CACHE[cache_key] = (time.time(), list(jobs), list(reports))
        if len(_SEARCH_CACHE) > SEARCH_CACHE_MAX_ENTRIES:
            oldest_key = min(_SEARCH_CACHE.items(), key=lambda item: item[1][0])[0]
            _SEARCH_CACHE.pop(oldest_key, None)


def run_single_source(source: str, config: SearchConfig) -> tuple[str, list[UnifiedJob], SourceRunReport]:
    started = perf_counter()
    status = "success"
    message = "ok"
    jobs: list[UnifiedJob] = []
    session = make_session()

    try:
        jobs = FETCHERS[source](config, session)
        if len(jobs) == 0:
            status = "empty"
            message = "no matching jobs returned"
        elif source == NAUKRI and all(str(job.job_id).startswith("fallback-") for job in jobs):
            message = f"API blocked by captcha; fallback index returned {len(jobs)} job(s)"
        else:
            message = f"fetched {len(jobs)} job(s)"
    except SourceBlockedError as exc:
        status = "blocked"
        message = str(exc)
    except requests.RequestException as exc:
        status = "error"
        message = f"request failed: {exc}"
    except Exception as exc:  # pylint: disable=broad-exception-caught
        status = "error"
        message = f"unexpected error: {exc}"
    finally:
        session.close()

    elapsed = perf_counter() - started
    report = SourceRunReport(
        source=source,
        status=status,
        jobs_fetched=len(jobs),
        elapsed_seconds=elapsed,
        message=message,
    )
    return source, jobs, report


def run_multi_source_search(
    config: SearchConfig,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[UnifiedJob], list[SourceRunReport]]:
    selected_sources = [source for source in config.sources if source in FETCHERS]
    if not selected_sources:
        return [], []

    cache_key = make_search_cache_key(config, selected_sources)
    cached = get_cached_search_result(cache_key)
    if cached:
        cached_jobs, cached_reports = cached
        if progress_callback:
            progress_callback(0, 1, "Using cached results")
            progress_callback(1, 1, f"Loaded cached results: {len(cached_jobs)} jobs")
        return cached_jobs, cached_reports

    total = len(selected_sources)
    source_jobs: dict[str, list[UnifiedJob]] = {}
    source_reports: dict[str, SourceRunReport] = {}
    completed = 0

    if progress_callback:
        progress_callback(0, total, f"Launching {total} source(s) in parallel")

    workers = max(1, min(SEARCH_WORKERS, total))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_single_source, source, config) for source in selected_sources]
        for future in as_completed(futures):
            source, jobs, report = future.result()
            source_jobs[source] = jobs
            source_reports[source] = report
            completed += 1
            if progress_callback:
                progress_callback(completed, total, f"{source}: {report.message}")

    all_jobs: list[UnifiedJob] = []
    reports: list[SourceRunReport] = []
    for source in selected_sources:
        jobs = source_jobs.get(source, [])
        report = source_reports.get(source)
        all_jobs.extend(jobs)
        if report:
            reports.append(report)

    set_cached_search_result(cache_key, all_jobs, reports)
    return all_jobs, reports


def clear_search_cache() -> None:
    with _SEARCH_CACHE_LOCK:
        _SEARCH_CACHE.clear()
