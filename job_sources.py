from __future__ import annotations

import hashlib
import html
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from time import perf_counter
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
DUCKDUCKGO_HTML_ENDPOINT = "https://duckduckgo.com/html/"
YAHOO_SEARCH_ENDPOINT = "https://search.yahoo.com/search"

ALL_SOURCES = [LINKEDIN, INDEED, NAUKRI, REMOTEOK, REMOTIVE, ARBEITNOW]
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

    marker_start = "/RU="
    marker_end = "/RK="
    if marker_start in resolved and marker_end in resolved:
        encoded = resolved.split(marker_start, 1)[1].split(marker_end, 1)[0]
        return clean_text(unquote(encoded))
    return resolved


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    return session


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
}


def run_multi_source_search(
    config: SearchConfig,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[UnifiedJob], list[SourceRunReport]]:
    selected_sources = [source for source in config.sources if source in FETCHERS]
    if not selected_sources:
        return [], []

    session = make_session()
    all_jobs: list[UnifiedJob] = []
    reports: list[SourceRunReport] = []
    total = len(selected_sources)

    for index, source in enumerate(selected_sources, start=1):
        if progress_callback:
            progress_callback(index - 1, total, f"{source}: starting")

        started = perf_counter()
        status = "success"
        message = "ok"
        jobs: list[UnifiedJob] = []

        try:
            jobs = FETCHERS[source](config, session)
            if len(jobs) == 0:
                status = "empty"
                message = "no matching jobs returned"
            elif source == NAUKRI and all(str(job.job_id).startswith("fallback-") for job in jobs):
                message = (
                    f"API blocked by captcha; fallback index returned {len(jobs)} job(s)"
                )
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

        elapsed = perf_counter() - started
        reports.append(
            SourceRunReport(
                source=source,
                status=status,
                jobs_fetched=len(jobs),
                elapsed_seconds=elapsed,
                message=message,
            )
        )
        all_jobs.extend(jobs)

        if progress_callback:
            progress_callback(index, total, f"{source}: {message}")

    return all_jobs, reports
