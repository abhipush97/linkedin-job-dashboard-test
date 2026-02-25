from __future__ import annotations

import hmac
import html
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from job_sources import (
    ALL_SOURCES,
    DEFAULT_UI_SOURCES,
    SearchConfig,
    run_multi_source_search,
)

DEFAULT_LOGIN_USERNAME = "recruiter"
DEFAULT_LOGIN_PASSWORD = "linkedin123"


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=Public+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');

        :root {
            --brand: #0a66c2;
            --brand-strong: #004182;
            --surface: #ffffff;
            --surface-soft: #f3f6f9;
            --line: #d5e3f1;
            --text: #1d2226;
            --muted: #536471;
        }

        .stApp {
            background:
                radial-gradient(1200px 460px at -15% -20%, rgba(10, 102, 194, 0.16), transparent 62%),
                radial-gradient(1100px 500px at 118% -20%, rgba(0, 65, 130, 0.10), transparent 64%),
                linear-gradient(180deg, #f5f9fd 0%, #edf3fa 42%, #f7fbff 100%);
            color: var(--text);
        }

        .block-container {
            max-width: 1280px;
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }

        html, body, [class*="css"] {
            font-family: 'Public Sans', sans-serif;
            color: var(--text);
        }

        h1, h2, h3, h4 {
            font-family: 'Manrope', sans-serif;
            color: var(--text);
            letter-spacing: -0.02em;
        }

        .mono {
            font-family: 'JetBrains Mono', monospace;
        }

        section[data-testid="stSidebar"] > div {
            background: linear-gradient(180deg, #ffffff 0%, #f4f8fc 100%);
            border-right: 1px solid var(--line);
        }

        .hero-banner {
            position: relative;
            overflow: hidden;
            border-radius: 1rem;
            border: 1px solid var(--line);
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.96), rgba(239, 246, 255, 0.95));
            padding: 1.15rem 1.3rem;
            margin-bottom: 1rem;
        }

        .hero-eyebrow {
            font-size: 0.73rem;
            font-family: 'Manrope', sans-serif;
            letter-spacing: 0.10em;
            font-weight: 800;
            text-transform: uppercase;
            color: var(--brand-strong);
        }

        .hero-title {
            margin: 0.25rem 0 0.35rem 0;
            font-size: 1.85rem;
            line-height: 1.1;
            font-weight: 800;
            color: var(--text);
        }

        .hero-subtitle {
            margin: 0;
            max-width: 780px;
            color: var(--muted);
            font-size: 0.97rem;
            line-height: 1.45;
        }

        .hero-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.8rem;
        }

        .hero-chip {
            font-size: 0.76rem;
            color: var(--text);
            border-radius: 999px;
            border: 1px solid var(--line);
            background: var(--surface);
            padding: 0.26rem 0.62rem;
        }

        .auth-banner {
            margin-bottom: 0.7rem;
        }

        .login-shell {
            border-radius: 1rem;
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.95);
            padding: 1rem 1.1rem;
            box-shadow: 0 8px 22px rgba(23, 77, 131, 0.09);
        }

        .metric-chip {
            margin: 0.4rem 0 0.6rem 0;
            padding: 0.55rem 0.75rem;
            border-radius: 0.65rem;
            border: 1px solid var(--line);
            background: var(--surface-soft);
            font-size: 0.91rem;
            color: var(--text);
        }

        div[data-testid="stDataFrame"], div[data-testid="stStatusWidget"] {
            border-radius: 0.78rem;
            border: 1px solid var(--line);
            overflow: hidden;
            background: rgba(255, 255, 255, 0.92);
        }

        div[data-testid="stMetric"] {
            border-radius: 0.9rem;
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.88);
            padding: 0.6rem 0.75rem;
        }

        div[data-testid="stMetricValue"] {
            font-family: 'Manrope', sans-serif;
            color: var(--brand-strong);
            font-weight: 800;
        }

        div[data-baseweb="tab-list"] {
            gap: 0.5rem;
        }

        [data-baseweb="tab"] {
            border-radius: 0.7rem;
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.85);
            font-family: 'Public Sans', sans-serif;
            font-weight: 600;
            padding: 0.42rem 0.85rem;
        }

        [data-baseweb="tab-highlight"] {
            background: var(--brand);
        }

        .stButton > button, .stDownloadButton > button {
            border-radius: 0.68rem;
            border: 1px solid var(--line);
            font-family: 'Public Sans', sans-serif;
            font-weight: 600;
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(180deg, var(--brand), var(--brand-strong));
            color: #ffffff;
            border: 0;
        }

        .stButton > button[kind="secondary"]:hover {
            border-color: rgba(10, 102, 194, 0.55);
            color: var(--brand-strong);
        }

        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(180deg, #1d78d6, var(--brand));
            color: #ffffff;
        }

        [data-testid="stExpander"] {
            border-radius: 0.75rem;
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.72);
        }

        @media (max-width: 900px) {
            .block-container {
                padding-top: 0.9rem;
                padding-left: 0.75rem;
                padding-right: 0.75rem;
            }
            .hero-title {
                font-size: 1.45rem;
            }
            .hero-subtitle {
                font-size: 0.89rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_session_state() -> None:
    defaults = {
        "jobs_rows": [],
        "report_rows": [],
        "last_query": {},
        "is_authenticated": False,
        "active_user": "",
        "auth_error": "",
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def _resolve_auth_value(key: str, fallback: str) -> str:
    secret_value = ""
    try:
        if key in st.secrets:
            secret_value = str(st.secrets[key]).strip()
    except Exception:  # pylint: disable=broad-exception-caught
        secret_value = ""

    env_value = os.getenv(key, "").strip()
    return secret_value or env_value or fallback


def get_auth_credentials() -> tuple[str, str]:
    username = _resolve_auth_value("APP_LOGIN_USERNAME", DEFAULT_LOGIN_USERNAME)
    password = _resolve_auth_value("APP_LOGIN_PASSWORD", DEFAULT_LOGIN_PASSWORD)
    return username, password


def render_login_screen() -> None:
    st.markdown(
        """
        <div class="hero-banner auth-banner">
            <div class="hero-eyebrow">Secure Workspace</div>
            <div class="hero-title">Sign In</div>
            <p class="hero-subtitle">
                Basic username/password authentication is enabled for this dashboard.
                After login, you can run multi-source job search and exports.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, center_col, _ = st.columns([1, 1.2, 1])
    with center_col:
        st.markdown('<div class="login-shell">', unsafe_allow_html=True)
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)

        if submitted:
            expected_username, expected_password = get_auth_credentials()
            is_valid_username = hmac.compare_digest(username.strip(), expected_username)
            is_valid_password = hmac.compare_digest(password, expected_password)
            if is_valid_username and is_valid_password:
                st.session_state["is_authenticated"] = True
                st.session_state["active_user"] = username.strip()
                st.session_state["auth_error"] = ""
                st.rerun()
            st.session_state["auth_error"] = "Invalid username or password."

        if st.session_state.get("auth_error"):
            st.error(st.session_state["auth_error"])

        st.caption("To change credentials, set APP_LOGIN_USERNAME and APP_LOGIN_PASSWORD.")
        st.markdown("</div>", unsafe_allow_html=True)


def render_workspace_header() -> None:
    safe_user = html.escape(str(st.session_state.get("active_user", "recruiter")))
    st.markdown(
        f"""
        <div class="hero-banner">
            <div class="hero-eyebrow">Talent Intelligence Workspace</div>
            <div class="hero-title">Professional Job Intelligence Dashboard</div>
            <p class="hero-subtitle">
                Aggregate jobs from multiple boards and career sites, monitor source health,
                and filter candidates quickly before exporting. Designed with a polished
                LinkedIn-inspired visual language.
            </p>
            <div class="hero-meta">
                <span class="hero-chip">Signed in: {safe_user}</span>
                <span class="hero-chip">Streamlit + Pandas</span>
                <span class="hero-chip">Multi-source search</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def prepare_dataframe(rows: list[dict[str, object]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "source",
                "keywords",
                "title",
                "company",
                "location",
                "listed_at",
                "listed_date",
                "job_id",
                "job_url",
                "apply_url",
                "employment_type",
                "salary",
                "is_remote",
                "description_snippet",
            ]
        )

    df = pd.DataFrame(rows)
    for column in (
        "source",
        "keywords",
        "title",
        "company",
        "location",
        "listed_at",
        "listed_date",
        "job_id",
        "job_url",
        "apply_url",
        "employment_type",
        "salary",
        "description_snippet",
    ):
        if column not in df.columns:
            df[column] = ""

    if "is_remote" not in df.columns:
        df["is_remote"] = False
    df["is_remote"] = df["is_remote"].fillna(False).astype(bool)

    df["listed_date_parsed"] = pd.to_datetime(df["listed_date"], errors="coerce", utc=True)
    df["search_blob"] = (
        df["title"].fillna("")
        + " "
        + df["company"].fillna("")
        + " "
        + df["location"].fillna("")
        + " "
        + df["salary"].fillna("")
        + " "
        + df["employment_type"].fillna("")
        + " "
        + df["description_snippet"].fillna("")
    ).str.lower()
    return df


def apply_filters(
    df: pd.DataFrame,
    *,
    text_query: str,
    source_filter: list[str],
    location_query: str,
    company_query: str,
    remote_only: bool,
    external_apply_only: bool,
    employment_types: list[str],
    dedupe: bool,
    date_start,
    date_end,
    sort_mode: str,
) -> pd.DataFrame:
    filtered = df.copy()

    if source_filter:
        filtered = filtered[filtered["source"].isin(source_filter)]

    if text_query.strip():
        filtered = filtered[
            filtered["search_blob"].str.contains(text_query.strip().lower(), na=False, regex=False)
        ]

    if location_query.strip():
        filtered = filtered[
            filtered["location"].fillna("").str.contains(location_query.strip(), case=False, na=False)
        ]

    if company_query.strip():
        filtered = filtered[
            filtered["company"].fillna("").str.contains(company_query.strip(), case=False, na=False)
        ]

    if remote_only:
        filtered = filtered[filtered["is_remote"] == True]  # noqa: E712

    if external_apply_only:
        filtered = filtered[
            ~filtered["apply_url"].fillna("").str.contains("linkedin.com/jobs/view", case=False, na=False)
        ]

    if employment_types:
        filtered = filtered[filtered["employment_type"].isin(employment_types)]

    if date_start:
        filtered = filtered[
            filtered["listed_date_parsed"].isna()
            | (filtered["listed_date_parsed"].dt.date >= date_start)
        ]
    if date_end:
        filtered = filtered[
            filtered["listed_date_parsed"].isna()
            | (filtered["listed_date_parsed"].dt.date <= date_end)
        ]

    if dedupe:
        filtered = filtered.sort_values(by=["listed_date_parsed"], ascending=False, na_position="last")
        filtered = filtered.drop_duplicates(
            subset=["source", "title", "company", "location", "apply_url"], keep="first"
        )

    if sort_mode == "Newest listed":
        filtered = filtered.sort_values(by=["listed_date_parsed", "title"], ascending=[False, True], na_position="last")
    elif sort_mode == "Oldest listed":
        filtered = filtered.sort_values(by=["listed_date_parsed", "title"], ascending=[True, True], na_position="last")
    elif sort_mode == "Company A-Z":
        filtered = filtered.sort_values(by=["company", "title"], ascending=[True, True], na_position="last")
    elif sort_mode == "Source A-Z":
        filtered = filtered.sort_values(by=["source", "listed_date_parsed"], ascending=[True, False], na_position="last")

    return filtered


def render_reports(report_rows: list[dict[str, object]]) -> None:
    st.subheader("Source Runs")
    if not report_rows:
        st.info("No runs yet.")
        return

    report_df = pd.DataFrame(report_rows)
    if report_df.empty:
        st.info("No run data captured.")
        return

    status_counts = report_df["status"].value_counts().to_dict()
    ok = int(status_counts.get("success", 0))
    blocked = int(status_counts.get("blocked", 0))
    errored = int(status_counts.get("error", 0))
    empty = int(status_counts.get("empty", 0))
    st.markdown(
        f"""
        <div class="metric-chip">
            Successful: <span class="mono">{ok}</span> |
            Empty: <span class="mono">{empty}</span> |
            Blocked: <span class="mono">{blocked}</span> |
            Errors: <span class="mono">{errored}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(report_df, use_container_width=True, hide_index=True)


def render_mobile_cards(filtered: pd.DataFrame, max_cards: int = 150) -> None:
    if filtered.empty:
        st.info("No rows match current filters.")
        return

    st.caption(f"Showing {min(len(filtered), max_cards)} of {len(filtered)} jobs")
    card_rows = filtered.head(max_cards).to_dict(orient="records")
    for row in card_rows:
        title = str(row.get("title", "")).strip() or "Job Opening"
        company = str(row.get("company", "")).strip() or "Unknown Company"
        source = str(row.get("source", "")).strip() or "Unknown Source"
        location = str(row.get("location", "")).strip() or "Unknown Location"
        listed = str(row.get("listed_date", "")).strip() or str(row.get("listed_at", "")).strip()
        description = str(row.get("description_snippet", "")).strip()
        apply_url = str(row.get("apply_url", "")).strip()
        job_url = str(row.get("job_url", "")).strip()

        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(f"{company} | {source}")
            st.caption(location)
            if listed:
                st.caption(f"Listed: {listed}")
            if description:
                st.caption(description)
            if apply_url:
                st.markdown(f"[Apply on company site]({apply_url})")
            if job_url and job_url != apply_url:
                st.markdown(f"[View job description]({job_url})")


def save_to_workspace(df: pd.DataFrame, path_text: str) -> tuple[bool, str]:
    try:
        target = Path(path_text).expanduser()
        if not target.is_absolute():
            target = Path.cwd() / target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(df.to_csv(index=False), encoding="utf-8")
        return True, str(target.resolve())
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return False, str(exc)


def main() -> None:
    st.set_page_config(
        page_title="Job Intelligence Dashboard",
        page_icon="J",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()
    initialize_session_state()

    if not st.session_state["is_authenticated"]:
        render_login_screen()
        st.stop()

    render_workspace_header()

    with st.sidebar:
        st.header("Search Controls")
        st.caption(f"Signed in as: {st.session_state.get('active_user', DEFAULT_LOGIN_USERNAME)}")
        if st.button("Log out", use_container_width=True):
            st.session_state["is_authenticated"] = False
            st.session_state["active_user"] = ""
            st.session_state["auth_error"] = ""
            st.rerun()

        st.markdown("---")
        keywords = st.text_input("Keywords", value="software engineer")
        location = st.text_input("Location", value="United States")
        limit_per_source = st.slider("Results per source", min_value=5, max_value=120, value=30, step=5)
        selected_sources = st.multiselect(
            "Sources",
            options=ALL_SOURCES,
            default=DEFAULT_UI_SOURCES,
            help="Indeed and Naukri may block non-browser automation depending on network/session.",
        )

        with st.expander("LinkedIn Options", expanded=False):
            linkedin_skip_apply = st.checkbox(
                "Skip LinkedIn detail-page apply extraction",
                value=True,
            )
            linkedin_delay = st.slider(
                "LinkedIn detail delay (seconds)",
                min_value=0.00,
                max_value=1.20,
                value=0.20,
                step=0.05,
            )

        with st.expander("Web Search (Company Careers) Options", expanded=False):
            web_result_pages = st.slider(
                "Search result pages",
                min_value=1,
                max_value=3,
                value=1,
                step=1,
            )
            web_max_sites = st.slider(
                "Websites to crawl",
                min_value=6,
                max_value=36,
                value=8,
                step=2,
            )
            web_follow_links_per_site = st.slider(
                "Career subpages per website",
                min_value=0,
                max_value=4,
                value=0,
                step=1,
            )
            web_links_per_site = st.slider(
                "Jobs/links per website",
                min_value=3,
                max_value=20,
                value=4,
                step=1,
            )

        mobile_mode = st.checkbox(
            "Mobile-friendly layout",
            value=True,
            help="Use compact filters and card-style results for smaller screens.",
        )

        run_search = st.button("Run Search", type="primary", use_container_width=True)

    if run_search:
        if not keywords.strip():
            st.error("Keywords are required.")
            st.stop()
        if not selected_sources:
            st.error("Select at least one source.")
            st.stop()

        config = SearchConfig(
            keywords=keywords.strip(),
            location=location.strip(),
            sources=selected_sources,
            limit_per_source=limit_per_source,
            linkedin_detail_delay=linkedin_delay,
            linkedin_skip_apply_url=linkedin_skip_apply,
            web_result_pages=web_result_pages,
            web_max_sites=web_max_sites,
            web_follow_links_per_site=web_follow_links_per_site,
            web_links_per_site=web_links_per_site,
        )

        progress_bar = st.progress(0.0, text="Initializing search...")
        status = st.status("Running providers...", expanded=True)
        events: list[str] = []

        def on_progress(step: int, total: int, message: str) -> None:
            ratio = 1.0 if total <= 0 else min(max(step / total, 0.0), 1.0)
            progress_bar.progress(ratio, text=message)
            status.write(message)
            events.append(message)

        jobs, reports = run_multi_source_search(config, progress_callback=on_progress)
        progress_bar.progress(1.0, text="Completed")
        status.update(label=f"Search complete: {len(jobs)} jobs aggregated", state="complete")

        st.session_state["jobs_rows"] = [job.to_dict() for job in jobs]
        st.session_state["report_rows"] = [report.to_dict() for report in reports]
        st.session_state["last_query"] = {
            "keywords": keywords,
            "location": location,
            "sources": selected_sources,
            "limit_per_source": limit_per_source,
            "events": events,
        }

    df = prepare_dataframe(st.session_state["jobs_rows"])
    report_rows = st.session_state["report_rows"]

    render_reports(report_rows)

    if df.empty:
        st.warning("No jobs loaded yet. Run a search from the sidebar.")
        st.stop()

    st.subheader("Slice and Dice")
    source_defaults = sorted(df["source"].dropna().unique())
    if mobile_mode:
        text_query = st.text_input("Text contains", value="")
        source_filter = st.multiselect("Source filter", options=source_defaults, default=source_defaults)
        location_query = st.text_input("Location contains", value="")
        company_query = st.text_input("Company contains", value="")
        remote_only = st.checkbox("Remote only", value=False)
        external_apply_only = st.checkbox("External apply links only", value=False)
        dedupe = st.checkbox("Dedupe similar postings", value=True)
        sort_mode = st.selectbox(
            "Sort",
            options=["Newest listed", "Oldest listed", "Company A-Z", "Source A-Z"],
            index=0,
        )
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            text_query = st.text_input("Text contains", value="")
        with c2:
            source_filter = st.multiselect("Source filter", options=source_defaults, default=source_defaults)
        with c3:
            location_query = st.text_input("Location contains", value="")
        with c4:
            company_query = st.text_input("Company contains", value="")

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            remote_only = st.checkbox("Remote only", value=False)
        with c6:
            external_apply_only = st.checkbox("External apply links only", value=False)
        with c7:
            dedupe = st.checkbox("Dedupe similar postings", value=True)
        with c8:
            sort_mode = st.selectbox(
                "Sort",
                options=["Newest listed", "Oldest listed", "Company A-Z", "Source A-Z"],
                index=0,
            )

    available_types = sorted(
        {
            value
            for value in df["employment_type"].fillna("").tolist()
            if str(value).strip()
        }
    )
    if mobile_mode:
        employment_types = st.multiselect(
            "Employment type",
            options=available_types,
            default=[],
        )
        date_candidates = df["listed_date_parsed"].dropna()
        if not date_candidates.empty:
            min_date = date_candidates.min().date()
            max_date = date_candidates.max().date()
            date_range = st.date_input("Listed between", value=(min_date, max_date), min_value=min_date, max_value=max_date)
            if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                date_start, date_end = date_range[0], date_range[1]
            elif isinstance(date_range, (list, tuple)) and len(date_range) == 1:
                date_start, date_end = date_range[0], date_range[0]
            else:
                date_start, date_end = date_range, date_range
        else:
            st.caption("No structured listing dates available in this result set.")
            date_start, date_end = None, None
    else:
        c9, c10 = st.columns(2)
        with c9:
            employment_types = st.multiselect(
                "Employment type",
                options=available_types,
                default=[],
            )
        with c10:
            date_candidates = df["listed_date_parsed"].dropna()
            if not date_candidates.empty:
                min_date = date_candidates.min().date()
                max_date = date_candidates.max().date()
                date_range = st.date_input("Listed between", value=(min_date, max_date), min_value=min_date, max_value=max_date)
                if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                    date_start, date_end = date_range[0], date_range[1]
                elif isinstance(date_range, (list, tuple)) and len(date_range) == 1:
                    date_start, date_end = date_range[0], date_range[0]
                else:
                    date_start, date_end = date_range, date_range
            else:
                st.caption("No structured listing dates available in this result set.")
                date_start, date_end = None, None

    filtered = apply_filters(
        df,
        text_query=text_query,
        source_filter=source_filter,
        location_query=location_query,
        company_query=company_query,
        remote_only=remote_only,
        external_apply_only=external_apply_only,
        employment_types=employment_types,
        dedupe=dedupe,
        date_start=date_start,
        date_end=date_end,
        sort_mode=sort_mode,
    )

    total = len(df)
    shown = len(filtered)
    companies = filtered["company"].nunique() if shown else 0
    sources = filtered["source"].nunique() if shown else 0
    metric_cols = st.columns(2) if mobile_mode else st.columns(4)
    metric_cols[0].metric("Filtered Jobs", shown)
    metric_cols[1].metric("Total Pulled", total)
    if mobile_mode:
        metric_cols = st.columns(2)
        metric_cols[0].metric("Companies", companies)
        metric_cols[1].metric("Sources", sources)
    else:
        metric_cols[2].metric("Companies", companies)
        metric_cols[3].metric("Sources", sources)

    table_tab, insights_tab, export_tab = st.tabs(["Jobs Table", "Insights", "Export"])

    with table_tab:
        if mobile_mode:
            result_view = st.radio(
                "Result view",
                options=["Mobile Cards", "Table"],
                horizontal=True,
            )
            if result_view == "Mobile Cards":
                render_mobile_cards(filtered)
            else:
                st.caption("Table view on mobile can require horizontal scrolling.")

        display_cols = [
            "source",
            "title",
            "company",
            "location",
            "employment_type",
            "salary",
            "listed_date",
            "description_snippet",
            "job_url",
            "apply_url",
        ]
        if (not mobile_mode) or (mobile_mode and result_view == "Table"):
            st.dataframe(
                filtered[display_cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "description_snippet": st.column_config.TextColumn("JD Snippet", width="large"),
                    "job_url": st.column_config.LinkColumn("Job URL"),
                    "apply_url": st.column_config.LinkColumn("Apply URL"),
                },
                height=560,
            )

    with insights_tab:
        if filtered.empty:
            st.info("No rows match current filters.")
        else:
            left, right = st.columns(2)
            with left:
                st.caption("Jobs by source")
                st.bar_chart(filtered["source"].value_counts())
            with right:
                st.caption("Top companies")
                st.bar_chart(filtered["company"].value_counts().head(12))
            st.caption("Top locations")
            st.bar_chart(filtered["location"].fillna("Unknown").value_counts().head(12))

    with export_tab:
        csv_bytes = filtered.drop(columns=["listed_date_parsed", "search_blob"], errors="ignore").to_csv(index=False).encode("utf-8")
        json_bytes = filtered.drop(columns=["listed_date_parsed", "search_blob"], errors="ignore").to_json(orient="records", indent=2).encode("utf-8")
        st.download_button(
            label="Download Filtered CSV",
            data=csv_bytes,
            file_name="jobs_filtered.csv",
            mime="text/csv",
        )
        st.download_button(
            label="Download Filtered JSON",
            data=json_bytes,
            file_name="jobs_filtered.json",
            mime="application/json",
        )

        st.markdown("---")
        out_path = st.text_input("Save CSV in workspace", value="jobs_filtered.csv")
        if st.button("Save CSV File"):
            ok, message = save_to_workspace(
                filtered.drop(columns=["listed_date_parsed", "search_blob"], errors="ignore"),
                out_path,
            )
            if ok:
                st.success(f"Saved: {message}")
            else:
                st.error(message)

    with st.expander("Last Run Event Log", expanded=False):
        for event in st.session_state["last_query"].get("events", []):
            st.write(event)


if __name__ == "__main__":
    main()
