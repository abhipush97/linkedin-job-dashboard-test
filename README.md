# Job Intelligence Dashboard (Mac)

This project now includes:

- a **Streamlit UI** to search across multiple job sources and analyze results
- the original **LinkedIn CLI exporter** (`linkedin_jobs_to_csv.py`)

## Best method for your use case

For local automation on a Mac, the most practical approach is:

1. Use public endpoints/APIs where available (stable sources)
2. Aggregate into one normalized table
3. Add interactive filters + exports in a local UI
4. Gracefully handle sources that block bot traffic (Indeed/Naukri can do this)

This is exactly how this implementation is structured.

## Setup (macOS)

```bash
cd /Users/abhishekpushkarjha/Downloads/linkedin
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the UI

```bash
streamlit run job_dashboard.py
```

The app provides:

- source selection (`LinkedIn`, `Indeed`, `Naukri`, `RemoteOK`, `Remotive`, `Arbeitnow`)
- progress/loading status for each provider
- per-source health report (`success`, `empty`, `blocked`, `error`)
- advanced filtering ("slice and dice"): text, source, company, location, date, remote, external apply links, employment type, dedupe, sorting
- charts and CSV/JSON download
- save filtered CSV directly to workspace

## Source notes

- `LinkedIn`: public guest jobs pages; includes apply-link extraction where available.
- `RemoteOK`, `Remotive`, `Arbeitnow`: public API-style sources.
- `Indeed`: often protected by anti-bot/captcha. The app surfaces blocked status cleanly.
- `Naukri`: tries API first; if captcha blocks API, the app falls back to indexed Naukri listing links.

## CLI still available (LinkedIn CSV)

Interactive:

```bash
python3 linkedin_jobs_to_csv.py
```

Flags:

```bash
python3 linkedin_jobs_to_csv.py \
  --keywords "machine learning engineer" \
  --location "United States" \
  --limit 30 \
  --output ml_jobs.csv
```

Useful CLI options:

- `--skip-apply-url`: skip detail-page apply extraction (faster, fewer requests).
- `--detail-delay 0.5`: increase delay between LinkedIn detail requests.
