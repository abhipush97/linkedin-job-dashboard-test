import { useEffect, useMemo, useState } from 'react'
import type { FormEvent } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { clsx } from 'clsx'
import {
  ArrowRight,
  Ban,
  Briefcase,
  Building2,
  CalendarDays,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  CircleAlert,
  Compass,
  ExternalLink,
  Gauge,
  Globe,
  ListChecks,
  LogOut,
  MapPin,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  User,
  XCircle,
} from 'lucide-react'
import { Link, Navigate, NavLink, Route, Routes, useNavigate } from 'react-router-dom'

import { ApiError, getCurrentUser, getSources, login, logout, searchJobs } from './api'
import type { Job, SearchRequest, SourceReport } from './types'
import './App.css'

const TOKEN_KEY = 'jobs_metasearch_token'
const PAGE_SIZE = 18

const initialSearchRequest: SearchRequest = {
  keywords: 'software engineer',
  location: 'United States',
  sources: [],
  limit_per_source: 30,
  linkedin_detail_delay: 0.2,
  linkedin_skip_apply_url: true,
  web_result_pages: 1,
  web_max_sites: 8,
  web_follow_links_per_site: 0,
  web_links_per_site: 4,
}

type SortMode = 'newest' | 'oldest' | 'company' | 'source'
type DateWindow = 'any' | '24h' | '7d' | '30d'

function parseListedDate(job: Job): number {
  const raw = job.listed_date || job.listed_at
  if (!raw) {
    return 0
  }
  const parsed = Date.parse(raw)
  return Number.isNaN(parsed) ? 0 : parsed
}

function matchesDateWindow(job: Job, window: DateWindow): boolean {
  if (window === 'any') {
    return true
  }

  const listedTs = parseListedDate(job)
  if (!listedTs) {
    return true
  }

  const now = Date.now()
  const oneDay = 24 * 60 * 60 * 1000
  if (window === '24h') {
    return now - listedTs <= oneDay
  }
  if (window === '7d') {
    return now - listedTs <= 7 * oneDay
  }
  return now - listedTs <= 30 * oneDay
}

function buildSearchBlob(job: Job): string {
  return [
    job.title,
    job.company,
    job.location,
    job.salary,
    job.employment_type,
    job.description_snippet,
  ]
    .join(' ')
    .toLowerCase()
}

function toggleSource(source: string, current: string[]): string[] {
  if (current.includes(source)) {
    return current.filter((value) => value !== source)
  }
  return [...current, source]
}

function LoginPage() {
  const [token, setToken] = useState<string>(() => localStorage.getItem(TOKEN_KEY) ?? '')
  const [loginForm, setLoginForm] = useState({ username: '', password: '' })

  const loginMutation = useMutation({
    mutationFn: ({ username, password }: { username: string; password: string }) =>
      login(username, password),
    onSuccess: (payload) => {
      localStorage.setItem(TOKEN_KEY, payload.access_token)
      setToken(payload.access_token)
      setLoginForm({ username: '', password: '' })
    },
  })

  const handleLogin = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    loginMutation.mutate({
      username: loginForm.username,
      password: loginForm.password,
    })
  }

  const loginError = loginMutation.error instanceof ApiError ? loginMutation.error.message : ''

  if (token) {
    return <Workspace token={token} onTokenCleared={() => setToken('')} />
  }

  return (
    <main className="login-page">
      <div className="login-glow" aria-hidden="true" />
      <section className="login-panel">
        <p className="eyebrow">Jobs Metasearch</p>
        <h1>Plan your next career move like a trip.</h1>
        <p className="muted-text">
          Search across multiple providers, compare quality and speed, and keep only the best opportunities.
        </p>

        <form onSubmit={handleLogin} className="login-form">
          <label>
            Username
            <input
              value={loginForm.username}
              onChange={(event) =>
                setLoginForm((prev) => ({ ...prev, username: event.target.value }))
              }
              autoComplete="username"
              placeholder="recruiter"
            />
          </label>
          <label>
            Password
            <input
              value={loginForm.password}
              onChange={(event) =>
                setLoginForm((prev) => ({ ...prev, password: event.target.value }))
              }
              type="password"
              autoComplete="current-password"
              placeholder="••••••••"
            />
          </label>
          <button type="submit" className="btn btn-primary" disabled={loginMutation.isPending}>
            {loginMutation.isPending ? 'Signing in...' : 'Sign In'}
            <ArrowRight size={16} />
          </button>
          {loginError ? <p className="form-error">{loginError}</p> : null}
        </form>

        <p className="helper-copy">Default local login: recruiter / linkedin123</p>
      </section>
    </main>
  )
}

type WorkspaceProps = {
  token: string
  onTokenCleared: () => void
}

function Workspace({ token, onTokenCleared }: WorkspaceProps) {
  const navigate = useNavigate()
  const [searchForm, setSearchForm] = useState<SearchRequest>(initialSearchRequest)

  const [textFilter, setTextFilter] = useState('')
  const [companyFilter, setCompanyFilter] = useState('')
  const [sourceFilter, setSourceFilter] = useState<string[]>([])
  const [remoteOnly, setRemoteOnly] = useState(false)
  const [externalOnly, setExternalOnly] = useState(false)
  const [dedupe, setDedupe] = useState(true)
  const [dateWindow, setDateWindow] = useState<DateWindow>('any')
  const [sortMode, setSortMode] = useState<SortMode>('newest')
  const [currentPage, setCurrentPage] = useState(1)

  const meQuery = useQuery({
    queryKey: ['me', token],
    queryFn: () => getCurrentUser(token),
    enabled: Boolean(token),
    retry: false,
  })

  const sourcesQuery = useQuery({
    queryKey: ['sources', token],
    queryFn: () => getSources(token),
    enabled: Boolean(token),
  })

  useEffect(() => {
    if (!sourcesQuery.data) {
      return
    }

    setSearchForm((prev) => {
      if (prev.sources.length > 0) {
        return prev
      }
      return { ...prev, sources: sourcesQuery.data.default_sources }
    })

    setSourceFilter((current) => {
      if (current.length > 0) {
        return current
      }
      return sourcesQuery.data.default_sources
    })
  }, [sourcesQuery.data])

  useEffect(() => {
    if (!meQuery.error) {
      return
    }

    localStorage.removeItem(TOKEN_KEY)
    onTokenCleared()
  }, [meQuery.error, onTokenCleared])

  const sourceOptions = useMemo(
    () => sourcesQuery.data?.sources ?? [],
    [sourcesQuery.data?.sources],
  )

  const searchMutation = useMutation({
    mutationFn: (payload: SearchRequest) => searchJobs(token, payload),
    onSuccess: () => {
      setCurrentPage(1)
      navigate('/results')
    },
  })

  const reports = useMemo(
    () => searchMutation.data?.reports ?? [],
    [searchMutation.data?.reports],
  )

  const filteredJobs = useMemo(() => {
    const jobs = searchMutation.data?.jobs ?? []
    const selectedSources = sourceFilter.length > 0 ? sourceFilter : sourceOptions

    let next = jobs.filter((job) => selectedSources.includes(job.source))

    if (textFilter.trim()) {
      const needle = textFilter.trim().toLowerCase()
      next = next.filter((job) => buildSearchBlob(job).includes(needle))
    }

    if (companyFilter.trim()) {
      const needle = companyFilter.trim().toLowerCase()
      next = next.filter((job) => job.company.toLowerCase().includes(needle))
    }

    if (remoteOnly) {
      next = next.filter((job) => job.is_remote)
    }

    if (externalOnly) {
      next = next.filter(
        (job) => !job.apply_url.toLowerCase().includes('linkedin.com/jobs/view'),
      )
    }

    next = next.filter((job) => matchesDateWindow(job, dateWindow))

    if (dedupe) {
      const seen = new Set<string>()
      next = next.filter((job) => {
        const key = [job.source, job.title, job.company, job.location, job.apply_url].join('|')
        if (seen.has(key)) {
          return false
        }
        seen.add(key)
        return true
      })
    }

    if (sortMode === 'newest') {
      next = [...next].sort((a, b) => parseListedDate(b) - parseListedDate(a))
    } else if (sortMode === 'oldest') {
      next = [...next].sort((a, b) => parseListedDate(a) - parseListedDate(b))
    } else if (sortMode === 'company') {
      next = [...next].sort((a, b) => a.company.localeCompare(b.company))
    } else {
      next = [...next].sort((a, b) => a.source.localeCompare(b.source))
    }

    return next
  }, [
    searchMutation.data?.jobs,
    sourceFilter,
    sourceOptions,
    textFilter,
    companyFilter,
    remoteOnly,
    externalOnly,
    dedupe,
    dateWindow,
    sortMode,
  ])

  useEffect(() => {
    setCurrentPage(1)
  }, [textFilter, companyFilter, sourceFilter, remoteOnly, externalOnly, dedupe, dateWindow, sortMode])

  const metrics = useMemo(() => {
    const jobs = filteredJobs
    const companies = new Set(jobs.map((job) => job.company).filter(Boolean)).size
    const sources = new Set(jobs.map((job) => job.source).filter(Boolean)).size
    const remoteRoles = jobs.filter((job) => job.is_remote).length

    return {
      filtered: jobs.length,
      total: searchMutation.data?.jobs.length ?? 0,
      companies,
      sources,
      remoteRoles,
    }
  }, [filteredJobs, searchMutation.data?.jobs.length])

  const reportStatusCounts = useMemo(() => {
    const result = { success: 0, empty: 0, blocked: 0, error: 0 }
    for (const report of reports) {
      result[report.status] += 1
    }
    return result
  }, [reports])

  const totalPages = Math.max(1, Math.ceil(filteredJobs.length / PAGE_SIZE))
  const safePage = Math.min(currentPage, totalPages)

  useEffect(() => {
    if (currentPage > totalPages) {
      setCurrentPage(totalPages)
    }
  }, [currentPage, totalPages])

  const paginatedJobs = useMemo(() => {
    const start = (safePage - 1) * PAGE_SIZE
    return filteredJobs.slice(start, start + PAGE_SIZE)
  }, [filteredJobs, safePage])

  const handleSearch = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()

    if (!searchForm.keywords.trim()) {
      return
    }

    const payload = {
      ...searchForm,
      sources:
        searchForm.sources.length > 0
          ? searchForm.sources
          : sourcesQuery.data?.default_sources ?? [],
    }

    searchMutation.mutate(payload)
  }

  const handleLogout = async () => {
    try {
      if (token) {
        await logout(token)
      }
    } catch {
      // no-op
    } finally {
      localStorage.removeItem(TOKEN_KEY)
      onTokenCleared()
      searchMutation.reset()
    }
  }

  return (
    <div className="app-shell">
      <header className="top-nav">
        <div className="brand-mark">JM</div>
        <div>
          <p className="eyebrow">Skyscanner for Jobs</p>
          <h2>Jobs Metasearch Workspace</h2>
        </div>
        <div className="top-nav-actions">
          <span className="auth-pill">
            <User size={14} />
            {meQuery.data?.username ?? 'Loading profile...'}
          </span>
          <button type="button" className="btn btn-ghost" onClick={handleLogout}>
            <LogOut size={14} />
            Log Out
          </button>
        </div>
      </header>

      <nav className="page-nav" aria-label="Workspace Sections">
        <NavLink
          to="/search"
          className={({ isActive }) => clsx('page-link', { 'page-link-active': isActive })}
        >
          <Compass size={16} />
          Search
        </NavLink>
        <NavLink
          to="/results"
          className={({ isActive }) => clsx('page-link', { 'page-link-active': isActive })}
        >
          <Briefcase size={16} />
          Results
        </NavLink>
        <NavLink
          to="/sources"
          className={({ isActive }) => clsx('page-link', { 'page-link-active': isActive })}
        >
          <ShieldCheck size={16} />
          Source Health
        </NavLink>
      </nav>

      <Routes>
        <Route path="/" element={<Navigate to="/search" replace />} />

        <Route
          path="/search"
          element={
            <section className="search-page-grid">
              <section className="search-hero">
                <div className="search-hero-copy">
                  <h1>Discover better roles faster.</h1>
                  <p>
                    One search runs across all selected providers. Then evaluate reliability, speed, and apply links before committing.
                  </p>
                </div>

                <form className="search-form" onSubmit={handleSearch}>
                  <div className="search-grid">
                    <label>
                      Job title or keywords
                      <div className="input-with-icon">
                        <Search size={16} />
                        <input
                          value={searchForm.keywords}
                          onChange={(event) =>
                            setSearchForm((prev) => ({ ...prev, keywords: event.target.value }))
                          }
                          placeholder="e.g. product manager"
                        />
                      </div>
                    </label>

                    <label>
                      Location
                      <div className="input-with-icon">
                        <MapPin size={16} />
                        <input
                          value={searchForm.location}
                          onChange={(event) =>
                            setSearchForm((prev) => ({ ...prev, location: event.target.value }))
                          }
                          placeholder="e.g. United States"
                        />
                      </div>
                    </label>

                    <label>
                      Roles per source
                      <div className="input-with-icon">
                        <Gauge size={16} />
                        <input
                          type="number"
                          min={5}
                          max={120}
                          value={searchForm.limit_per_source}
                          onChange={(event) =>
                            setSearchForm((prev) => ({
                              ...prev,
                              limit_per_source: Math.min(
                                120,
                                Math.max(5, Number(event.target.value) || 30),
                              ),
                            }))
                          }
                        />
                      </div>
                    </label>

                    <button
                      type="submit"
                      className="btn btn-primary"
                      disabled={searchMutation.isPending || sourcesQuery.isLoading}
                    >
                      <Sparkles size={15} />
                      {searchMutation.isPending ? 'Searching...' : 'Find Roles'}
                    </button>
                  </div>

                  <div className="chip-group">
                    {(sourceOptions.length ? sourceOptions : searchForm.sources).map((source) => (
                      <button
                        type="button"
                        key={source}
                        className={clsx('chip', {
                          'chip-active': searchForm.sources.includes(source),
                        })}
                        onClick={() =>
                          setSearchForm((prev) => ({
                            ...prev,
                            sources: toggleSource(source, prev.sources),
                          }))
                        }
                      >
                        <Globe size={13} />
                        {source}
                      </button>
                    ))}
                  </div>

                  <details className="advanced-card">
                    <summary>
                      <SlidersHorizontal size={15} />
                      Advanced source options
                    </summary>
                    <div className="advanced-grid">
                      <label>
                        LinkedIn detail delay (seconds)
                        <input
                          type="number"
                          step={0.05}
                          min={0}
                          max={1.5}
                          value={searchForm.linkedin_detail_delay}
                          onChange={(event) =>
                            setSearchForm((prev) => ({
                              ...prev,
                              linkedin_detail_delay: Math.min(
                                1.5,
                                Math.max(0, Number(event.target.value) || 0),
                              ),
                            }))
                          }
                        />
                      </label>
                      <label>
                        Web result pages
                        <input
                          type="number"
                          min={1}
                          max={3}
                          value={searchForm.web_result_pages}
                          onChange={(event) =>
                            setSearchForm((prev) => ({
                              ...prev,
                              web_result_pages: Math.min(
                                3,
                                Math.max(1, Number(event.target.value) || 1),
                              ),
                            }))
                          }
                        />
                      </label>
                      <label>
                        Max websites to crawl
                        <input
                          type="number"
                          min={4}
                          max={36}
                          value={searchForm.web_max_sites}
                          onChange={(event) =>
                            setSearchForm((prev) => ({
                              ...prev,
                              web_max_sites: Math.min(
                                36,
                                Math.max(4, Number(event.target.value) || 8),
                              ),
                            }))
                          }
                        />
                      </label>
                    </div>
                  </details>
                </form>
              </section>

              <aside className="panel quick-panel">
                <div className="panel-header">
                  <h3>Search Snapshot</h3>
                  <p>What happened in your latest run.</p>
                </div>

                <div className="status-metrics compact">
                  <span className="status-pill status-success">
                    <CheckCircle2 size={13} />
                    OK {reportStatusCounts.success}
                  </span>
                  <span className="status-pill status-empty">
                    <CircleAlert size={13} />
                    Empty {reportStatusCounts.empty}
                  </span>
                  <span className="status-pill status-blocked">
                    <Ban size={13} />
                    Blocked {reportStatusCounts.blocked}
                  </span>
                  <span className="status-pill status-error">
                    <XCircle size={13} />
                    Errors {reportStatusCounts.error}
                  </span>
                </div>

                <div className="mini-metrics">
                  <article>
                    <p>Total pulled</p>
                    <strong>{metrics.total}</strong>
                  </article>
                  <article>
                    <p>Unique companies</p>
                    <strong>{metrics.companies}</strong>
                  </article>
                  <article>
                    <p>Remote roles</p>
                    <strong>{metrics.remoteRoles}</strong>
                  </article>
                </div>

                <Link to="/results" className="btn btn-ghost full-width">
                  <ListChecks size={16} />
                  Go to Results
                </Link>
              </aside>
            </section>
          }
        />

        <Route
          path="/results"
          element={
            <section className="results-layout">
              <aside className="panel filter-panel">
                <div className="panel-header">
                  <h3>Refine Results</h3>
                  <p>Filters and sort controls.</p>
                </div>

                <label>
                  Search in results
                  <div className="input-with-icon">
                    <Search size={16} />
                    <input
                      value={textFilter}
                      onChange={(event) => setTextFilter(event.target.value)}
                      placeholder="title, company, skill"
                    />
                  </div>
                </label>

                <label>
                  Company
                  <div className="input-with-icon">
                    <Building2 size={16} />
                    <input
                      value={companyFilter}
                      onChange={(event) => setCompanyFilter(event.target.value)}
                      placeholder="e.g. Google"
                    />
                  </div>
                </label>

                <label>
                  Listed date
                  <select
                    value={dateWindow}
                    onChange={(event) => setDateWindow(event.target.value as DateWindow)}
                  >
                    <option value="any">Any time</option>
                    <option value="24h">Past 24 hours</option>
                    <option value="7d">Past 7 days</option>
                    <option value="30d">Past 30 days</option>
                  </select>
                </label>

                <label>
                  Sort by
                  <select
                    value={sortMode}
                    onChange={(event) => setSortMode(event.target.value as SortMode)}
                  >
                    <option value="newest">Newest first</option>
                    <option value="oldest">Oldest first</option>
                    <option value="company">Company A-Z</option>
                    <option value="source">Source A-Z</option>
                  </select>
                </label>

                <div className="toggle-grid">
                  <label className="check-row">
                    <input
                      type="checkbox"
                      checked={remoteOnly}
                      onChange={(event) => setRemoteOnly(event.target.checked)}
                    />
                    Remote only
                  </label>
                  <label className="check-row">
                    <input
                      type="checkbox"
                      checked={externalOnly}
                      onChange={(event) => setExternalOnly(event.target.checked)}
                    />
                    External apply links
                  </label>
                  <label className="check-row">
                    <input
                      type="checkbox"
                      checked={dedupe}
                      onChange={(event) => setDedupe(event.target.checked)}
                    />
                    Dedupe similar roles
                  </label>
                </div>

                <div className="chip-group compact">
                  {sourceOptions.map((source) => (
                    <button
                      key={source}
                      type="button"
                      className={clsx('chip', { 'chip-active': sourceFilter.includes(source) })}
                      onClick={() => setSourceFilter((current) => toggleSource(source, current))}
                    >
                      <Globe size={13} />
                      {source}
                    </button>
                  ))}
                </div>
              </aside>

              <main className="panel results-panel">
                <div className="panel-header sticky-header">
                  <h3>Best Matches</h3>
                  <p>
                    {searchMutation.isPending
                      ? 'Searching across providers...'
                      : `${filteredJobs.length} roles after filters`}
                  </p>
                </div>

                {searchMutation.isPending ? (
                  <div className="loading-grid">
                    {Array.from({ length: 6 }).map((_, idx) => (
                      <div key={idx} className="role-card skeleton" />
                    ))}
                  </div>
                ) : null}

                {!searchMutation.isPending && searchMutation.isSuccess && filteredJobs.length === 0 ? (
                  <div className="empty-state">
                    <h4>No roles matched these filters.</h4>
                    <p>Try broadening keywords or selecting more sources.</p>
                    <Link to="/search" className="btn btn-ghost">
                      <Compass size={15} />
                      Back to Search
                    </Link>
                  </div>
                ) : null}

                {!searchMutation.isPending && searchMutation.data?.jobs?.length === 0 ? (
                  <div className="empty-state">
                    <h4>No results yet.</h4>
                    <p>Run a search first, then come back here to refine and compare.</p>
                    <Link to="/search" className="btn btn-primary">
                      <Search size={15} />
                      Run Search
                    </Link>
                  </div>
                ) : null}

                {!searchMutation.isPending && paginatedJobs.length > 0 ? (
                  <>
                    <div className="role-list">
                      {paginatedJobs.map((job) => (
                        <article className="role-card" key={`${job.source}-${job.job_id}-${job.title}`}>
                          <div className="role-head">
                            <p className="role-source">{job.source}</p>
                            {job.is_remote ? <span className="badge badge-remote">Remote</span> : null}
                          </div>

                          <h4>{job.title}</h4>
                          <p className="role-company">{job.company || 'Unknown company'}</p>

                          <p className="role-meta">
                            <MapPin size={14} />
                            {job.location || 'Unknown location'}
                          </p>
                          {job.listed_date ? (
                            <p className="role-meta">
                              <CalendarDays size={14} />
                              Listed {job.listed_date}
                            </p>
                          ) : null}

                          {job.description_snippet ? (
                            <p className="role-snippet">{job.description_snippet}</p>
                          ) : null}

                          <div className="role-footer">
                            <a
                              className="btn btn-primary"
                              href={job.apply_url || job.job_url}
                              target="_blank"
                              rel="noreferrer"
                            >
                              Apply
                              <ArrowRight size={15} />
                            </a>
                            {job.job_url && job.job_url !== job.apply_url ? (
                              <a
                                className="btn btn-ghost"
                                href={job.job_url}
                                target="_blank"
                                rel="noreferrer"
                              >
                                <ExternalLink size={14} />
                                View JD
                              </a>
                            ) : null}
                          </div>
                        </article>
                      ))}
                    </div>

                    <div className="pagination-row">
                      <button
                        type="button"
                        className="btn btn-ghost"
                        onClick={() => setCurrentPage((value) => Math.max(1, value - 1))}
                        disabled={safePage <= 1}
                      >
                        <ChevronLeft size={15} />
                        Previous
                      </button>
                      <span>
                        Page <strong>{safePage}</strong> of <strong>{totalPages}</strong>
                      </span>
                      <button
                        type="button"
                        className="btn btn-ghost"
                        onClick={() => setCurrentPage((value) => Math.min(totalPages, value + 1))}
                        disabled={safePage >= totalPages}
                      >
                        Next
                        <ChevronRight size={15} />
                      </button>
                    </div>
                  </>
                ) : null}

                {searchMutation.error instanceof ApiError ? (
                  <p className="form-error">{searchMutation.error.message}</p>
                ) : null}
              </main>
            </section>
          }
        />

        <Route
          path="/sources"
          element={
            <section className="source-health-layout">
              <section className="panel source-metrics-panel">
                <div className="panel-header">
                  <h3>Source Health</h3>
                  <p>Provider reliability for the latest run.</p>
                </div>

                <div className="status-metrics">
                  <span className="status-pill status-success">
                    <CheckCircle2 size={13} />
                    OK {reportStatusCounts.success}
                  </span>
                  <span className="status-pill status-empty">
                    <CircleAlert size={13} />
                    Empty {reportStatusCounts.empty}
                  </span>
                  <span className="status-pill status-blocked">
                    <Ban size={13} />
                    Blocked {reportStatusCounts.blocked}
                  </span>
                  <span className="status-pill status-error">
                    <XCircle size={13} />
                    Errors {reportStatusCounts.error}
                  </span>
                </div>

                <div className="metrics-strip compact-grid">
                  <article>
                    <p>Filtered roles</p>
                    <strong>{metrics.filtered}</strong>
                  </article>
                  <article>
                    <p>Total pulled</p>
                    <strong>{metrics.total}</strong>
                  </article>
                  <article>
                    <p>Sources</p>
                    <strong>{metrics.sources}</strong>
                  </article>
                </div>
              </section>

              <section className="panel report-panel-wide">
                <div className="panel-header">
                  <h3>Run Reports</h3>
                  <p>Per-source response times and status.</p>
                </div>

                <div className="report-list">
                  {reports.length === 0 ? (
                    <p className="muted-text">No source runs yet.</p>
                  ) : null}

                  {reports.map((report: SourceReport) => (
                    <article key={report.source} className="report-item">
                      <div className="report-row">
                        <strong>{report.source}</strong>
                        <span className={clsx('status-dot', `status-${report.status}`)}>
                          {report.status}
                        </span>
                      </div>
                      <p>{report.message}</p>
                      <small>
                        {report.jobs_fetched} jobs · {report.elapsed_seconds.toFixed(2)}s
                      </small>
                    </article>
                  ))}
                </div>
              </section>

              <section className="panel events-panel">
                <div className="panel-header">
                  <h3>Event Timeline</h3>
                  <p>Debug trace from latest search request.</p>
                </div>

                <div className="events-feed">
                  {(searchMutation.data?.events ?? []).map((event, idx) => (
                    <p key={`${event}-${idx}`}>{event}</p>
                  ))}
                  {(searchMutation.data?.events ?? []).length === 0 ? (
                    <p className="muted-text">No events captured yet.</p>
                  ) : null}
                </div>
              </section>
            </section>
          }
        />

        <Route path="*" element={<Navigate to="/search" replace />} />
      </Routes>
    </div>
  )
}

function App() {
  return (
    <Routes>
      <Route path="*" element={<LoginPage />} />
    </Routes>
  )
}

export default App
