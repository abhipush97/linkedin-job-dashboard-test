export type SourceStatus = 'success' | 'empty' | 'blocked' | 'error'

export type Job = {
  source: string
  keywords: string
  title: string
  company: string
  location: string
  listed_at: string
  listed_date: string
  job_id: string
  job_url: string
  apply_url: string
  employment_type: string
  salary: string
  is_remote: boolean
  description_snippet: string
}

export type SourceReport = {
  source: string
  status: SourceStatus
  jobs_fetched: number
  elapsed_seconds: number
  message: string
}

export type SearchRequest = {
  keywords: string
  location: string
  sources: string[]
  limit_per_source: number
  linkedin_detail_delay: number
  linkedin_skip_apply_url: boolean
  web_result_pages: number
  web_max_sites: number
  web_follow_links_per_site: number
  web_links_per_site: number
}

export type SearchResponse = {
  jobs: Job[]
  reports: SourceReport[]
  events: string[]
}

export type LoginResponse = {
  access_token: string
  token_type: 'Bearer'
  expires_in: number
  username: string
}

export type SourcesResponse = {
  sources: string[]
  default_sources: string[]
  user: string
}
