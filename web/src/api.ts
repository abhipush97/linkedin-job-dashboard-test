import type {
  LoginResponse,
  SearchRequest,
  SearchResponse,
  SourcesResponse,
} from './types'

const API_BASE =
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, '') ?? ''

class ApiError extends Error {
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

const jsonHeaders = {
  'Content-Type': 'application/json',
}

async function readErrorMessage(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: string }
    return payload.detail ?? `Request failed (${response.status})`
  } catch {
    return `Request failed (${response.status})`
  }
}

async function request<T>(
  path: string,
  options: RequestInit = {},
  token?: string,
): Promise<T> {
  const headers = new Headers(options.headers ?? {})
  if (!headers.has('Content-Type') && options.body) {
    headers.set('Content-Type', 'application/json')
  }
  if (token) {
    headers.set('Authorization', `Bearer ${token}`)
  }

  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers,
  })

  if (!response.ok) {
    throw new ApiError(await readErrorMessage(response), response.status)
  }

  if (response.status === 204) {
    return undefined as T
  }

  return (await response.json()) as T
}

export async function login(username: string, password: string): Promise<LoginResponse> {
  return request<LoginResponse>('/api/auth/login', {
    method: 'POST',
    headers: jsonHeaders,
    body: JSON.stringify({ username, password }),
  })
}

export async function logout(token: string): Promise<void> {
  await request('/api/auth/logout', { method: 'POST' }, token)
}

export async function getCurrentUser(token: string): Promise<{ username: string }> {
  return request<{ username: string }>('/api/auth/me', {}, token)
}

export async function getSources(token: string): Promise<SourcesResponse> {
  return request<SourcesResponse>('/api/sources', {}, token)
}

export async function searchJobs(
  token: string,
  payload: SearchRequest,
): Promise<SearchResponse> {
  return request<SearchResponse>('/api/search', {
    method: 'POST',
    headers: jsonHeaders,
    body: JSON.stringify(payload),
  }, token)
}

export { ApiError, API_BASE }
