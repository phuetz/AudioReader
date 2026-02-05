import apiClient from './client'
import type {
  AnalysisResult,
  AudiobookOptions,
  ConfigResponse,
  CorrectionRule,
  CorrectionsList,
  FileInfo,
  HealthResponse,
  JobCreatedResponse,
  JobResponse,
  PodcastStatus,
  ProjectInfo,
  UploadResponse,
  VoiceInfo,
} from './types'

const V2 = '/api/v2'

// ── Health / Config ─────────────────────────────────────────────────────────

export const getHealth = () =>
  apiClient.get<HealthResponse>(`${V2}/health`).then(r => r.data)

export const getConfig = () =>
  apiClient.get<ConfigResponse>(`${V2}/config`).then(r => r.data)

// ── Voices ──────────────────────────────────────────────────────────────────

export const getVoices = (language?: string) =>
  apiClient.get<{ voices: VoiceInfo[]; total: number }>(`${V2}/voices`, { params: { language } }).then(r => r.data)

export const previewVoice = (voiceId: string, text?: string, speed?: number, language?: string) =>
  apiClient.post(`${V2}/voices/preview`, {
    voice_id: voiceId,
    text: text || 'Bonjour, je suis une voix de synthèse.',
    speed: speed || 1.0,
    language: language || 'fr',
  }, { responseType: 'blob' }).then(r => r.data as Blob)

export const cloneVoice = (formData: FormData) =>
  apiClient.post<{ voice_id: string; name: string; message: string }>(`${V2}/voices/clone`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  }).then(r => r.data)

export const getClonedVoices = () =>
  apiClient.get<{ voices: VoiceInfo[] }>(`${V2}/voices/cloned`).then(r => r.data)

export const deleteClonedVoice = (voiceId: string) =>
  apiClient.delete(`${V2}/voices/cloned/${voiceId}`).then(r => r.data)

// ── Generation ──────────────────────────────────────────────────────────────

export const generateAudio = (params: {
  text: string; voice?: string; speed?: number; language?: string; output_name?: string
}) =>
  apiClient.post<JobCreatedResponse>(`${V2}/generate`, params).then(r => r.data)

export const generateAudiobook = (params: AudiobookOptions) =>
  apiClient.post<JobCreatedResponse>(`${V2}/audiobook`, params).then(r => r.data)

export const generatePreview = (params: {
  text: string; voice?: string; speed?: number; language?: string; duration?: number
}) =>
  apiClient.post<JobCreatedResponse>(`${V2}/preview`, params).then(r => r.data)

// ── Jobs ────────────────────────────────────────────────────────────────────

export const getJobs = (status?: string, limit?: number) =>
  apiClient.get<JobResponse[]>(`${V2}/jobs`, { params: { status, limit } }).then(r => r.data)

export const getJob = (jobId: string) =>
  apiClient.get<JobResponse>(`${V2}/jobs/${jobId}`).then(r => r.data)

export const cancelJob = (jobId: string) =>
  apiClient.post(`${V2}/jobs/${jobId}/cancel`).then(r => r.data)

// ── Files ───────────────────────────────────────────────────────────────────

export const uploadFile = (file: File) => {
  const fd = new FormData()
  fd.append('file', file)
  return apiClient.post<UploadResponse>(`${V2}/files/upload`, fd, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 300_000,
  }).then(r => r.data)
}

export const getFiles = (extension?: string) =>
  apiClient.get<{ files: FileInfo[]; total: number }>(`${V2}/files`, { params: { extension } }).then(r => r.data)

// ── Analysis ────────────────────────────────────────────────────────────────

export const analyzeText = (params: { text?: string; file_id?: string; language?: string }) =>
  apiClient.post<AnalysisResult>(`${V2}/analyze`, params).then(r => r.data)

// ── Projects ────────────────────────────────────────────────────────────────

export const getProjects = () =>
  apiClient.get<{ projects: ProjectInfo[]; total: number }>(`${V2}/projects`).then(r => r.data)

export const createProject = (name: string, description?: string) =>
  apiClient.post<ProjectInfo>(`${V2}/projects`, { name, description }).then(r => r.data)

export const deleteProject = (id: string) =>
  apiClient.delete(`${V2}/projects/${id}`).then(r => r.data)

// ── Podcast ─────────────────────────────────────────────────────────────────

export const startPodcast = (params?: { audio_dir?: string; port?: number; title?: string }) =>
  apiClient.post(`${V2}/podcast/start`, params || {}).then(r => r.data)

export const stopPodcast = () =>
  apiClient.post(`${V2}/podcast/stop`).then(r => r.data)

export const getPodcastStatus = () =>
  apiClient.get<PodcastStatus>(`${V2}/podcast/status`).then(r => r.data)

// ── Corrections ──────────────────────────────────────────────────────────────

export const getCorrections = (search?: string) =>
  apiClient.get<CorrectionsList>(`${V2}/corrections`, { params: { search } }).then(r => r.data)

export const createCorrection = (params: {
  pattern: string; replacement: string; confidence?: string; notes?: string
}) =>
  apiClient.post<CorrectionRule>(`${V2}/corrections`, params).then(r => r.data)

export const updateCorrection = (id: string, params: {
  pattern?: string; replacement?: string; confidence?: string; notes?: string
}) =>
  apiClient.put<CorrectionRule>(`${V2}/corrections/${id}`, params).then(r => r.data)

export const deleteCorrection = (id: string) =>
  apiClient.delete(`${V2}/corrections/${id}`).then(r => r.data)

export const applyCorrections = (text: string, confidenceLevels?: string[]) =>
  apiClient.post<{ original: string; corrected: string; changes_count: number }>(
    `${V2}/corrections/apply`,
    { text, confidence_levels: confidenceLevels || ['high', 'medium'] }
  ).then(r => r.data)
