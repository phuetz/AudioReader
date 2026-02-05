/** TypeScript interfaces matching backend Pydantic models */

export interface VoiceInfo {
  id: string
  name: string
  gender: string
  language: string
  engine: string
  style: string
  preview_url?: string
}

export interface JobResponse {
  job_id: string
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled'
  progress: number
  phase?: string
  result?: Record<string, unknown>
  error?: string
  created_at: string
  updated_at: string
}

export interface JobCreatedResponse {
  job_id: string
  status: string
  message: string
}

export interface FileInfo {
  id: string
  name: string
  path: string
  size_mb: number
  mime_type: string
  created_at: string
  download_url: string
}

export interface UploadResponse {
  file_id: string
  original_name: string
  converted_path?: string
  mime_type: string
  chapters?: string[]
  text_preview?: string
}

export interface CharacterInfo {
  name: string
  gender: string
  dialogue_count: number
  suggested_voice?: string
}

export interface DialogueInfo {
  text: string
  speaker: string
  method: string
  confidence: number
}

export interface EmotionInfo {
  text: string
  emotion: string
  intensity: number
  intonation: string
}

export interface AnalysisResult {
  total_characters: number
  characters: CharacterInfo[]
  dialogues: DialogueInfo[]
  emotions: EmotionInfo[]
  chapter_count: number
  word_count: number
}

export interface ProjectInfo {
  id: string
  name: string
  description: string
  settings: Record<string, unknown>
  created_at: string
  updated_at: string
  files: string[]
}

export interface PodcastStatus {
  running: boolean
  url?: string
  port?: number
  qr_code_url?: string
  episode_count: number
}

export interface HealthResponse {
  status: string
  version: string
  engines: Record<string, boolean>
  uptime_seconds: number
}

export interface ConfigResponse {
  output_dir: string
  default_voice: string
  default_language: string
  features: Record<string, boolean>
  styles_available: string[]
  version: string
}

export type NarrationStyle = 'formal' | 'conversational' | 'dramatic' | 'storytelling' | 'documentary' | 'intimate' | 'energetic'

export type LLMProvider = 'ollama' | 'openai' | 'anthropic' | 'gemini'

export type SubtitleFormat = 'srt' | 'vtt' | 'json'

export interface CorrectionRule {
  id: string
  pattern: string
  replacement: string
  confidence: 'high' | 'medium' | 'low'
  notes: string
}

export interface CorrectionsList {
  corrections: CorrectionRule[]
  total: number
}

/** Extended audiobook generation options (v5.0) */
export interface AudiobookOptions {
  // Basic
  text?: string
  file_id?: string
  title?: string
  narrator_voice?: string
  style?: NarrationStyle
  enable_emotions?: boolean
  enable_multi_voice?: boolean
  language?: string
  enable_mastering?: boolean
  character_voices?: Record<string, string>

  // LLM Enhancement
  enable_llm_enhance?: boolean
  llm_provider?: LLMProvider
  llm_model?: string

  // Sound Effects
  enable_sound_effects?: boolean
  sound_effects_intensity?: number

  // Subtitles
  enable_subtitles?: boolean
  subtitle_format?: SubtitleFormat

  // Timing & Prosody
  enable_timing_humanization?: boolean
  pause_variation?: number
  enable_intonation_contours?: boolean

  // ACX Compliance
  enable_acx_compliance?: boolean
  acx_target_lufs?: number
}
