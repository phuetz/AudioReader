import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { LLMProvider, NarrationStyle, SubtitleFormat } from '../api/types'

interface SettingsState {
  // Basic
  language: string
  defaultVoice: string
  speed: number
  style: NarrationStyle

  // Pipeline HQ
  enableEmotions: boolean
  enableMultiVoice: boolean
  enableMastering: boolean

  // LLM Enhancement (v5.0)
  enableLLMEnhance: boolean
  llmProvider: LLMProvider
  llmModel: string

  // Sound Effects (v5.0)
  enableSoundEffects: boolean
  soundEffectsIntensity: number

  // Subtitles (v5.0)
  enableSubtitles: boolean
  subtitleFormat: SubtitleFormat

  // Timing & Prosody (v5.0)
  enableTimingHumanization: boolean
  pauseVariation: number
  enableIntonationContours: boolean

  // ACX Compliance (v5.0)
  enableACXCompliance: boolean
  acxTargetLufs: number

  // Actions
  setLanguage: (l: string) => void
  setDefaultVoice: (v: string) => void
  setSpeed: (s: number) => void
  setStyle: (s: NarrationStyle) => void
  setEnableEmotions: (e: boolean) => void
  setEnableMultiVoice: (m: boolean) => void
  setEnableMastering: (m: boolean) => void
  setEnableLLMEnhance: (e: boolean) => void
  setLLMProvider: (p: LLMProvider) => void
  setLLMModel: (m: string) => void
  setEnableSoundEffects: (e: boolean) => void
  setSoundEffectsIntensity: (i: number) => void
  setEnableSubtitles: (e: boolean) => void
  setSubtitleFormat: (f: SubtitleFormat) => void
  setEnableTimingHumanization: (e: boolean) => void
  setPauseVariation: (v: number) => void
  setEnableIntonationContours: (e: boolean) => void
  setEnableACXCompliance: (e: boolean) => void
  setACXTargetLufs: (l: number) => void
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      // Basic defaults
      language: 'fr',
      defaultVoice: 'ff_siwis',
      speed: 1.0,
      style: 'storytelling',

      // Pipeline HQ defaults
      enableEmotions: true,
      enableMultiVoice: true,
      enableMastering: false,

      // LLM Enhancement defaults
      enableLLMEnhance: false,
      llmProvider: 'ollama',
      llmModel: '',

      // Sound Effects defaults
      enableSoundEffects: false,
      soundEffectsIntensity: 0.3,

      // Subtitles defaults
      enableSubtitles: false,
      subtitleFormat: 'srt',

      // Timing & Prosody defaults
      enableTimingHumanization: true,
      pauseVariation: 0.15,
      enableIntonationContours: true,

      // ACX Compliance defaults
      enableACXCompliance: false,
      acxTargetLufs: -19.0,

      // Actions
      setLanguage: (language) => set({ language }),
      setDefaultVoice: (defaultVoice) => set({ defaultVoice }),
      setSpeed: (speed) => set({ speed }),
      setStyle: (style) => set({ style }),
      setEnableEmotions: (enableEmotions) => set({ enableEmotions }),
      setEnableMultiVoice: (enableMultiVoice) => set({ enableMultiVoice }),
      setEnableMastering: (enableMastering) => set({ enableMastering }),
      setEnableLLMEnhance: (enableLLMEnhance) => set({ enableLLMEnhance }),
      setLLMProvider: (llmProvider) => set({ llmProvider }),
      setLLMModel: (llmModel) => set({ llmModel }),
      setEnableSoundEffects: (enableSoundEffects) => set({ enableSoundEffects }),
      setSoundEffectsIntensity: (soundEffectsIntensity) => set({ soundEffectsIntensity }),
      setEnableSubtitles: (enableSubtitles) => set({ enableSubtitles }),
      setSubtitleFormat: (subtitleFormat) => set({ subtitleFormat }),
      setEnableTimingHumanization: (enableTimingHumanization) => set({ enableTimingHumanization }),
      setPauseVariation: (pauseVariation) => set({ pauseVariation }),
      setEnableIntonationContours: (enableIntonationContours) => set({ enableIntonationContours }),
      setEnableACXCompliance: (enableACXCompliance) => set({ enableACXCompliance }),
      setACXTargetLufs: (acxTargetLufs) => set({ acxTargetLufs }),
    }),
    { name: 'audioreader-settings' },
  ),
)
