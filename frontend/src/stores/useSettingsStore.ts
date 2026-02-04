import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { NarrationStyle } from '../api/types'

interface SettingsState {
  language: string
  defaultVoice: string
  speed: number
  style: NarrationStyle
  enableEmotions: boolean
  enableMultiVoice: boolean
  enableMastering: boolean
  setLanguage: (l: string) => void
  setDefaultVoice: (v: string) => void
  setSpeed: (s: number) => void
  setStyle: (s: NarrationStyle) => void
  setEnableEmotions: (e: boolean) => void
  setEnableMultiVoice: (m: boolean) => void
  setEnableMastering: (m: boolean) => void
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      language: 'fr',
      defaultVoice: 'ff_siwis',
      speed: 1.0,
      style: 'storytelling',
      enableEmotions: true,
      enableMultiVoice: true,
      enableMastering: false,
      setLanguage: (language) => set({ language }),
      setDefaultVoice: (defaultVoice) => set({ defaultVoice }),
      setSpeed: (speed) => set({ speed }),
      setStyle: (style) => set({ style }),
      setEnableEmotions: (enableEmotions) => set({ enableEmotions }),
      setEnableMultiVoice: (enableMultiVoice) => set({ enableMultiVoice }),
      setEnableMastering: (enableMastering) => set({ enableMastering }),
    }),
    { name: 'audioreader-settings' },
  ),
)
