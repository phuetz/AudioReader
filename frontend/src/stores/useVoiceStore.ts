import { create } from 'zustand'
import type { VoiceInfo } from '../api/types'
import { getVoices } from '../api/endpoints'

interface VoiceState {
  voices: VoiceInfo[]
  loading: boolean
  selectedVoice: string
  setSelectedVoice: (id: string) => void
  fetchVoices: (language?: string) => Promise<void>
}

export const useVoiceStore = create<VoiceState>((set) => ({
  voices: [],
  loading: false,
  selectedVoice: 'ff_siwis',
  setSelectedVoice: (id) => set({ selectedVoice: id }),
  fetchVoices: async (language) => {
    set({ loading: true })
    try {
      const data = await getVoices(language)
      set({ voices: data.voices, loading: false })
    } catch {
      set({ loading: false })
    }
  },
}))
