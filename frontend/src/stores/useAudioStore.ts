import { create } from 'zustand'

interface AudioState {
  currentUrl: string | null
  isPlaying: boolean
  currentTime: number
  duration: number
  setCurrentUrl: (url: string | null) => void
  setIsPlaying: (p: boolean) => void
  setCurrentTime: (t: number) => void
  setDuration: (d: number) => void
}

export const useAudioStore = create<AudioState>((set) => ({
  currentUrl: null,
  isPlaying: false,
  currentTime: 0,
  duration: 0,
  setCurrentUrl: (currentUrl) => set({ currentUrl, currentTime: 0 }),
  setIsPlaying: (isPlaying) => set({ isPlaying }),
  setCurrentTime: (currentTime) => set({ currentTime }),
  setDuration: (duration) => set({ duration }),
}))
