import { create } from 'zustand'
import type { PodcastStatus } from '../api/types'
import { getPodcastStatus, startPodcast as apiStart, stopPodcast as apiStop } from '../api/endpoints'

interface PodcastState {
  status: PodcastStatus | null
  loading: boolean
  fetchStatus: () => Promise<void>
  start: (params?: { audio_dir?: string; port?: number; title?: string }) => Promise<void>
  stop: () => Promise<void>
}

export const usePodcastStore = create<PodcastState>((set) => ({
  status: null,
  loading: false,
  fetchStatus: async () => {
    try {
      const status = await getPodcastStatus()
      set({ status })
    } catch {
      set({ status: { running: false, episode_count: 0 } })
    }
  },
  start: async (params) => {
    set({ loading: true })
    try {
      await apiStart(params)
      const status = await getPodcastStatus()
      set({ status, loading: false })
    } catch {
      set({ loading: false })
    }
  },
  stop: async () => {
    set({ loading: true })
    try {
      await apiStop()
      set({ status: { running: false, episode_count: 0 }, loading: false })
    } catch {
      set({ loading: false })
    }
  },
}))
