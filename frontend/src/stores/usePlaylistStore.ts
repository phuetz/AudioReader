import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export interface PlaylistTrack {
  id: string
  url: string
  title: string
  duration?: number
  metadata?: {
    chapter?: string
    book?: string
    voice?: string
  }
}

type RepeatMode = 'none' | 'one' | 'all'

interface PlaylistState {
  queue: PlaylistTrack[]
  currentIndex: number
  repeat: RepeatMode
  shuffle: boolean

  // Actions
  addTrack: (track: Omit<PlaylistTrack, 'id'>) => void
  addTracks: (tracks: Omit<PlaylistTrack, 'id'>[]) => void
  removeTrack: (id: string) => void
  clearQueue: () => void

  // Playback control
  setCurrentIndex: (index: number) => void
  next: () => void
  previous: () => void

  // Settings
  setRepeat: (mode: RepeatMode) => void
  toggleShuffle: () => void

  // Getters
  getCurrentTrack: () => PlaylistTrack | null
  hasNext: () => boolean
  hasPrevious: () => boolean
}

let trackIdCounter = 0
const generateTrackId = () => `track_${Date.now()}_${++trackIdCounter}`

export const usePlaylistStore = create<PlaylistState>()(
  persist(
    (set, get) => ({
      queue: [],
      currentIndex: -1,
      repeat: 'none',
      shuffle: false,

      addTrack: (track) => {
        const newTrack: PlaylistTrack = {
          ...track,
          id: generateTrackId(),
        }
        set((state) => ({
          queue: [...state.queue, newTrack],
          currentIndex: state.currentIndex === -1 ? 0 : state.currentIndex,
        }))
      },

      addTracks: (tracks) => {
        const newTracks = tracks.map((track) => ({
          ...track,
          id: generateTrackId(),
        }))
        set((state) => ({
          queue: [...state.queue, ...newTracks],
          currentIndex: state.currentIndex === -1 ? 0 : state.currentIndex,
        }))
      },

      removeTrack: (id) => {
        set((state) => {
          const index = state.queue.findIndex((t) => t.id === id)
          if (index === -1) return state

          const newQueue = state.queue.filter((t) => t.id !== id)
          let newIndex = state.currentIndex

          // Adjust current index if needed
          if (index < state.currentIndex) {
            newIndex = state.currentIndex - 1
          } else if (index === state.currentIndex) {
            newIndex = Math.min(state.currentIndex, newQueue.length - 1)
          }

          return {
            queue: newQueue,
            currentIndex: newQueue.length === 0 ? -1 : newIndex,
          }
        })
      },

      clearQueue: () => set({ queue: [], currentIndex: -1 }),

      setCurrentIndex: (index) => {
        const { queue } = get()
        if (index >= 0 && index < queue.length) {
          set({ currentIndex: index })
        }
      },

      next: () => {
        const { queue, currentIndex, repeat, shuffle } = get()
        if (queue.length === 0) return

        let nextIndex: number

        if (shuffle) {
          // Random track (different from current if possible)
          if (queue.length === 1) {
            nextIndex = 0
          } else {
            do {
              nextIndex = Math.floor(Math.random() * queue.length)
            } while (nextIndex === currentIndex)
          }
        } else if (currentIndex >= queue.length - 1) {
          // At end of queue
          if (repeat === 'all') {
            nextIndex = 0
          } else {
            return // Don't advance
          }
        } else {
          nextIndex = currentIndex + 1
        }

        set({ currentIndex: nextIndex })
      },

      previous: () => {
        const { queue, currentIndex, repeat } = get()
        if (queue.length === 0) return

        let prevIndex: number

        if (currentIndex <= 0) {
          if (repeat === 'all') {
            prevIndex = queue.length - 1
          } else {
            prevIndex = 0 // Stay at start
          }
        } else {
          prevIndex = currentIndex - 1
        }

        set({ currentIndex: prevIndex })
      },

      setRepeat: (mode) => set({ repeat: mode }),

      toggleShuffle: () => set((state) => ({ shuffle: !state.shuffle })),

      getCurrentTrack: () => {
        const { queue, currentIndex } = get()
        if (currentIndex >= 0 && currentIndex < queue.length) {
          return queue[currentIndex]
        }
        return null
      },

      hasNext: () => {
        const { queue, currentIndex, repeat } = get()
        if (queue.length === 0) return false
        if (repeat === 'all') return true
        return currentIndex < queue.length - 1
      },

      hasPrevious: () => {
        const { queue, currentIndex, repeat } = get()
        if (queue.length === 0) return false
        if (repeat === 'all') return true
        return currentIndex > 0
      },
    }),
    {
      name: 'audioreader-playlist',
      partialize: (state) => ({
        queue: state.queue,
        currentIndex: state.currentIndex,
        repeat: state.repeat,
        shuffle: state.shuffle,
      }),
    }
  )
)
