import { describe, it, expect, beforeEach, vi } from 'vitest'
import { usePodcastStore } from './usePodcastStore'

vi.mock('../api/endpoints', () => ({
  getPodcastStatus: vi.fn().mockResolvedValue({ running: true, episode_count: 5, port: 8080 }),
  startPodcast: vi.fn().mockResolvedValue({}),
  stopPodcast: vi.fn().mockResolvedValue({}),
}))

describe('usePodcastStore', () => {
  beforeEach(() => {
    usePodcastStore.setState({ status: null, loading: false })
  })

  it('has correct initial state', () => {
    const state = usePodcastStore.getState()
    expect(state.status).toBeNull()
    expect(state.loading).toBe(false)
  })

  it('fetches status', async () => {
    await usePodcastStore.getState().fetchStatus()
    const state = usePodcastStore.getState()
    expect(state.status?.running).toBe(true)
    expect(state.status?.episode_count).toBe(5)
  })

  it('starts podcast', async () => {
    await usePodcastStore.getState().start()
    const state = usePodcastStore.getState()
    expect(state.loading).toBe(false)
    expect(state.status?.running).toBe(true)
  })

  it('stops podcast', async () => {
    await usePodcastStore.getState().stop()
    const state = usePodcastStore.getState()
    expect(state.loading).toBe(false)
    expect(state.status?.running).toBe(false)
    expect(state.status?.episode_count).toBe(0)
  })
})
