import { describe, it, expect, beforeEach } from 'vitest'
import { usePlaylistStore } from './usePlaylistStore'

describe('usePlaylistStore', () => {
  beforeEach(() => {
    usePlaylistStore.setState({ queue: [], currentIndex: -1, repeat: 'none', shuffle: false })
  })

  it('has correct initial state', () => {
    const state = usePlaylistStore.getState()
    expect(state.queue).toEqual([])
    expect(state.currentIndex).toBe(-1)
    expect(state.repeat).toBe('none')
    expect(state.shuffle).toBe(false)
  })

  it('adds a track and sets index to 0', () => {
    usePlaylistStore.getState().addTrack({ url: '/a.wav', title: 'Track A' })
    const state = usePlaylistStore.getState()
    expect(state.queue).toHaveLength(1)
    expect(state.queue[0].title).toBe('Track A')
    expect(state.currentIndex).toBe(0)
  })

  it('adds multiple tracks', () => {
    usePlaylistStore.getState().addTracks([
      { url: '/a.wav', title: 'A' },
      { url: '/b.wav', title: 'B' },
    ])
    expect(usePlaylistStore.getState().queue).toHaveLength(2)
    expect(usePlaylistStore.getState().currentIndex).toBe(0)
  })

  it('removes a track', () => {
    usePlaylistStore.getState().addTrack({ url: '/a.wav', title: 'A' })
    const id = usePlaylistStore.getState().queue[0].id
    usePlaylistStore.getState().removeTrack(id)
    expect(usePlaylistStore.getState().queue).toHaveLength(0)
    expect(usePlaylistStore.getState().currentIndex).toBe(-1)
  })

  it('clears queue', () => {
    usePlaylistStore.getState().addTracks([
      { url: '/a.wav', title: 'A' },
      { url: '/b.wav', title: 'B' },
    ])
    usePlaylistStore.getState().clearQueue()
    expect(usePlaylistStore.getState().queue).toEqual([])
    expect(usePlaylistStore.getState().currentIndex).toBe(-1)
  })

  it('navigates next', () => {
    usePlaylistStore.getState().addTracks([
      { url: '/a.wav', title: 'A' },
      { url: '/b.wav', title: 'B' },
    ])
    usePlaylistStore.getState().next()
    expect(usePlaylistStore.getState().currentIndex).toBe(1)
  })

  it('does not go past end without repeat', () => {
    usePlaylistStore.getState().addTrack({ url: '/a.wav', title: 'A' })
    usePlaylistStore.getState().next()
    expect(usePlaylistStore.getState().currentIndex).toBe(0)
  })

  it('wraps around with repeat all', () => {
    usePlaylistStore.getState().addTracks([
      { url: '/a.wav', title: 'A' },
      { url: '/b.wav', title: 'B' },
    ])
    usePlaylistStore.getState().setRepeat('all')
    usePlaylistStore.getState().setCurrentIndex(1)
    usePlaylistStore.getState().next()
    expect(usePlaylistStore.getState().currentIndex).toBe(0)
  })

  it('navigates previous', () => {
    usePlaylistStore.getState().addTracks([
      { url: '/a.wav', title: 'A' },
      { url: '/b.wav', title: 'B' },
    ])
    usePlaylistStore.getState().setCurrentIndex(1)
    usePlaylistStore.getState().previous()
    expect(usePlaylistStore.getState().currentIndex).toBe(0)
  })

  it('toggles shuffle', () => {
    usePlaylistStore.getState().toggleShuffle()
    expect(usePlaylistStore.getState().shuffle).toBe(true)
    usePlaylistStore.getState().toggleShuffle()
    expect(usePlaylistStore.getState().shuffle).toBe(false)
  })

  it('getCurrentTrack returns correct track', () => {
    usePlaylistStore.getState().addTrack({ url: '/a.wav', title: 'A' })
    const track = usePlaylistStore.getState().getCurrentTrack()
    expect(track?.title).toBe('A')
  })

  it('getCurrentTrack returns null when empty', () => {
    expect(usePlaylistStore.getState().getCurrentTrack()).toBeNull()
  })

  it('hasNext returns correct value', () => {
    expect(usePlaylistStore.getState().hasNext()).toBe(false)
    usePlaylistStore.getState().addTracks([
      { url: '/a.wav', title: 'A' },
      { url: '/b.wav', title: 'B' },
    ])
    expect(usePlaylistStore.getState().hasNext()).toBe(true)
  })

  it('hasPrevious returns correct value', () => {
    usePlaylistStore.getState().addTracks([
      { url: '/a.wav', title: 'A' },
      { url: '/b.wav', title: 'B' },
    ])
    expect(usePlaylistStore.getState().hasPrevious()).toBe(false)
    usePlaylistStore.getState().setCurrentIndex(1)
    expect(usePlaylistStore.getState().hasPrevious()).toBe(true)
  })
})
