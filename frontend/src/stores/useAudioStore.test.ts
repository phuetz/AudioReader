import { describe, it, expect, beforeEach } from 'vitest'
import { useAudioStore } from './useAudioStore'

describe('useAudioStore', () => {
  beforeEach(() => {
    useAudioStore.setState({
      currentUrl: null,
      isPlaying: false,
      currentTime: 0,
      duration: 0,
    })
  })

  it('has correct initial state', () => {
    const state = useAudioStore.getState()
    expect(state.currentUrl).toBeNull()
    expect(state.isPlaying).toBe(false)
    expect(state.currentTime).toBe(0)
    expect(state.duration).toBe(0)
  })

  it('sets current URL and resets time', () => {
    useAudioStore.getState().setCurrentTime(30)
    useAudioStore.getState().setCurrentUrl('/audio/test.wav')
    const state = useAudioStore.getState()
    expect(state.currentUrl).toBe('/audio/test.wav')
    expect(state.currentTime).toBe(0)
  })

  it('sets playing state', () => {
    useAudioStore.getState().setIsPlaying(true)
    expect(useAudioStore.getState().isPlaying).toBe(true)
    useAudioStore.getState().setIsPlaying(false)
    expect(useAudioStore.getState().isPlaying).toBe(false)
  })

  it('sets current time', () => {
    useAudioStore.getState().setCurrentTime(42.5)
    expect(useAudioStore.getState().currentTime).toBe(42.5)
  })

  it('sets duration', () => {
    useAudioStore.getState().setDuration(120)
    expect(useAudioStore.getState().duration).toBe(120)
  })
})
