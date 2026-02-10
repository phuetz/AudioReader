import { describe, it, expect, beforeEach, vi } from 'vitest'
import { useVoiceStore } from './useVoiceStore'

vi.mock('../api/endpoints', () => ({
  getVoices: vi.fn().mockResolvedValue({
    voices: [
      { id: 'ff_siwis', name: 'Siwis', gender: 'F', language: 'fr', engine: 'kokoro', style: '' },
      { id: 'am_adam', name: 'Adam', gender: 'M', language: 'en', engine: 'kokoro', style: '' },
    ],
    total: 2,
  }),
}))

describe('useVoiceStore', () => {
  beforeEach(() => {
    useVoiceStore.setState({ voices: [], loading: false, selectedVoice: 'ff_siwis' })
  })

  it('has correct initial state', () => {
    const state = useVoiceStore.getState()
    expect(state.voices).toEqual([])
    expect(state.loading).toBe(false)
    expect(state.selectedVoice).toBe('ff_siwis')
  })

  it('sets selected voice', () => {
    useVoiceStore.getState().setSelectedVoice('am_adam')
    expect(useVoiceStore.getState().selectedVoice).toBe('am_adam')
  })

  it('fetches voices', async () => {
    await useVoiceStore.getState().fetchVoices()
    const state = useVoiceStore.getState()
    expect(state.voices).toHaveLength(2)
    expect(state.voices[0].id).toBe('ff_siwis')
    expect(state.loading).toBe(false)
  })

  it('sets loading during fetch', async () => {
    const promise = useVoiceStore.getState().fetchVoices()
    expect(useVoiceStore.getState().loading).toBe(true)
    await promise
    expect(useVoiceStore.getState().loading).toBe(false)
  })
})
