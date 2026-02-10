import { describe, it, expect, beforeEach } from 'vitest'
import { useCharacterStore } from './useCharacterStore'

describe('useCharacterStore', () => {
  beforeEach(() => {
    useCharacterStore.setState({ characters: [], voiceAssignments: {} })
  })

  it('has correct initial state', () => {
    const state = useCharacterStore.getState()
    expect(state.characters).toEqual([])
    expect(state.voiceAssignments).toEqual({})
  })

  it('sets characters', () => {
    const chars = [
      { name: 'Marie', gender: 'F', dialogue_count: 5 },
      { name: 'Pierre', gender: 'M', dialogue_count: 3 },
    ]
    useCharacterStore.getState().setCharacters(chars as any)
    expect(useCharacterStore.getState().characters).toHaveLength(2)
    expect(useCharacterStore.getState().characters[0].name).toBe('Marie')
  })

  it('assigns voice to character', () => {
    useCharacterStore.getState().assignVoice('Marie', 'ff_siwis')
    expect(useCharacterStore.getState().voiceAssignments['Marie']).toBe('ff_siwis')
  })

  it('assigns multiple voices', () => {
    useCharacterStore.getState().assignVoice('Marie', 'ff_siwis')
    useCharacterStore.getState().assignVoice('Pierre', 'am_adam')
    const assignments = useCharacterStore.getState().voiceAssignments
    expect(assignments['Marie']).toBe('ff_siwis')
    expect(assignments['Pierre']).toBe('am_adam')
  })

  it('overwrites voice assignment', () => {
    useCharacterStore.getState().assignVoice('Marie', 'ff_siwis')
    useCharacterStore.getState().assignVoice('Marie', 'af_bella')
    expect(useCharacterStore.getState().voiceAssignments['Marie']).toBe('af_bella')
  })
})
