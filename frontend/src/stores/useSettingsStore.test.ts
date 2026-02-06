import { describe, it, expect, beforeEach } from 'vitest'
import { useSettingsStore } from './useSettingsStore'

describe('useSettingsStore', () => {
  beforeEach(() => {
    // Reset store state
    useSettingsStore.setState({
      theme: 'dark',
      language: 'fr',
      defaultVoice: 'ff_siwis',
      speed: 1.0,
    })
  })

  it('has correct defaults', () => {
    const state = useSettingsStore.getState()
    expect(state.language).toBe('fr')
    expect(state.defaultVoice).toBe('ff_siwis')
    expect(state.speed).toBe(1.0)
    expect(state.theme).toBe('dark')
  })

  it('updates language', () => {
    useSettingsStore.getState().setLanguage('en')
    expect(useSettingsStore.getState().language).toBe('en')
  })

  it('toggles theme', () => {
    // Mock classList
    document.documentElement.classList.toggle = () => false
    useSettingsStore.getState().toggleTheme()
    expect(useSettingsStore.getState().theme).toBe('light')
    useSettingsStore.getState().toggleTheme()
    expect(useSettingsStore.getState().theme).toBe('dark')
  })

  it('updates speed', () => {
    useSettingsStore.getState().setSpeed(1.5)
    expect(useSettingsStore.getState().speed).toBe(1.5)
  })
})
