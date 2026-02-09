import { render, screen } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { MemoryRouter } from 'react-router-dom'
import SettingsPage from './SettingsPage'

const mockSettings = {
  language: 'fr',
  defaultVoice: 'ff_siwis',
  speed: 1.0,
  narrationStyle: 'storytelling',
  enableEmotions: true,
  enableMultiVoice: false,
  enableMastering: true,
  enableTimingHumanization: false,
  pauseVariation: 15,
  enableIntonationContours: false,
  enableLLMEnhance: false,
  llmProvider: 'ollama',
  llmModel: '',
  enableSFX: false,
  sfxIntensity: 0.5,
  enableSubtitles: false,
  subtitleFormat: 'srt',
  enableACX: false,
  targetLUFS: -19,
  theme: 'dark',
  toggleTheme: vi.fn(),
  setLanguage: vi.fn(),
  setDefaultVoice: vi.fn(),
  setSpeed: vi.fn(),
  setNarrationStyle: vi.fn(),
  setEnableEmotions: vi.fn(),
  setEnableMultiVoice: vi.fn(),
  setEnableMastering: vi.fn(),
  setEnableTimingHumanization: vi.fn(),
  setPauseVariation: vi.fn(),
  setEnableIntonationContours: vi.fn(),
  setEnableLLMEnhance: vi.fn(),
  setLLMProvider: vi.fn(),
  setLLMModel: vi.fn(),
  setEnableSFX: vi.fn(),
  setSFXIntensity: vi.fn(),
  setEnableSubtitles: vi.fn(),
  setSubtitleFormat: vi.fn(),
  setEnableACX: vi.fn(),
  setTargetLUFS: vi.fn(),
}

vi.mock('../stores/useSettingsStore', () => ({
  useSettingsStore: vi.fn((selector) =>
    typeof selector === 'function' ? selector(mockSettings) : mockSettings
  ),
}))

vi.mock('../api/endpoints', () => ({
  getVoices: vi.fn().mockResolvedValue({
    voices: [{ id: 'ff_siwis', name: 'Siwis', gender: 'F', language: 'fr' }],
    total: 1,
  }),
}))

const renderPage = () =>
  render(
    <MemoryRouter>
      <SettingsPage />
    </MemoryRouter>
  )

describe('SettingsPage', () => {
  beforeEach(() => vi.clearAllMocks())

  it('renders the settings title', () => {
    renderPage()
    expect(screen.getByText('Paramètres')).toBeInTheDocument()
  })

  it('shows language & voice section', () => {
    renderPage()
    expect(screen.getByText('Langue et voix')).toBeInTheDocument()
  })

  it('shows audio settings', () => {
    renderPage()
    expect(screen.getByText('Audio')).toBeInTheDocument()
  })

  it('shows HQ pipeline toggles', () => {
    renderPage()
    expect(screen.getByText('Pipeline HQ')).toBeInTheDocument()
    expect(screen.getByText('Analyse des émotions')).toBeInTheDocument()
    expect(screen.getByText('Mastering audio')).toBeInTheDocument()
  })

  it('shows LLM section', () => {
    renderPage()
    expect(screen.getByText('Amélioration LLM')).toBeInTheDocument()
  })

  it('shows auto-save note', () => {
    renderPage()
    expect(screen.getByText(/sauvegardés automatiquement/)).toBeInTheDocument()
  })
})
