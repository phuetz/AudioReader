import { render, screen } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { MemoryRouter } from 'react-router-dom'
import BookConversionPage from './BookConversionPage'

vi.mock('../api/endpoints', () => ({
  generateAudiobook: vi.fn().mockResolvedValue({ job_id: 'book-job', status: 'pending' }),
  getVoices: vi.fn().mockResolvedValue({
    voices: [{ id: 'ff_siwis', name: 'Siwis', gender: 'F', language: 'fr' }],
    total: 1,
  }),
}))

vi.mock('../stores/useSettingsStore', () => ({
  useSettingsStore: vi.fn((selector) => {
    const state = {
      language: 'fr',
      defaultVoice: 'ff_siwis',
      speed: 1.0,
      narrationStyle: 'storytelling',
      enableEmotions: true,
      enableMultiVoice: false,
      enableMastering: true,
      enableLLMEnhance: false,
      llmProvider: 'ollama',
      enableSFX: false,
      sfxIntensity: 0.5,
      enableSubtitles: false,
      subtitleFormat: 'srt',
      enableTimingHumanization: false,
      pauseVariation: 15,
      enableIntonationContours: false,
      enableACX: false,
      targetLUFS: -19,
    }
    return typeof selector === 'function' ? selector(state) : state
  }),
}))

const renderPage = () =>
  render(
    <MemoryRouter>
      <BookConversionPage />
    </MemoryRouter>
  )

describe('BookConversionPage', () => {
  beforeEach(() => vi.clearAllMocks())

  it('renders the page title', () => {
    renderPage()
    expect(screen.getByText('Conversion livre')).toBeInTheDocument()
  })

  it('shows file drop zone', () => {
    renderPage()
    expect(screen.getByText(/Déposez un fichier/)).toBeInTheDocument()
  })

  it('shows generate button', () => {
    renderPage()
    expect(screen.getByText("Générer l'audiobook")).toBeInTheDocument()
  })

  it('shows pipeline toggles', () => {
    renderPage()
    expect(screen.getByText('Analyse émotions')).toBeInTheDocument()
    expect(screen.getByText('Multi-voix')).toBeInTheDocument()
    expect(screen.getByText('Mastering audio')).toBeInTheDocument()
  })
})
