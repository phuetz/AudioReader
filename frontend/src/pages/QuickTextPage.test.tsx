import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { MemoryRouter } from 'react-router-dom'
import QuickTextPage from './QuickTextPage'

vi.mock('../api/endpoints', () => ({
  generateAudio: vi.fn().mockResolvedValue({ job_id: 'test-job', status: 'pending' }),
  generatePreview: vi.fn().mockResolvedValue({ job_id: 'preview-job', status: 'pending' }),
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
      enableTimingHumanization: false,
      enableIntonationContours: false,
    }
    return typeof selector === 'function' ? selector(state) : state
  }),
}))

const renderPage = () =>
  render(
    <MemoryRouter>
      <QuickTextPage />
    </MemoryRouter>
  )

describe('QuickTextPage', () => {
  beforeEach(() => vi.clearAllMocks())

  it('renders page title', () => {
    renderPage()
    expect(screen.getByText('Texte rapide')).toBeInTheDocument()
  })

  it('renders text input', () => {
    renderPage()
    expect(screen.getByPlaceholderText('Entrez votre texte ici...')).toBeInTheDocument()
  })

  it('shows generation buttons', () => {
    renderPage()
    expect(screen.getByText('Générer')).toBeInTheDocument()
    expect(screen.getByText('Preview 30s')).toBeInTheDocument()
  })

  it('allows typing text', async () => {
    renderPage()
    const textarea = screen.getByPlaceholderText('Entrez votre texte ici...')
    await userEvent.type(textarea, 'Bonjour')
    expect(textarea).toHaveValue('Bonjour')
  })

  it('shows advanced options', () => {
    renderPage()
    expect(screen.getByText('Options avancées')).toBeInTheDocument()
  })
})
