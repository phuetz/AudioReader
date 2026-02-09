import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { MemoryRouter } from 'react-router-dom'
import CharactersPage from './CharactersPage'

vi.mock('../api/endpoints', () => ({
  analyzeText: vi.fn().mockResolvedValue({
    characters: [
      { name: 'Marie', gender: 'F', suggested_voice: 'ff_siwis', occurrences: 5 },
      { name: 'Pierre', gender: 'M', suggested_voice: 'am_adam', occurrences: 3 },
    ],
    dialogues: [
      { speaker: 'Marie', text: 'Bonjour Pierre !', start: 0, end: 20 },
    ],
    emotions: [
      { text: 'Bonjour Pierre', emotion: 'joy', intensity: 0.7 },
    ],
    word_count: 42,
    total_characters: 200,
    chapter_count: 1,
  }),
  getCharacterProfiles: vi.fn().mockResolvedValue({ profiles: [], total: 0 }),
  createCharacterProfile: vi.fn().mockResolvedValue({ id: 'cp1', name: 'Marie' }),
  deleteCharacterProfile: vi.fn().mockResolvedValue({}),
  getVoices: vi.fn().mockResolvedValue({
    voices: [{ id: 'ff_siwis', name: 'Siwis', gender: 'F', language: 'fr' }],
    total: 1,
  }),
}))

vi.mock('../stores/useCharacterStore', () => ({
  useCharacterStore: vi.fn(() => ({
    characters: [],
    setCharacters: vi.fn(),
  })),
}))

vi.mock('../stores/useVoiceStore', () => ({
  useVoiceStore: vi.fn(() => ({
    voices: [{ id: 'ff_siwis', name: 'Siwis', gender: 'F', language: 'fr' }],
    fetchVoices: vi.fn(),
  })),
}))

vi.mock('../stores/useSettingsStore', () => ({
  useSettingsStore: vi.fn((selector) => {
    const state = {
      language: 'fr',
      enableLLMEnhance: false,
      llmProvider: 'ollama',
    }
    return typeof selector === 'function' ? selector(state) : state
  }),
}))

const renderPage = () =>
  render(
    <MemoryRouter>
      <CharactersPage />
    </MemoryRouter>
  )

describe('CharactersPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders the page title', () => {
    renderPage()
    expect(screen.getByText('Personnages')).toBeInTheDocument()
  })

  it('shows detection and profiles tabs', () => {
    renderPage()
    expect(screen.getByText('Détection')).toBeInTheDocument()
    expect(screen.getByText('Profils sauvegardés')).toBeInTheDocument()
  })

  it('shows text analysis input', () => {
    renderPage()
    expect(screen.getByPlaceholderText(/Collez un extrait/)).toBeInTheDocument()
    expect(screen.getByText('Détecter les personnages')).toBeInTheDocument()
  })

  it('shows LLM validation toggle', () => {
    renderPage()
    expect(screen.getByText('Validation LLM')).toBeInTheDocument()
  })

  it('switches to profiles tab', async () => {
    renderPage()
    await userEvent.click(screen.getByText('Profils sauvegardés'))
    await waitFor(() => {
      expect(screen.getByText('Aucun profil sauvegardé')).toBeInTheDocument()
    })
  })

  it('shows import/export buttons on profiles tab', async () => {
    renderPage()
    await userEvent.click(screen.getByText('Profils sauvegardés'))
    expect(screen.getByText('Importer')).toBeInTheDocument()
    expect(screen.getByText('Exporter')).toBeInTheDocument()
  })
})
