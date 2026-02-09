import { render, screen, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { MemoryRouter } from 'react-router-dom'
import FilesPage from './FilesPage'

vi.mock('../api/endpoints', () => ({
  getFiles: vi.fn().mockResolvedValue({
    files: [
      { id: 'f1', name: 'chapter1.wav', size_mb: 5.0, mime_type: 'audio/wav', created_at: '2025-06-01T10:00:00', path: '/output/chapter1.wav', download_url: '/output/chapter1.wav' },
      { id: 'f2', name: 'chapter2.mp3', size_mb: 2.0, mime_type: 'audio/mpeg', created_at: '2025-06-02T10:00:00', path: '/output/chapter2.mp3', download_url: '/output/chapter2.mp3' },
    ],
    total: 2,
  }),
}))

vi.mock('../stores/usePlaylistStore', () => ({
  usePlaylistStore: vi.fn(() => ({
    items: [],
    addItem: vi.fn(),
    removeItem: vi.fn(),
  })),
}))

vi.mock('../stores/useAudioStore', () => ({
  useAudioStore: vi.fn(() => ({
    currentTrack: null,
    isPlaying: false,
    play: vi.fn(),
    pause: vi.fn(),
  })),
}))

const renderPage = () =>
  render(
    <MemoryRouter>
      <FilesPage />
    </MemoryRouter>
  )

describe('FilesPage', () => {
  beforeEach(() => vi.clearAllMocks())

  it('renders the page title', () => {
    renderPage()
    expect(screen.getByText('Fichiers générés')).toBeInTheDocument()
  })

  it('shows search input', () => {
    renderPage()
    expect(screen.getByPlaceholderText('Rechercher...')).toBeInTheDocument()
  })

  it('loads and displays files', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getByText('chapter1.wav')).toBeInTheDocument()
    })
    expect(screen.getByText('chapter2.mp3')).toBeInTheDocument()
  })

  it('shows file type filter', () => {
    renderPage()
    const selects = screen.getAllByRole('combobox')
    expect(selects.length).toBeGreaterThan(0)
  })
})
