import { render, screen } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { MemoryRouter } from 'react-router-dom'
import PodcastPage from './PodcastPage'

vi.mock('../stores/usePodcastStore', () => ({
  usePodcastStore: vi.fn((selector) => {
    const state = {
      status: { running: false, port: 8080, episode_count: 0, rss_url: '', qr_code_url: '' },
      loading: false,
      fetchStatus: vi.fn(),
      start: vi.fn(),
      stop: vi.fn(),
    }
    return typeof selector === 'function' ? selector(state) : state
  }),
}))

vi.mock('../api/client', () => ({
  default: {
    get: vi.fn().mockResolvedValue({ data: { episodes: [] } }),
  },
}))

const renderPage = () =>
  render(
    <MemoryRouter>
      <PodcastPage />
    </MemoryRouter>
  )

describe('PodcastPage', () => {
  beforeEach(() => vi.clearAllMocks())

  it('renders the page title', () => {
    renderPage()
    expect(screen.getByText('Podcast RSS')).toBeInTheDocument()
  })

  it('shows start server button', () => {
    renderPage()
    expect(screen.getByText('Démarrer le serveur')).toBeInTheDocument()
  })

  it('shows instructions section', () => {
    renderPage()
    expect(screen.getByText('Instructions')).toBeInTheDocument()
  })

  it('shows compatible apps', () => {
    renderPage()
    expect(screen.getByText('Apple Podcasts')).toBeInTheDocument()
  })
})
