import { render, screen, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { MemoryRouter } from 'react-router-dom'
import DashboardPage from './DashboardPage'

vi.mock('../api/endpoints', () => ({
  getHealth: vi.fn().mockResolvedValue({
    status: 'ok',
    version: '5.2.0',
    engines: { kokoro: true, 'edge-tts': true },
    uptime_seconds: 3600,
  }),
  getFiles: vi.fn().mockResolvedValue({
    files: [
      { id: '1', name: 'ch1.wav', size_mb: 1.0, mime_type: 'audio/wav', created_at: '2025-01-01T00:00:00', path: '/output/ch1.wav', download_url: '/output/ch1.wav' },
    ],
    total: 1,
  }),
  getJobs: vi.fn().mockResolvedValue([
    { job_id: 'j1', status: 'completed', progress: 100, phase: 'done', created_at: '2025-01-01T00:00:00', updated_at: '2025-01-01T00:01:00' },
    { job_id: 'j2', status: 'failed', progress: 50, phase: 'tts', error: 'err', created_at: '2025-01-02T00:00:00', updated_at: '2025-01-02T00:01:00' },
  ]),
}))

const renderPage = () =>
  render(
    <MemoryRouter>
      <DashboardPage />
    </MemoryRouter>
  )

describe('DashboardPage', () => {
  beforeEach(() => vi.clearAllMocks())

  it('renders the page title', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getByText('AudioReader Studio')).toBeInTheDocument()
    })
  })

  it('shows quick action buttons', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getByText('Texte rapide')).toBeInTheDocument()
    })
    expect(screen.getByText('Livre')).toBeInTheDocument()
  })

  it('fetches and displays stats', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getByText('Fichiers générés')).toBeInTheDocument()
    })
    expect(screen.getByText('Moteurs TTS')).toBeInTheDocument()
  })

  it('shows distribution section', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getByText('Distribution des jobs')).toBeInTheDocument()
    })
  })

  it('shows recent jobs', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getByText('Jobs récents')).toBeInTheDocument()
    })
  })
})
