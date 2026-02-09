import { render, screen, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { MemoryRouter } from 'react-router-dom'
import QueuePage from './QueuePage'

vi.mock('../api/client', () => ({
  default: {
    get: vi.fn().mockResolvedValue({
      data: {
        paused: false,
        processing: null,
        waiting: [],
        completed: [],
        failed: [],
        total: 0,
      },
    }),
    post: vi.fn().mockResolvedValue({ data: {} }),
    delete: vi.fn().mockResolvedValue({ data: {} }),
  },
}))

const renderPage = () =>
  render(
    <MemoryRouter>
      <QueuePage />
    </MemoryRouter>
  )

describe('QueuePage', () => {
  beforeEach(() => vi.clearAllMocks())

  it('renders the page title', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getByText("File d'attente")).toBeInTheDocument()
    })
  })

  it('shows add to queue form', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getByPlaceholderText('ID du fichier uploadé')).toBeInTheDocument()
    })
  })

  it('shows refresh button', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getByText('Actualiser')).toBeInTheDocument()
    })
  })

  it('shows empty queue state', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getByText('Queue vide')).toBeInTheDocument()
    })
  })
})
