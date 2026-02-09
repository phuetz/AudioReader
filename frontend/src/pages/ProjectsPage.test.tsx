import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { MemoryRouter } from 'react-router-dom'
import ProjectsPage from './ProjectsPage'

vi.mock('../api/endpoints', () => ({
  getProjects: vi.fn().mockResolvedValue({
    projects: [
      { id: 'p1', name: 'Mon audiobook', description: 'Un livre test', created_at: '2025-01-01T00:00:00', file_count: 3 },
      { id: 'p2', name: 'Projet 2', description: '', created_at: '2025-02-01T00:00:00', file_count: 0 },
    ],
    total: 2,
  }),
  createProject: vi.fn().mockResolvedValue({ id: 'p3', name: 'Nouveau', created_at: '2025-03-01T00:00:00' }),
  deleteProject: vi.fn().mockResolvedValue({}),
}))

const renderPage = () =>
  render(
    <MemoryRouter>
      <ProjectsPage />
    </MemoryRouter>
  )

describe('ProjectsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders the page title', () => {
    renderPage()
    expect(screen.getByText('Projets')).toBeInTheDocument()
  })

  it('shows new project button', () => {
    renderPage()
    expect(screen.getByText('Nouveau projet')).toBeInTheDocument()
  })

  it('loads and displays projects', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getByText('Mon audiobook')).toBeInTheDocument()
    })
    expect(screen.getByText('Projet 2')).toBeInTheDocument()
  })

  it('shows search input', () => {
    renderPage()
    expect(screen.getByPlaceholderText('Rechercher un projet...')).toBeInTheDocument()
  })

  it('opens create form when clicking new project', async () => {
    renderPage()
    await userEvent.click(screen.getByText('Nouveau projet'))
    expect(screen.getByPlaceholderText('Mon audiobook')).toBeInTheDocument()
    expect(screen.getByText('Créer')).toBeInTheDocument()
    expect(screen.getByText('Annuler')).toBeInTheDocument()
  })
})
