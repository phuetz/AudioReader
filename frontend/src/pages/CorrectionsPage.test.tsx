import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { MemoryRouter } from 'react-router-dom'
import CorrectionsPage from './CorrectionsPage'

vi.mock('../api/endpoints', () => ({
  getCorrections: vi.fn().mockResolvedValue({
    corrections: [
      { id: 'c1', pattern: 'M.', replacement: 'Monsieur', confidence: 'high', notes: '' },
      { id: 'c2', pattern: 'Mme', replacement: 'Madame', confidence: 'high', notes: '' },
    ],
    total: 2,
  }),
  createCorrection: vi.fn().mockResolvedValue({ id: 'c3', pattern: 'Dr.', replacement: 'Docteur' }),
  updateCorrection: vi.fn().mockResolvedValue({}),
  deleteCorrection: vi.fn().mockResolvedValue({}),
  applyCorrections: vi.fn().mockResolvedValue({
    original: 'M. Dupont',
    corrected: 'Monsieur Dupont',
    changes_count: 1,
  }),
}))

const renderPage = () =>
  render(
    <MemoryRouter>
      <CorrectionsPage />
    </MemoryRouter>
  )

describe('CorrectionsPage', () => {
  beforeEach(() => vi.clearAllMocks())

  it('renders the page title', () => {
    renderPage()
    expect(screen.getByText('Corrections de prononciation')).toBeInTheDocument()
  })

  it('shows new correction button', () => {
    renderPage()
    expect(screen.getByText('Nouvelle correction')).toBeInTheDocument()
  })

  it('loads and displays correction patterns', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getByText('M.')).toBeInTheDocument()
    })
    expect(screen.getAllByText('Mme').length).toBeGreaterThan(0)
  })

  it('shows test panel', () => {
    renderPage()
    expect(screen.getByPlaceholderText(/tester les corrections/i)).toBeInTheDocument()
  })

  it('shows common corrections section', () => {
    renderPage()
    expect(screen.getByText('Corrections courantes')).toBeInTheDocument()
  })

  it('opens add form on click', async () => {
    renderPage()
    await userEvent.click(screen.getByText('Nouvelle correction'))
    expect(screen.getByPlaceholderText('ex: M.')).toBeInTheDocument()
  })
})
