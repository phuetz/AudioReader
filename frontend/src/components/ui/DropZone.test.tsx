import { render, screen } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import DropZone from './DropZone'

describe('DropZone', () => {
  it('renders default label', () => {
    render(<DropZone onFile={vi.fn()} />)
    expect(screen.getByText('Déposez un fichier ou cliquez')).toBeInTheDocument()
  })

  it('renders custom label', () => {
    render(<DropZone onFile={vi.fn()} label="Glissez ici" />)
    expect(screen.getByText('Glissez ici')).toBeInTheDocument()
  })

  it('shows accepted formats', () => {
    render(<DropZone onFile={vi.fn()} accept=".md,.epub,.pdf" />)
    expect(screen.getByText(/\.md.*\.epub.*\.pdf/)).toBeInTheDocument()
  })

  it('shows spinner when loading', () => {
    const { container } = render(<DropZone onFile={vi.fn()} loading={true} />)
    expect(container.querySelector('.animate-spin')).toBeInTheDocument()
  })
})
