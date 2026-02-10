import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import Badge from './Badge'

describe('Badge', () => {
  it('renders children text', () => {
    render(<Badge>Actif</Badge>)
    expect(screen.getByText('Actif')).toBeInTheDocument()
  })

  it('renders with default muted color', () => {
    const { container } = render(<Badge>Test</Badge>)
    expect(container.firstChild).toBeInTheDocument()
  })

  it('renders with accent color', () => {
    render(<Badge color="accent">Important</Badge>)
    expect(screen.getByText('Important')).toBeInTheDocument()
  })

  it('renders with green color', () => {
    render(<Badge color="green">Succès</Badge>)
    expect(screen.getByText('Succès')).toBeInTheDocument()
  })

  it('renders with red color', () => {
    render(<Badge color="red">Erreur</Badge>)
    expect(screen.getByText('Erreur')).toBeInTheDocument()
  })
})
