import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect } from 'vitest'
import Input from './Input'

describe('Input', () => {
  it('renders with label', () => {
    render(<Input label="Nom" />)
    expect(screen.getByText('Nom')).toBeInTheDocument()
  })

  it('renders without label', () => {
    render(<Input placeholder="Entrez..." />)
    expect(screen.getByPlaceholderText('Entrez...')).toBeInTheDocument()
  })

  it('links label to input via id', () => {
    render(<Input label="Mon champ" />)
    const input = screen.getByLabelText('Mon champ')
    expect(input).toBeInTheDocument()
    expect(input.id).toBe('mon-champ')
  })

  it('accepts typed text', async () => {
    render(<Input label="Texte" />)
    const input = screen.getByLabelText('Texte')
    await userEvent.type(input, 'Bonjour')
    expect(input).toHaveValue('Bonjour')
  })

  it('uses custom id when provided', () => {
    render(<Input label="Test" id="custom-id" />)
    const input = screen.getByLabelText('Test')
    expect(input.id).toBe('custom-id')
  })
})
