import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import GenderBadge from './GenderBadge'

describe('GenderBadge', () => {
  it('renders F for feminine', () => {
    render(<GenderBadge gender="F" />)
    expect(screen.getByText('F')).toBeInTheDocument()
  })

  it('renders M for masculine', () => {
    render(<GenderBadge gender="M" />)
    expect(screen.getByText('M')).toBeInTheDocument()
  })

  it('renders ? for unknown gender', () => {
    render(<GenderBadge gender="X" />)
    expect(screen.getByText('?')).toBeInTheDocument()
  })
})
