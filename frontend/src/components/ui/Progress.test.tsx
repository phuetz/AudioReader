import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import Progress from './Progress'

describe('Progress', () => {
  it('renders with label and percentage', () => {
    render(<Progress value={75} label="Progression" />)
    expect(screen.getByText('Progression')).toBeInTheDocument()
    expect(screen.getByText('75%')).toBeInTheDocument()
  })

  it('renders without label', () => {
    const { container } = render(<Progress value={50} />)
    expect(container.firstChild).toBeInTheDocument()
  })

  it('shows 0% correctly', () => {
    render(<Progress value={0} label="Début" />)
    expect(screen.getByText('0%')).toBeInTheDocument()
  })

  it('shows 100% correctly', () => {
    render(<Progress value={100} label="Fini" />)
    expect(screen.getByText('100%')).toBeInTheDocument()
  })
})
