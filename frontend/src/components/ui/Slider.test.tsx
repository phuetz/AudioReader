import { render, screen } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import Slider from './Slider'

describe('Slider', () => {
  it('renders with label and value', () => {
    render(<Slider label="Volume" value={75} onChange={vi.fn()} />)
    expect(screen.getByText('Volume')).toBeInTheDocument()
    expect(screen.getByText('75')).toBeInTheDocument()
  })

  it('renders range input', () => {
    render(<Slider value={50} onChange={vi.fn()} />)
    const input = screen.getByRole('slider')
    expect(input).toHaveValue('50')
  })

  it('shows unit', () => {
    render(<Slider label="Vitesse" value={1} onChange={vi.fn()} unit="x" />)
    expect(screen.getByText('1x')).toBeInTheDocument()
  })

  it('uses custom displayValue', () => {
    render(
      <Slider label="Test" value={50} onChange={vi.fn()} displayValue={(v) => `${v}%`} />
    )
    expect(screen.getByText('50%')).toBeInTheDocument()
  })
})
