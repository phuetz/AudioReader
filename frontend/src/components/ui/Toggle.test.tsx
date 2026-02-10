import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi } from 'vitest'
import Toggle from './Toggle'

describe('Toggle', () => {
  it('renders label', () => {
    render(<Toggle label="Activer" checked={false} onChange={vi.fn()} />)
    expect(screen.getByText('Activer')).toBeInTheDocument()
  })

  it('renders description', () => {
    render(<Toggle label="Mode" checked={false} onChange={vi.fn()} description="Détails ici" />)
    expect(screen.getByText('Détails ici')).toBeInTheDocument()
  })

  it('has switch role with correct aria-checked', () => {
    render(<Toggle label="Test" checked={true} onChange={vi.fn()} />)
    const sw = screen.getByRole('switch')
    expect(sw).toHaveAttribute('aria-checked', 'true')
  })

  it('calls onChange on click', async () => {
    const onChange = vi.fn()
    render(<Toggle label="Test" checked={false} onChange={onChange} />)
    await userEvent.click(screen.getByRole('switch'))
    expect(onChange).toHaveBeenCalledWith(true)
  })

  it('calls onChange with false when checked', async () => {
    const onChange = vi.fn()
    render(<Toggle label="Test" checked={true} onChange={onChange} />)
    await userEvent.click(screen.getByRole('switch'))
    expect(onChange).toHaveBeenCalledWith(false)
  })
})
