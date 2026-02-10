import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi } from 'vitest'
import Select from './Select'

const options = [
  { value: 'fr', label: 'Français' },
  { value: 'en', label: 'English' },
  { value: 'es', label: 'Español' },
]

describe('Select', () => {
  it('renders with label', () => {
    render(<Select label="Langue" options={options} />)
    expect(screen.getByText('Langue')).toBeInTheDocument()
  })

  it('renders all options', () => {
    render(<Select label="Langue" options={options} />)
    expect(screen.getByText('Français')).toBeInTheDocument()
    expect(screen.getByText('English')).toBeInTheDocument()
    expect(screen.getByText('Español')).toBeInTheDocument()
  })

  it('links label to select via id', () => {
    render(<Select label="Langue" options={options} />)
    expect(screen.getByLabelText('Langue')).toBeInTheDocument()
  })

  it('handles change', async () => {
    const onChange = vi.fn()
    render(<Select label="Langue" options={options} onChange={onChange} />)
    await userEvent.selectOptions(screen.getByLabelText('Langue'), 'en')
    expect(onChange).toHaveBeenCalled()
  })
})
