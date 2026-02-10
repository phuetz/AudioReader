import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi } from 'vitest'
import FormatSelector from './FormatSelector'

describe('FormatSelector', () => {
  it('renders all format options', () => {
    render(<FormatSelector value="wav" onChange={vi.fn()} />)
    expect(screen.getByText('WAV')).toBeInTheDocument()
    expect(screen.getByText('MP3')).toBeInTheDocument()
    expect(screen.getByText('M4B')).toBeInTheDocument()
  })

  it('calls onChange when clicking a format', async () => {
    const onChange = vi.fn()
    render(<FormatSelector value="wav" onChange={onChange} />)
    await userEvent.click(screen.getByText('MP3'))
    expect(onChange).toHaveBeenCalledWith('mp3')
  })

  it('calls onChange for M4B', async () => {
    const onChange = vi.fn()
    render(<FormatSelector value="wav" onChange={onChange} />)
    await userEvent.click(screen.getByText('M4B'))
    expect(onChange).toHaveBeenCalledWith('m4b')
  })
})
