import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi } from 'vitest'
import VoiceCard from './VoiceCard'

const voice = {
  id: 'ff_siwis',
  name: 'Siwis',
  engine: 'kokoro',
  gender: 'F',
  language: 'fr',
  style: 'neutral',
}

describe('VoiceCard', () => {
  it('renders voice name', () => {
    render(<VoiceCard voice={voice} />)
    expect(screen.getByText('Siwis')).toBeInTheDocument()
  })

  it('renders engine badge', () => {
    render(<VoiceCard voice={voice} />)
    expect(screen.getByText('kokoro')).toBeInTheDocument()
  })

  it('renders gender', () => {
    render(<VoiceCard voice={voice} />)
    expect(screen.getByText('Féminine')).toBeInTheDocument()
  })

  it('calls onSelect on card click', async () => {
    const onSelect = vi.fn()
    render(<VoiceCard voice={voice} onSelect={onSelect} />)
    await userEvent.click(screen.getByText('Siwis'))
    expect(onSelect).toHaveBeenCalled()
  })

  it('shows preview button when onPreview provided', () => {
    render(<VoiceCard voice={voice} onPreview={vi.fn()} />)
    expect(screen.getByText('Preview')).toBeInTheDocument()
  })

  it('calls onPreview on preview click', async () => {
    const onPreview = vi.fn()
    render(<VoiceCard voice={voice} onPreview={onPreview} />)
    await userEvent.click(screen.getByText('Preview'))
    expect(onPreview).toHaveBeenCalled()
  })
})
