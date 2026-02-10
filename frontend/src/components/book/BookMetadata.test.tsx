import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import BookMetadata from './BookMetadata'

describe('BookMetadata', () => {
  it('renders word count', () => {
    render(<BookMetadata wordCount={15000} chapterCount={5} />)
    expect(screen.getByText(/15.*000/)).toBeInTheDocument()
  })

  it('renders chapter count', () => {
    render(<BookMetadata wordCount={100} chapterCount={5} />)
    expect(screen.getByText('5 chapitres')).toBeInTheDocument()
  })

  it('renders text preview when provided', () => {
    render(<BookMetadata wordCount={100} chapterCount={1} textPreview="Il était une fois..." />)
    expect(screen.getByText('Aperçu')).toBeInTheDocument()
    expect(screen.getByText('Il était une fois...')).toBeInTheDocument()
  })

  it('does not show preview when not provided', () => {
    render(<BookMetadata wordCount={100} chapterCount={1} />)
    expect(screen.queryByText('Aperçu')).not.toBeInTheDocument()
  })
})
