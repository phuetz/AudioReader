import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import ChapterList from './ChapterList'

describe('ChapterList', () => {
  it('renders nothing for empty chapters', () => {
    const { container } = render(<ChapterList chapters={[]} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders chapter count', () => {
    render(<ChapterList chapters={['Intro', 'Chapitre 1', 'Conclusion']} />)
    expect(screen.getByText(/\(3\)/)).toBeInTheDocument()
  })

  it('renders chapter titles', () => {
    render(<ChapterList chapters={['Prologue', 'Épilogue']} />)
    expect(screen.getByText('Prologue')).toBeInTheDocument()
    expect(screen.getByText('Épilogue')).toBeInTheDocument()
  })
})
