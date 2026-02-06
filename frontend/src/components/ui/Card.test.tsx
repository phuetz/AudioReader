import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import Card from './Card'

describe('Card', () => {
  it('renders children', () => {
    render(<Card>Hello</Card>)
    expect(screen.getByText('Hello')).toBeInTheDocument()
  })

  it('renders title when provided', () => {
    render(<Card title="My Card">Content</Card>)
    expect(screen.getByText('My Card')).toBeInTheDocument()
  })

  it('renders action when provided', () => {
    render(<Card title="Title" action={<button>Action</button>}>Body</Card>)
    expect(screen.getByText('Action')).toBeInTheDocument()
  })

  it('does not render header when no title or action', () => {
    const { container } = render(<Card>Just content</Card>)
    expect(container.querySelector('.border-b')).toBeNull()
  })
})
