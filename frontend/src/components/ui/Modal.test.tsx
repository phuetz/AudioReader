import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi } from 'vitest'
import Modal from './Modal'

describe('Modal', () => {
  it('renders nothing when closed', () => {
    const { container } = render(
      <Modal open={false} onClose={vi.fn()} title="Test">
        <p>Contenu</p>
      </Modal>
    )
    expect(container.innerHTML).toBe('')
  })

  it('renders title and content when open', () => {
    render(
      <Modal open={true} onClose={vi.fn()} title="Mon Modal">
        <p>Contenu modal</p>
      </Modal>
    )
    expect(screen.getByText('Mon Modal')).toBeInTheDocument()
    expect(screen.getByText('Contenu modal')).toBeInTheDocument()
  })

  it('calls onClose on close button click', async () => {
    const onClose = vi.fn()
    render(
      <Modal open={true} onClose={onClose} title="Test">
        <p>Contenu</p>
      </Modal>
    )
    const buttons = screen.getAllByRole('button')
    await userEvent.click(buttons[0])
    expect(onClose).toHaveBeenCalled()
  })
})
