import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import JobCard from './JobCard'

describe('JobCard', () => {
  it('renders job id and status', () => {
    render(
      <JobCard job={{ job_id: 'abc-123', status: 'completed', progress: 100, phase: '' }} />
    )
    expect(screen.getByText('abc-123')).toBeInTheDocument()
    expect(screen.getByText('completed')).toBeInTheDocument()
  })

  it('shows progress when processing', () => {
    render(
      <JobCard job={{ job_id: 'j1', status: 'processing', progress: 42, phase: 'Synthèse' }} />
    )
    expect(screen.getAllByText('42%').length).toBeGreaterThan(0)
    expect(screen.getAllByText('Synthèse').length).toBeGreaterThan(0)
  })

  it('shows pending status', () => {
    render(
      <JobCard job={{ job_id: 'j2', status: 'pending', progress: 0, phase: '' }} />
    )
    expect(screen.getByText('pending')).toBeInTheDocument()
  })

  it('shows failed status', () => {
    render(
      <JobCard job={{ job_id: 'j3', status: 'failed', progress: 0, phase: '' }} />
    )
    expect(screen.getByText('failed')).toBeInTheDocument()
  })
})
