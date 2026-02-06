import { describe, it, expect, beforeEach, vi } from 'vitest'
import { useJobStore } from './useJobStore'

// Mock the endpoints module
vi.mock('../api/endpoints', () => ({
  getJobs: vi.fn().mockResolvedValue([]),
}))

describe('useJobStore', () => {
  beforeEach(() => {
    useJobStore.setState({ jobs: [], activeJobId: null, loading: false })
  })

  it('has empty initial state', () => {
    const state = useJobStore.getState()
    expect(state.jobs).toEqual([])
    expect(state.activeJobId).toBeNull()
    expect(state.loading).toBe(false)
  })

  it('sets active job', () => {
    useJobStore.getState().setActiveJob('abc123')
    expect(useJobStore.getState().activeJobId).toBe('abc123')
  })

  it('updates existing job', () => {
    const job = {
      job_id: 'j1',
      status: 'pending' as const,
      progress: 0,
      created_at: '',
      updated_at: '',
    }
    useJobStore.setState({ jobs: [job] })

    useJobStore.getState().updateJob({ ...job, status: 'completed', progress: 100 })
    expect(useJobStore.getState().jobs[0].status).toBe('completed')
    expect(useJobStore.getState().jobs[0].progress).toBe(100)
  })

  it('adds new job if not found', () => {
    const job = {
      job_id: 'new1',
      status: 'pending' as const,
      progress: 0,
      created_at: '',
      updated_at: '',
    }
    useJobStore.getState().updateJob(job)
    expect(useJobStore.getState().jobs).toHaveLength(1)
    expect(useJobStore.getState().jobs[0].job_id).toBe('new1')
  })
})
