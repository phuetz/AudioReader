import { describe, it, expect, beforeEach, vi } from 'vitest'
import { useProjectStore } from './useProjectStore'

vi.mock('../api/endpoints', () => ({
  getProjects: vi.fn().mockResolvedValue({
    projects: [
      { id: 'p1', name: 'Tome 1', description: 'Premier tome', created_at: '2025-01-01' },
    ],
  }),
  createProject: vi.fn().mockResolvedValue({
    id: 'p2', name: 'Tome 2', description: '', created_at: '2025-02-01',
  }),
  deleteProject: vi.fn().mockResolvedValue({}),
}))

describe('useProjectStore', () => {
  beforeEach(() => {
    useProjectStore.setState({ projects: [], loading: false })
  })

  it('has correct initial state', () => {
    const state = useProjectStore.getState()
    expect(state.projects).toEqual([])
    expect(state.loading).toBe(false)
  })

  it('fetches projects', async () => {
    await useProjectStore.getState().fetchProjects()
    const state = useProjectStore.getState()
    expect(state.projects).toHaveLength(1)
    expect(state.projects[0].name).toBe('Tome 1')
    expect(state.loading).toBe(false)
  })

  it('adds project', async () => {
    const project = await useProjectStore.getState().addProject('Tome 2')
    expect(project.id).toBe('p2')
    expect(useProjectStore.getState().projects).toHaveLength(1)
    expect(useProjectStore.getState().projects[0].name).toBe('Tome 2')
  })

  it('removes project', async () => {
    useProjectStore.setState({
      projects: [{ id: 'p1', name: 'Tome 1', description: '', created_at: '2025-01-01' }] as any,
    })
    await useProjectStore.getState().removeProject('p1')
    expect(useProjectStore.getState().projects).toHaveLength(0)
  })
})
