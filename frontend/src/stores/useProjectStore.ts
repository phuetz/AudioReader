import { create } from 'zustand'
import type { ProjectInfo } from '../api/types'
import { getProjects, createProject as apiCreate, deleteProject as apiDelete } from '../api/endpoints'

interface ProjectState {
  projects: ProjectInfo[]
  loading: boolean
  fetchProjects: () => Promise<void>
  addProject: (name: string, description?: string) => Promise<ProjectInfo>
  removeProject: (id: string) => Promise<void>
}

export const useProjectStore = create<ProjectState>((set) => ({
  projects: [],
  loading: false,
  fetchProjects: async () => {
    set({ loading: true })
    try {
      const data = await getProjects()
      set({ projects: data.projects, loading: false })
    } catch {
      set({ loading: false })
    }
  },
  addProject: async (name, description) => {
    const project = await apiCreate(name, description)
    set((s) => ({ projects: [project, ...s.projects] }))
    return project
  },
  removeProject: async (id) => {
    await apiDelete(id)
    set((s) => ({ projects: s.projects.filter(p => p.id !== id) }))
  },
}))
