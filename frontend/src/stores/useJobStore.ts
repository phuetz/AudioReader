import { create } from 'zustand'
import type { JobResponse } from '../api/types'
import { getJobs } from '../api/endpoints'

interface JobState {
  jobs: JobResponse[]
  activeJobId: string | null
  loading: boolean
  setActiveJob: (id: string | null) => void
  fetchJobs: () => Promise<void>
  updateJob: (job: JobResponse) => void
}

export const useJobStore = create<JobState>((set, get) => ({
  jobs: [],
  activeJobId: null,
  loading: false,
  setActiveJob: (id) => set({ activeJobId: id }),
  fetchJobs: async () => {
    set({ loading: true })
    try {
      const data = await getJobs()
      set({ jobs: data, loading: false })
    } catch {
      set({ loading: false })
    }
  },
  updateJob: (job) => {
    const jobs = get().jobs
    const idx = jobs.findIndex(j => j.job_id === job.job_id)
    if (idx >= 0) {
      const updated = [...jobs]
      updated[idx] = job
      set({ jobs: updated })
    } else {
      set({ jobs: [job, ...jobs] })
    }
  },
}))
