import { useEffect, useCallback, useRef } from 'react'
import { subscribeToJob, type SSEJobEvent } from '../api/sse'
import { getJob } from '../api/endpoints'
import { useJobStore } from '../stores/useJobStore'

/**
 * Subscribe to a job's SSE stream with polling fallback.
 * Returns current job state from the store.
 */
export function useJob(jobId: string | null) {
  const { jobs, updateJob } = useJobStore()
  const unsubRef = useRef<(() => void) | null>(null)

  const handleEvent = useCallback((event: SSEJobEvent) => {
    updateJob({
      job_id: event.job_id,
      status: event.status as 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled',
      progress: event.progress,
      phase: event.phase,
      result: event.result,
      error: event.error,
      created_at: '',
      updated_at: '',
    })
  }, [updateJob])

  useEffect(() => {
    if (!jobId) return

    // Try SSE first
    const unsub = subscribeToJob(jobId, handleEvent, () => {
      // Fallback to polling on SSE error
      const interval = setInterval(async () => {
        try {
          const job = await getJob(jobId)
          updateJob(job)
          if (['completed', 'failed', 'cancelled'].includes(job.status)) {
            clearInterval(interval)
          }
        } catch {
          clearInterval(interval)
        }
      }, 2000)

      // Store cleanup for polling
      unsubRef.current = () => clearInterval(interval)
    })

    unsubRef.current = unsub

    return () => {
      unsubRef.current?.()
    }
  }, [jobId, handleEvent, updateJob])

  const job = jobs.find(j => j.job_id === jobId) || null
  return job
}
