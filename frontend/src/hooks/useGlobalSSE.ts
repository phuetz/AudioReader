import { useEffect, useRef } from 'react'
import { useJobStore } from '../stores/useJobStore'

interface SSEJobEvent {
  job_id: string
  status: string
  progress: number
  phase?: string
  result?: Record<string, unknown>
  error?: string
}

type ToastFn = {
  success: (msg: string) => void
  error: (msg: string) => void
}

export function useGlobalSSE(toast?: ToastFn) {
  const updateJob = useJobStore(s => s.updateJob)
  const fetchJobs = useJobStore(s => s.fetchJobs)
  const esRef = useRef<EventSource | null>(null)

  useEffect(() => {
    const es = new EventSource('/api/v2/events')
    esRef.current = es

    const handleEvent = (eventType: string, data: SSEJobEvent) => {
      updateJob({
        job_id: data.job_id,
        status: data.status as 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled',
        progress: data.progress,
        phase: data.phase,
        result: data.result,
        error: data.error,
        created_at: '',
        updated_at: '',
      })

      if (toast) {
        if (eventType === 'job:completed') {
          toast.success(`Job ${data.job_id} terminé`)
        } else if (eventType === 'job:failed') {
          toast.error(`Job ${data.job_id} échoué: ${data.error || 'erreur inconnue'}`)
        }
      }
    }

    for (const evt of ['job:pending', 'job:progress', 'job:completed', 'job:failed', 'job:cancelled']) {
      es.addEventListener(evt, (e: MessageEvent) => {
        try {
          const data = JSON.parse(e.data) as SSEJobEvent
          handleEvent(evt, data)
        } catch { /* ignore */ }
      })
    }

    es.onerror = () => {
      // Reconnect handled by browser EventSource
    }

    // Initial fetch
    fetchJobs()

    return () => {
      es.close()
      esRef.current = null
    }
  }, [updateJob, fetchJobs, toast])
}
