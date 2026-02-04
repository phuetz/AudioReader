/** SSE client for job progress streaming */

export interface SSEJobEvent {
  job_id: string
  status: string
  progress: number
  phase?: string
  chapter_index?: number
  chapter_title?: string
  segments_done?: number
  segments_total?: number
  result?: Record<string, unknown>
  error?: string
}

type EventHandler = (event: SSEJobEvent) => void

export function subscribeToJob(
  jobId: string,
  onEvent: EventHandler,
  onError?: (err: Event) => void,
): () => void {
  const baseUrl = import.meta.env.VITE_API_URL || ''
  const url = `${baseUrl}/api/v2/jobs/${jobId}/events`
  const source = new EventSource(url)

  const handleEvent = (e: MessageEvent) => {
    try {
      const data: SSEJobEvent = JSON.parse(e.data)
      onEvent(data)
      // Auto-close on terminal states
      if (['completed', 'failed', 'cancelled'].includes(data.status)) {
        source.close()
      }
    } catch {
      // ignore parse errors
    }
  }

  // Subscribe to all event types
  for (const evt of ['job:pending', 'job:progress', 'job:completed', 'job:failed', 'job:cancelled']) {
    source.addEventListener(evt, handleEvent)
  }

  source.onerror = (e) => {
    onError?.(e)
    source.close()
  }

  return () => source.close()
}
