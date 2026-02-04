import { useEffect, useRef } from 'react'
import { useJob } from '../../hooks/useJob'
import Progress from '../ui/Progress'
import Badge from '../ui/Badge'

interface JobProgressProps {
  jobId: string
  onComplete?: (result: Record<string, unknown>) => void
}

export default function JobProgress({ jobId, onComplete }: JobProgressProps) {
  const job = useJob(jobId)
  const completedRef = useRef(false)

  useEffect(() => {
    if (job?.status === 'completed' && job.result && onComplete && !completedRef.current) {
      completedRef.current = true
      onComplete(job.result as Record<string, unknown>)
    }
  }, [job?.status, job?.result, onComplete])

  if (!job) return null

  const statusColor = {
    pending: 'muted' as const,
    processing: 'cyan' as const,
    completed: 'green' as const,
    failed: 'red' as const,
    cancelled: 'muted' as const,
  }[job.status]

  return (
    <div className="space-y-3 p-4 bg-panel border border-border rounded-xl">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-primary">Job {jobId}</span>
        <Badge color={statusColor}>{job.status}</Badge>
      </div>

      {job.status === 'processing' && (
        <Progress value={job.progress} label={job.phase || 'En cours...'} />
      )}

      {job.status === 'failed' && job.error && (
        <p className="text-xs text-red">{job.error}</p>
      )}

      {job.status === 'completed' && job.result && (
        <p className="text-xs text-green">
          Terminé — {String((job.result as Record<string, unknown>).duration_seconds ?? '')}s
        </p>
      )}
    </div>
  )
}
