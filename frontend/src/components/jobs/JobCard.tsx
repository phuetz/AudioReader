import type { JobResponse } from '../../api/types'
import Badge from '../ui/Badge'
import Progress from '../ui/Progress'

export default function JobCard({ job }: { job: JobResponse }) {
  const statusColor = {
    pending: 'muted' as const,
    processing: 'cyan' as const,
    completed: 'green' as const,
    failed: 'red' as const,
    cancelled: 'muted' as const,
  }[job.status]

  return (
    <div className="p-3 bg-panel border border-border rounded-lg space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-mono text-muted">{job.job_id}</span>
        <Badge color={statusColor}>{job.status}</Badge>
      </div>
      {job.status === 'processing' && (
        <Progress value={job.progress} label={job.phase || ''} />
      )}
      {job.phase && job.status === 'processing' && (
        <p className="text-xs text-secondary">{job.phase}</p>
      )}
    </div>
  )
}
