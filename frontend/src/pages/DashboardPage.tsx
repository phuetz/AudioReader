import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Type, BookOpen, Mic, FolderOpen, Headphones } from 'lucide-react'
import Card from '../components/ui/Card'
import Button from '../components/ui/Button'
import { getHealth, getFiles, getJobs } from '../api/endpoints'
import type { HealthResponse, FileInfo, JobResponse } from '../api/types'
import JobCard from '../components/jobs/JobCard'
import { ToastContainer } from '../components/ui/Toast'

export default function DashboardPage() {
  const navigate = useNavigate()
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [files, setFiles] = useState<FileInfo[]>([])
  const [recentJobs, setRecentJobs] = useState<JobResponse[]>([])

  useEffect(() => {
    getHealth().then(setHealth).catch(() => {})
    getFiles().then(d => setFiles(d.files)).catch(() => {})
    getJobs(undefined, 5).then(setRecentJobs).catch(() => {})
  }, [])

  const quickActions = [
    { icon: Type, label: 'Texte rapide', desc: 'Convertir du texte en audio', to: '/text' },
    { icon: BookOpen, label: 'Livre', desc: 'Convertir un livre complet', to: '/book' },
    { icon: Mic, label: 'Cloner une voix', desc: 'Depuis un fichier audio/vidéo', to: '/cloning' },
    { icon: FolderOpen, label: 'Fichiers', desc: 'Parcourir les fichiers générés', to: '/files' },
  ]

  return (
    <div className="space-y-6 max-w-6xl">
      {/* Header */}
      <div>
        <h1 className="text-xl font-semibold text-primary flex items-center gap-2">
          <Headphones className="w-5 h-5 text-accent" />
          AudioReader Studio
        </h1>
        <p className="text-sm text-secondary mt-1">Convertissez vos textes en audiobooks de qualité professionnelle</p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-surface border border-border rounded-xl p-4">
          <p className="text-xs text-muted">Fichiers générés</p>
          <p className="text-2xl font-semibold font-mono text-accent mt-1">{files.length}</p>
        </div>
        <div className="bg-surface border border-border rounded-xl p-4">
          <p className="text-xs text-muted">Moteurs TTS</p>
          <p className="text-2xl font-semibold font-mono text-cyan mt-1">
            {health ? Object.values(health.engines).filter(Boolean).length : 0}
          </p>
        </div>
        <div className="bg-surface border border-border rounded-xl p-4">
          <p className="text-xs text-muted">Jobs récents</p>
          <p className="text-2xl font-semibold font-mono text-green mt-1">{recentJobs.length}</p>
        </div>
        <div className="bg-surface border border-border rounded-xl p-4">
          <p className="text-xs text-muted">Uptime</p>
          <p className="text-2xl font-semibold font-mono text-primary mt-1">
            {health ? `${Math.floor(health.uptime_seconds / 60)}m` : '--'}
          </p>
        </div>
      </div>

      {/* Quick Actions */}
      <Card title="Actions rapides">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {quickActions.map(({ icon: Icon, label, desc, to }) => (
            <button
              key={to}
              onClick={() => navigate(to)}
              className="flex flex-col items-center gap-2 p-4 rounded-lg border border-border
                hover:border-accent/50 hover:bg-panel transition-colors text-center cursor-pointer"
            >
              <Icon className="w-6 h-6 text-accent" />
              <span className="text-sm font-medium text-primary">{label}</span>
              <span className="text-xs text-muted">{desc}</span>
            </button>
          ))}
        </div>
      </Card>

      {/* Recent jobs */}
      {recentJobs.length > 0 && (
        <Card title="Jobs récents" action={
          <Button variant="ghost" size="sm" onClick={() => navigate('/files')}>Voir tout</Button>
        }>
          <div className="space-y-2">
            {recentJobs.map(j => <JobCard key={j.job_id} job={j} />)}
          </div>
        </Card>
      )}

      <ToastContainer />
    </div>
  )
}
