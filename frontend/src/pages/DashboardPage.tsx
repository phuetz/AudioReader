import { useEffect, useState, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { Type, BookOpen, Mic, FolderOpen, FolderKanban, Headphones, Shield, Users, Radio } from 'lucide-react'
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
    getJobs(undefined, 20).then(setRecentJobs).catch(() => {})
  }, [])

  const quickActions = [
    { icon: Type, label: 'Texte rapide', desc: 'Convertir du texte en audio', to: '/text' },
    { icon: BookOpen, label: 'Livre', desc: 'Convertir un livre complet', to: '/book' },
    { icon: Users, label: 'Personnages', desc: 'Détecter et assigner les voix', to: '/characters' },
    { icon: Mic, label: 'Cloner une voix', desc: 'Depuis un fichier audio/vidéo', to: '/cloning' },
    { icon: Shield, label: 'Analyse ACX', desc: 'Vérifier la conformité Audible', to: '/acx' },
    { icon: FolderKanban, label: 'Projets', desc: 'Organiser vos audiobooks', to: '/projects' },
    { icon: Radio, label: 'Podcast', desc: 'Serveur RSS local', to: '/podcast' },
    { icon: FolderOpen, label: 'Fichiers', desc: 'Parcourir les fichiers générés', to: '/files' },
  ]

  // Status distribution
  const statusCounts = useMemo(() => {
    const counts = { completed: 0, failed: 0, processing: 0, pending: 0, cancelled: 0 }
    for (const j of recentJobs) {
      if (j.status in counts) counts[j.status as keyof typeof counts]++
    }
    return counts
  }, [recentJobs])

  // Total audio duration
  const totalDuration = useMemo(() => {
    let sec = 0
    for (const j of recentJobs) {
      const dur = j.result?.duration_seconds
      if (typeof dur === 'number') sec += dur
    }
    return sec
  }, [recentJobs])

  const formatDuration = (s: number) => {
    if (s < 60) return `${Math.round(s)}s`
    if (s < 3600) return `${Math.floor(s / 60)}m`
    return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`
  }

  // Simple SVG bar chart for status distribution
  const StatusChart = () => {
    const total = Object.values(statusCounts).reduce((a, b) => a + b, 0)
    if (!total) return null
    const items = [
      { label: 'Terminé', count: statusCounts.completed, color: '#22c55e' },
      { label: 'Échoué', count: statusCounts.failed, color: '#ef4444' },
      { label: 'En cours', count: statusCounts.processing, color: '#f5a623' },
      { label: 'En attente', count: statusCounts.pending, color: '#6a6a82' },
    ].filter(i => i.count > 0)

    return (
      <div className="space-y-2">
        <div className="flex gap-1 h-3 rounded-full overflow-hidden bg-panel">
          {items.map(item => (
            <div
              key={item.label}
              style={{ width: `${(item.count / total) * 100}%`, backgroundColor: item.color }}
              title={`${item.label}: ${item.count}`}
            />
          ))}
        </div>
        <div className="flex gap-3 text-xs text-muted">
          {items.map(item => (
            <span key={item.label} className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full" style={{ backgroundColor: item.color }} />
              {item.label} ({item.count})
            </span>
          ))}
        </div>
      </div>
    )
  }

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
          <p className="text-xs text-muted">Durée totale</p>
          <p className="text-2xl font-semibold font-mono text-green mt-1">{formatDuration(totalDuration)}</p>
        </div>
        <div className="bg-surface border border-border rounded-xl p-4">
          <p className="text-xs text-muted">Uptime</p>
          <p className="text-2xl font-semibold font-mono text-primary mt-1">
            {health ? `${Math.floor(health.uptime_seconds / 60)}m` : '--'}
          </p>
        </div>
      </div>

      {/* Status Distribution */}
      {recentJobs.length > 0 && (
        <Card title="Distribution des jobs">
          <StatusChart />
        </Card>
      )}

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
            {recentJobs.slice(0, 5).map(j => <JobCard key={j.job_id} job={j} />)}
          </div>
        </Card>
      )}

      <ToastContainer />
    </div>
  )
}
