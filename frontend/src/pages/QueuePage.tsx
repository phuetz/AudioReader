import { useState, useEffect, useCallback } from 'react'
import { ListOrdered, Plus, Pause, Play, Trash2, GripVertical, RefreshCw } from 'lucide-react'
import Card from '../components/ui/Card'
import Button from '../components/ui/Button'
import { ToastContainer, toast } from '../components/ui/Toast'
import apiClient from '../api/client'

const V2 = '/api/v2'

interface QueuedJob {
  id: string
  file_id: string
  title: string
  config: Record<string, unknown>
  priority: number
  status: string
  job_id?: string
  created_at: string
  started_at?: string
  completed_at?: string
  error?: string
}

interface QueueStatus {
  waiting: QueuedJob[]
  processing?: QueuedJob
  completed: QueuedJob[]
  failed: QueuedJob[]
  paused: boolean
  total: number
}

export default function QueuePage() {
  const [queue, setQueue] = useState<QueueStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [fileId, setFileId] = useState('')
  const [title, setTitle] = useState('')
  const [dragIndex, setDragIndex] = useState<number | null>(null)

  const fetchQueue = useCallback(async () => {
    try {
      const res = await apiClient.get(`${V2}/queue`)
      setQueue(res.data)
    } catch {
      toast.error('Erreur chargement queue')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchQueue()
    const interval = setInterval(fetchQueue, 5000)
    return () => clearInterval(interval)
  }, [fetchQueue])

  const addToQueue = async () => {
    if (!fileId.trim()) { toast.error('Entrez un File ID'); return }
    try {
      await apiClient.post(`${V2}/queue/add`, {
        file_id: fileId,
        title: title || `Job ${(queue?.total || 0) + 1}`,
      })
      setFileId('')
      setTitle('')
      fetchQueue()
      toast.success('Ajouté à la queue')
    } catch {
      toast.error('Erreur ajout à la queue')
    }
  }

  const removeFromQueue = async (id: string) => {
    try {
      await apiClient.delete(`${V2}/queue/${id}`)
      fetchQueue()
      toast.success('Retiré de la queue')
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Erreur suppression'
      toast.error(msg)
    }
  }

  const togglePause = async () => {
    try {
      if (queue?.paused) {
        await apiClient.post(`${V2}/queue/resume`)
        toast.success('Queue reprise')
      } else {
        await apiClient.post(`${V2}/queue/pause`)
        toast.success('Queue en pause')
      }
      fetchQueue()
    } catch {
      toast.error('Erreur')
    }
  }

  const clearCompleted = async () => {
    try {
      const res = await apiClient.post(`${V2}/queue/clear`)
      toast.success(`${res.data.removed} jobs nettoyés`)
      fetchQueue()
    } catch {
      toast.error('Erreur nettoyage')
    }
  }

  const statusBadge = (status: string) => {
    const colors: Record<string, string> = {
      waiting: 'bg-yellow-500/20 text-yellow-400',
      processing: 'bg-accent/20 text-accent',
      completed: 'bg-green-500/20 text-green-400',
      failed: 'bg-red-500/20 text-red-400',
    }
    return (
      <span className={`text-xs px-2 py-0.5 rounded-full ${colors[status] || 'bg-panel text-muted'}`}>
        {status}
      </span>
    )
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="w-8 h-8 border-2 border-accent border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  return (
    <div className="space-y-6 max-w-4xl">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold text-primary flex items-center gap-2">
          <ListOrdered className="w-5 h-5 text-accent" />
          File d'attente
          {queue?.paused && <span className="text-xs bg-yellow-500/20 text-yellow-400 px-2 py-0.5 rounded-full ml-2">PAUSE</span>}
        </h1>
        <div className="flex gap-2">
          <Button variant="ghost" onClick={fetchQueue} icon={<RefreshCw className="w-4 h-4" />}>
            Actualiser
          </Button>
          <Button
            variant={queue?.paused ? 'primary' : 'ghost'}
            onClick={togglePause}
            icon={queue?.paused ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
          >
            {queue?.paused ? 'Reprendre' : 'Pause'}
          </Button>
        </div>
      </div>

      {/* Add to queue */}
      <Card title="Ajouter à la queue" icon={<Plus className="w-4 h-4" />}>
        <div className="flex gap-3 items-end">
          <div className="flex-1">
            <label className="block text-xs text-muted mb-1">File ID</label>
            <input
              type="text"
              value={fileId}
              onChange={(e) => setFileId(e.target.value)}
              placeholder="ID du fichier uploadé"
              className="w-full px-3 py-2 rounded-lg bg-panel border border-border text-primary text-sm"
            />
          </div>
          <div className="flex-1">
            <label className="block text-xs text-muted mb-1">Titre (optionnel)</label>
            <input
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Nom du projet"
              className="w-full px-3 py-2 rounded-lg bg-panel border border-border text-primary text-sm"
            />
          </div>
          <Button onClick={addToQueue} icon={<Plus className="w-4 h-4" />}>Ajouter</Button>
        </div>
      </Card>

      {/* Processing */}
      {queue?.processing && (
        <Card title="En cours de traitement">
          <div className="flex items-center gap-3 p-2">
            <div className="w-3 h-3 rounded-full bg-accent animate-pulse" />
            <span className="font-medium text-primary">{queue.processing.title}</span>
            {statusBadge(queue.processing.status)}
            {queue.processing.job_id && (
              <a href={`/review/${queue.processing.job_id}`} className="text-xs text-accent hover:underline ml-auto">
                Voir le job
              </a>
            )}
          </div>
        </Card>
      )}

      {/* Waiting */}
      <Card title={`En attente (${queue?.waiting.length || 0})`}>
        {(!queue?.waiting || queue.waiting.length === 0) ? (
          <p className="text-sm text-muted py-4 text-center">Queue vide</p>
        ) : (
          <div className="space-y-1">
            {queue.waiting.map((job, index) => (
              <div
                key={job.id}
                draggable
                onDragStart={() => setDragIndex(index)}
                onDragOver={(e) => { e.preventDefault() }}
                onDrop={() => {
                  if (dragIndex !== null && dragIndex !== index) {
                    const ids = queue.waiting.map(j => j.id)
                    const [moved] = ids.splice(dragIndex, 1)
                    ids.splice(index, 0, moved)
                    apiClient.post(`${V2}/queue/reorder`, { order: ids })
                      .then(() => { fetchQueue(); toast.success('Queue réordonnée') })
                      .catch(() => toast.error('Erreur réordonnancement'))
                  }
                  setDragIndex(null)
                }}
                onDragEnd={() => setDragIndex(null)}
                className={`flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-panel group transition-opacity ${
                  dragIndex === index ? 'opacity-50' : ''
                }`}
              >
                <GripVertical className="w-4 h-4 text-muted opacity-0 group-hover:opacity-100 cursor-grab" />
                <span className="text-sm text-primary flex-1">{job.title}</span>
                <span className="text-xs text-muted">P{job.priority}</span>
                {statusBadge(job.status)}
                <button
                  onClick={() => removeFromQueue(job.id)}
                  className="opacity-0 group-hover:opacity-100 p-1 text-muted hover:text-red-400 transition-colors"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            ))}
          </div>
        )}
      </Card>

      {/* Completed & Failed */}
      {((queue?.completed?.length || 0) + (queue?.failed?.length || 0)) > 0 && (
        <Card title="Terminés">
          <div className="space-y-1">
            {queue?.failed?.map((job) => (
              <div key={job.id} className="flex items-center gap-3 px-3 py-2 rounded-lg">
                <span className="text-sm text-primary flex-1">{job.title}</span>
                {statusBadge('failed')}
                {job.error && <span className="text-xs text-red-400 truncate max-w-48">{job.error}</span>}
              </div>
            ))}
            {queue?.completed?.map((job) => (
              <div key={job.id} className="flex items-center gap-3 px-3 py-2 rounded-lg">
                <span className="text-sm text-primary flex-1">{job.title}</span>
                {statusBadge('completed')}
                {job.job_id && (
                  <a href={`/review/${job.job_id}`} className="text-xs text-accent hover:underline">
                    Réviser
                  </a>
                )}
              </div>
            ))}
          </div>
          <Button variant="ghost" onClick={clearCompleted} className="mt-3 text-xs">
            Nettoyer les terminés
          </Button>
        </Card>
      )}

      <ToastContainer />
    </div>
  )
}
