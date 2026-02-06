import { useState, useEffect, useRef, useCallback } from 'react'
import { useParams } from 'react-router-dom'
import { Play, Pause, RotateCcw, Save, Layers, AlertTriangle, FileText, Undo2, Redo2 } from 'lucide-react'
import Card from '../components/ui/Card'
import Button from '../components/ui/Button'
import Select from '../components/ui/Select'
import { ToastContainer, toast } from '../components/ui/Toast'
import apiClient from '../api/client'

const V2 = '/api/v2'

interface ReviewSegment {
  id: string
  index: number
  text: string
  audio_url: string
  duration: number
  speaker?: string
  emotion?: string
  voice_id: string
}

interface ConsistencyIssue {
  type: string
  severity: string
  segment_indices: number[]
  description: string
  suggestion: string
}

export default function ReviewPage() {
  const { jobId } = useParams<{ jobId: string }>()
  const [segments, setSegments] = useState<ReviewSegment[]>([])
  const [loading, setLoading] = useState(true)
  const [playingId, setPlayingId] = useState<string | null>(null)
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editText, setEditText] = useState('')
  const [regenVoice] = useState('')
  const [regenSpeed] = useState(1.0)
  const [rebuilding, setRebuilding] = useState(false)
  const [consistencyIssues, setConsistencyIssues] = useState<ConsistencyIssue[]>([])
  const [consistencyScore, setConsistencyScore] = useState<number | null>(null)
  const [showConsistency, setShowConsistency] = useState(false)
  const [subtitleFormat, setSubtitleFormat] = useState('srt')
  const [generatingSubs, setGeneratingSubs] = useState(false)
  const [currentAudioTime, setCurrentAudioTime] = useState(0)
  const audioRef = useRef<HTMLAudioElement | null>(null)

  // Undo/redo
  const [history, setHistory] = useState<ReviewSegment[][]>([])
  const [historyIndex, setHistoryIndex] = useState(-1)

  const pushHistory = useCallback((segs: ReviewSegment[]) => {
    setHistory(prev => [...prev.slice(0, historyIndex + 1), segs])
    setHistoryIndex(prev => prev + 1)
  }, [historyIndex])

  const undo = useCallback(() => {
    if (historyIndex > 0) {
      setHistoryIndex(prev => prev - 1)
      setSegments(history[historyIndex - 1])
    }
  }, [historyIndex, history])

  const redo = useCallback(() => {
    if (historyIndex < history.length - 1) {
      setHistoryIndex(prev => prev + 1)
      setSegments(history[historyIndex + 1])
    }
  }, [historyIndex, history])

  useEffect(() => {
    if (jobId) fetchSegments()
  }, [jobId])

  // Keyboard shortcuts for undo/redo
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'z') {
        if (e.shiftKey) { redo(); } else { undo(); }
        e.preventDefault()
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [undo, redo])

  // Track audio time for subtitle sync
  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return
    const onTime = () => setCurrentAudioTime(audio.currentTime)
    audio.addEventListener('timeupdate', onTime)
    return () => audio.removeEventListener('timeupdate', onTime)
  }, [])

  const fetchSegments = async () => {
    try {
      setLoading(true)
      const res = await apiClient.get(`${V2}/jobs/${jobId}/segments`)
      setSegments(res.data.segments || [])
    } catch {
      toast.error('Impossible de charger les segments')
    } finally {
      setLoading(false)
    }
  }

  const playSegment = (seg: ReviewSegment) => {
    if (playingId === seg.id) {
      audioRef.current?.pause()
      setPlayingId(null)
      return
    }
    if (audioRef.current) {
      audioRef.current.src = seg.audio_url
      audioRef.current.play()
      setPlayingId(seg.id)
      audioRef.current.onended = () => setPlayingId(null)
    }
  }

  const startEdit = (seg: ReviewSegment) => {
    setEditingId(seg.id)
    setEditText(seg.text)
  }

  const saveEdit = async (segId: string) => {
    try {
      await apiClient.put(`${V2}/jobs/${jobId}/segments/${segId}`, { text: editText })
      const updated = segments.map(s => s.id === segId ? { ...s, text: editText } : s)
      pushHistory(updated)
      setSegments(updated)
      setEditingId(null)
      toast.success('Texte mis à jour')
    } catch {
      toast.error('Erreur lors de la sauvegarde')
    }
  }

  const regenerateSegment = async (segId: string) => {
    try {
      const res = await apiClient.post(`${V2}/jobs/${jobId}/segments/${segId}/regenerate`, {
        voice_id: regenVoice || undefined,
        speed: regenSpeed,
      })
      toast.success(`Régénération démarrée (job: ${res.data.job_id})`)
    } catch {
      toast.error('Erreur lors de la régénération')
    }
  }

  const rebuildAudio = async () => {
    try {
      setRebuilding(true)
      const res = await apiClient.post(`${V2}/jobs/${jobId}/rebuild`)
      toast.success(`Reconstruction démarrée (job: ${res.data.job_id})`)
    } catch {
      toast.error('Erreur lors de la reconstruction')
    } finally {
      setRebuilding(false)
    }
  }

  const analyzeConsistency = async () => {
    try {
      const res = await apiClient.post(`${V2}/jobs/${jobId}/consistency`)
      setConsistencyIssues(res.data.issues || [])
      setConsistencyScore(res.data.score)
      setShowConsistency(true)
      if (res.data.issues.length === 0) {
        toast.success('Aucune incohérence détectée!')
      } else {
        toast.info(`${res.data.issues.length} problème(s) détecté(s)`)
      }
    } catch {
      toast.error('Erreur lors de l\'analyse')
    }
  }

  const generateSubtitles = async () => {
    try {
      setGeneratingSubs(true)
      const res = await apiClient.post(`${V2}/jobs/${jobId}/subtitles`, {
        format: subtitleFormat,
        mode: 'estimated',
      })
      toast.success(`Sous-titres en cours de génération (job: ${res.data.job_id})`)
    } catch {
      toast.error('Erreur lors de la génération des sous-titres')
    } finally {
      setGeneratingSubs(false)
    }
  }

  const severityColor = (sev: string) => {
    if (sev === 'error') return 'text-red-400'
    if (sev === 'warning') return 'text-yellow-400'
    return 'text-blue-400'
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="w-8 h-8 border-2 border-accent border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  return (
    <div className="space-y-6 max-w-5xl">
      <audio ref={audioRef} className="hidden" />

      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold text-primary flex items-center gap-2">
          <Layers className="w-5 h-5 text-accent" />
          Mode Révision
          <span className="text-sm font-normal text-muted">Job: {jobId}</span>
        </h1>
        <div className="flex gap-2">
          <Button variant="ghost" onClick={undo} disabled={historyIndex <= 0} icon={<Undo2 className="w-4 h-4" />}>
            Annuler
          </Button>
          <Button variant="ghost" onClick={redo} disabled={historyIndex >= history.length - 1} icon={<Redo2 className="w-4 h-4" />}>
            Rétablir
          </Button>
          <Button variant="ghost" onClick={analyzeConsistency} icon={<AlertTriangle className="w-4 h-4" />}>
            Cohérence
          </Button>
          <Button onClick={rebuildAudio} loading={rebuilding} icon={<Save className="w-4 h-4" />}>
            Reconstruire
          </Button>
        </div>
      </div>

      {/* Consistency panel */}
      {showConsistency && (
        <Card title={`Analyse de cohérence — Score: ${consistencyScore}/100`}>
          {consistencyIssues.length === 0 ? (
            <p className="text-sm text-green-400">Aucune incohérence détectée.</p>
          ) : (
            <div className="space-y-2">
              {consistencyIssues.map((issue, i) => (
                <div key={i} className="flex items-start gap-3 p-2 rounded-lg bg-panel">
                  <AlertTriangle className={`w-4 h-4 mt-0.5 shrink-0 ${severityColor(issue.severity)}`} />
                  <div>
                    <p className="text-sm text-primary">{issue.description}</p>
                    {issue.suggestion && (
                      <p className="text-xs text-muted mt-1">{issue.suggestion}</p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </Card>
      )}

      {/* Subtitles section */}
      <Card title="Sous-titres" icon={<FileText className="w-4 h-4" />}>
        <div className="flex items-center gap-3">
          <Select
            label=""
            value={subtitleFormat}
            onChange={(e) => setSubtitleFormat(e.target.value)}
            options={[
              { value: 'srt', label: 'SRT' },
              { value: 'vtt', label: 'WebVTT' },
              { value: 'json', label: 'JSON (mot par mot)' },
            ]}
          />
          <Button onClick={generateSubtitles} loading={generatingSubs} variant="secondary">
            Générer
          </Button>
        </div>
      </Card>

      {/* Segments list */}
      <div className="space-y-2">
        <p className="text-sm text-muted">{segments.length} segments</p>
        {segments.map((seg) => {
          // Subtitle sync: estimate if this segment is "current" based on cumulative duration
          const cumulativeBefore = segments.slice(0, seg.index).reduce((s, x) => s + x.duration, 0)
          const isActiveSubtitle = playingId && currentAudioTime >= cumulativeBefore && currentAudioTime < cumulativeBefore + seg.duration

          return (
          <div
            key={seg.id}
            className={`bg-surface border rounded-xl p-4 transition-colors ${
              playingId === seg.id ? 'border-accent' : isActiveSubtitle ? 'border-cyan' : 'border-border'
            }`}
          >
            <div className="flex items-start gap-3">
              {/* Play button */}
              <button
                onClick={() => playSegment(seg)}
                className="mt-1 w-8 h-8 flex items-center justify-center rounded-full bg-accent/20 text-accent hover:bg-accent/30 transition-colors shrink-0"
              >
                {playingId === seg.id ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5 ml-0.5" />}
              </button>

              {/* Content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-xs font-mono text-muted">#{seg.index + 1}</span>
                  {seg.speaker && (
                    <span className="text-xs px-1.5 py-0.5 rounded bg-cyan/20 text-cyan">{seg.speaker}</span>
                  )}
                  {seg.emotion && (
                    <span className="text-xs px-1.5 py-0.5 rounded bg-accent/20 text-accent">{seg.emotion}</span>
                  )}
                  <span className="text-xs text-muted">{seg.duration.toFixed(1)}s</span>
                  <span className="text-xs text-muted font-mono">{seg.voice_id}</span>
                </div>

                {editingId === seg.id ? (
                  <div className="space-y-2">
                    <textarea
                      value={editText}
                      onChange={(e) => setEditText(e.target.value)}
                      rows={3}
                      className="w-full px-3 py-2 rounded-lg bg-panel border border-border text-primary text-sm resize-y"
                    />
                    <div className="flex gap-2">
                      <Button size="sm" onClick={() => saveEdit(seg.id)}>Sauvegarder</Button>
                      <Button size="sm" variant="ghost" onClick={() => setEditingId(null)}>Annuler</Button>
                    </div>
                  </div>
                ) : (
                  <p
                    className="text-sm text-primary cursor-pointer hover:bg-panel rounded px-1 -mx-1 py-0.5"
                    onClick={() => startEdit(seg)}
                    title="Cliquer pour éditer"
                  >
                    {seg.text}
                  </p>
                )}
              </div>

              {/* Regenerate button */}
              <button
                onClick={() => regenerateSegment(seg.id)}
                className="mt-1 p-1.5 rounded-lg text-muted hover:text-primary hover:bg-panel transition-colors"
                title="Régénérer ce segment"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
            </div>
          </div>
        )})}
      </div>

      {segments.length === 0 && !loading && (
        <div className="text-center py-12 text-muted">
          <Layers className="w-12 h-12 mx-auto mb-3 opacity-30" />
          <p>Aucun segment disponible pour ce job.</p>
          <p className="text-xs mt-1">Les segments sont créés lors de la génération d'un audiobook.</p>
        </div>
      )}

      <ToastContainer />
    </div>
  )
}
