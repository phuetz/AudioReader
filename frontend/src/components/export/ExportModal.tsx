import { useState } from 'react'
import { X, Download, Music, Youtube, Radio, Headphones, Loader2 } from 'lucide-react'
import Button from '../ui/Button'
import apiClient from '../../api/client'

const V2 = '/api/v2'

interface ExportModalProps {
  jobId: string
  title?: string
  onClose: () => void
}

type Platform = 'spotify' | 'youtube' | 'podcast' | 'audible'

const PLATFORMS: { id: Platform; name: string; icon: React.ReactNode; description: string; specs: string }[] = [
  {
    id: 'spotify',
    name: 'Spotify',
    icon: <Music className="w-5 h-5" />,
    description: 'MP3 320kbps optimisé streaming',
    specs: '-14 LUFS, 44.1kHz, Stéréo',
  },
  {
    id: 'youtube',
    name: 'YouTube',
    icon: <Youtube className="w-5 h-5" />,
    description: 'MP4 avec waveform animée',
    specs: 'AAC 192kbps, 1920x1080',
  },
  {
    id: 'podcast',
    name: 'Podcast',
    icon: <Radio className="w-5 h-5" />,
    description: 'MP3 optimisé podcast',
    specs: '-16 LUFS, 44.1kHz, Mono',
  },
  {
    id: 'audible',
    name: 'Audible/ACX',
    icon: <Headphones className="w-5 h-5" />,
    description: 'MP3 ACX-compliant',
    specs: '-20 LUFS, 192kbps, Mono',
  },
]

export default function ExportModal({ jobId, title, onClose }: ExportModalProps) {
  const [selected, setSelected] = useState<Platform>('spotify')
  const [exporting, setExporting] = useState(false)
  const [result, setResult] = useState<{ download_url: string; format: string; file_size_mb: number } | null>(null)
  const [author, setAuthor] = useState('AudioReader')
  const [exportTitle, setExportTitle] = useState(title || 'audiobook')

  const handleExport = async () => {
    try {
      setExporting(true)
      setResult(null)
      const payload: Record<string, unknown> = {
        job_id: jobId,
        title: exportTitle,
        author,
      }
      if (selected === 'youtube') {
        (payload as Record<string, unknown>).show_waveform = true
      }

      const res = await apiClient.post(`${V2}/export/${selected}`, payload)

      // Poll for completion
      const exportJobId = res.data.job_id
      let attempts = 0
      while (attempts < 60) {
        await new Promise(r => setTimeout(r, 2000))
        const jobRes = await apiClient.get(`${V2}/jobs/${exportJobId}`)
        if (jobRes.data.status === 'completed') {
          setResult(jobRes.data.result)
          break
        }
        if (jobRes.data.status === 'failed') {
          throw new Error(jobRes.data.error || 'Export échoué')
        }
        attempts++
      }
    } catch (e) {
      // Error handled silently since toast isn't available in modal
      console.error('Export error:', e)
    } finally {
      setExporting(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-surface border border-border rounded-2xl w-full max-w-lg mx-4 shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-border">
          <h2 className="text-lg font-semibold text-primary flex items-center gap-2">
            <Download className="w-5 h-5 text-accent" />
            Export multi-plateformes
          </h2>
          <button onClick={onClose} className="p-1 text-muted hover:text-primary transition-colors">
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-4">
          {/* Metadata */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-muted mb-1">Titre</label>
              <input
                type="text"
                value={exportTitle}
                onChange={(e) => setExportTitle(e.target.value)}
                className="w-full px-3 py-2 rounded-lg bg-panel border border-border text-primary text-sm"
              />
            </div>
            <div>
              <label className="block text-xs text-muted mb-1">Auteur</label>
              <input
                type="text"
                value={author}
                onChange={(e) => setAuthor(e.target.value)}
                className="w-full px-3 py-2 rounded-lg bg-panel border border-border text-primary text-sm"
              />
            </div>
          </div>

          {/* Platform selection */}
          <div className="grid grid-cols-2 gap-2">
            {PLATFORMS.map((p) => (
              <button
                key={p.id}
                onClick={() => setSelected(p.id)}
                className={`flex items-start gap-3 p-3 rounded-xl border transition-colors text-left ${
                  selected === p.id
                    ? 'border-accent bg-accent/10'
                    : 'border-border hover:border-accent/50'
                }`}
              >
                <div className={selected === p.id ? 'text-accent' : 'text-muted'}>{p.icon}</div>
                <div>
                  <p className="text-sm font-medium text-primary">{p.name}</p>
                  <p className="text-xs text-muted">{p.description}</p>
                  <p className="text-xs text-cyan mt-0.5 font-mono">{p.specs}</p>
                </div>
              </button>
            ))}
          </div>

          {/* Result */}
          {result && (
            <div className="bg-green-500/10 border border-green-500/30 rounded-xl p-4">
              <p className="text-sm text-green-400 font-medium mb-2">Export terminé!</p>
              <p className="text-xs text-muted">Format: {result.format}</p>
              <p className="text-xs text-muted">Taille: {result.file_size_mb.toFixed(1)} MB</p>
              <a
                href={result.download_url}
                download
                className="inline-flex items-center gap-1 mt-2 text-sm text-accent hover:underline"
              >
                <Download className="w-3.5 h-3.5" />
                Télécharger
              </a>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex justify-end gap-3 px-6 py-4 border-t border-border">
          <Button variant="ghost" onClick={onClose}>Fermer</Button>
          <Button onClick={handleExport} loading={exporting} icon={
            exporting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />
          }>
            Exporter pour {PLATFORMS.find(p => p.id === selected)?.name}
          </Button>
        </div>
      </div>
    </div>
  )
}
