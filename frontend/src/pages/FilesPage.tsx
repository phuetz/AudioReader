import { useEffect, useState } from 'react'
import { FolderOpen, Download, Play } from 'lucide-react'
import Card from '../components/ui/Card'
import Badge from '../components/ui/Badge'
import AudioPlayer from '../components/audio/AudioPlayer'
import { ToastContainer } from '../components/ui/Toast'
import { getFiles } from '../api/endpoints'
import type { FileInfo } from '../api/types'

export default function FilesPage() {
  const [files, setFiles] = useState<FileInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedFile, setSelectedFile] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    getFiles().then(d => { setFiles(d.files); setLoading(false) }).catch(() => setLoading(false))
  }, [])

  const formatSize = (mb: number) => mb < 1 ? `${(mb * 1024).toFixed(0)} KB` : `${mb.toFixed(1)} MB`
  const formatDate = (iso: string) => new Date(iso).toLocaleDateString('fr-FR', {
    day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit',
  })

  return (
    <div className="space-y-6 max-w-5xl">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold text-primary flex items-center gap-2">
          <FolderOpen className="w-5 h-5 text-accent" />
          Fichiers générés
        </h1>
        <span className="text-sm text-muted">{files.length} fichiers</span>
      </div>

      {selectedFile && <AudioPlayer url={selectedFile} title="Lecture" />}

      <Card>
        {loading ? (
          <div className="py-8 text-center text-muted">Chargement...</div>
        ) : files.length === 0 ? (
          <div className="py-8 text-center text-muted">Aucun fichier généré</div>
        ) : (
          <div className="divide-y divide-border/50">
            {files.map((f) => (
              <div key={f.id} className="flex items-center gap-4 py-3 hover:bg-panel/50 px-2 rounded">
                <button
                  onClick={() => setSelectedFile(f.download_url)}
                  className="text-cyan hover:text-accent transition-colors cursor-pointer"
                >
                  <Play className="w-4 h-4" />
                </button>

                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-primary truncate">{f.name}</p>
                  <div className="flex items-center gap-3 mt-0.5">
                    <span className="text-xs text-muted">{formatSize(f.size_mb)}</span>
                    <span className="text-xs text-muted">{formatDate(f.created_at)}</span>
                  </div>
                </div>

                <Badge color="cyan">{f.name.split('.').pop()?.toUpperCase()}</Badge>

                <a
                  href={f.download_url}
                  download
                  className="text-muted hover:text-accent transition-colors"
                >
                  <Download className="w-4 h-4" />
                </a>
              </div>
            ))}
          </div>
        )}
      </Card>

      <ToastContainer />
    </div>
  )
}
