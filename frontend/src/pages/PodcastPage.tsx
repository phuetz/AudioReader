import { useEffect, useState } from 'react'
import {
  Radio,
  Play,
  Square,
  ExternalLink,
  QrCode,
  RefreshCw,
  FileAudio,
  Clock,
  HardDrive,
  Smartphone,
  Wifi,
  Rss,
} from 'lucide-react'
import Card from '../components/ui/Card'
import Button from '../components/ui/Button'
import Input from '../components/ui/Input'
import Badge from '../components/ui/Badge'
import { ToastContainer, toast } from '../components/ui/Toast'
import { usePodcastStore } from '../stores/usePodcastStore'
import apiClient from '../api/client'

interface Episode {
  name: string
  size_mb: number
  modified: number
}

export default function PodcastPage() {
  const { status, loading, fetchStatus, start, stop } = usePodcastStore()
  const [port, setPort] = useState(8080)
  const [title, setTitle] = useState('AudioReader Podcast')
  const [episodes, setEpisodes] = useState<Episode[]>([])
  const [loadingEpisodes, setLoadingEpisodes] = useState(false)

  useEffect(() => {
    fetchStatus()
  }, [fetchStatus])

  useEffect(() => {
    if (status?.running) {
      loadEpisodes()
    }
  }, [status?.running])

  const loadEpisodes = async () => {
    setLoadingEpisodes(true)
    try {
      const res = await apiClient.get<{ episodes: Episode[]; total: number }>('/api/v2/podcast/episodes')
      setEpisodes(res.data.episodes)
    } catch {
      // Ignore errors
    } finally {
      setLoadingEpisodes(false)
    }
  }

  const handleStart = async () => {
    try {
      await start({ port, title })
      toast.success('Serveur podcast démarré')
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Erreur')
    }
  }

  const handleStop = async () => {
    try {
      await stop()
      toast.success('Serveur podcast arrêté')
      setEpisodes([])
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Erreur')
    }
  }

  const handleRefresh = async () => {
    await fetchStatus()
    if (status?.running) {
      await loadEpisodes()
    }
    toast.info('Status mis à jour')
  }

  const formatDate = (timestamp: number) => {
    return new Date(timestamp * 1000).toLocaleDateString('fr-FR', {
      day: '2-digit',
      month: 'short',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  const formatSize = (mb: number) =>
    mb < 1 ? `${(mb * 1024).toFixed(0)} KB` : `${mb.toFixed(1)} MB`

  return (
    <div className="space-y-6 max-w-5xl">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold text-primary flex items-center gap-2">
          <Radio className="w-5 h-5 text-accent" />
          Podcast RSS
        </h1>
        <Button
          variant="ghost"
          size="sm"
          onClick={handleRefresh}
          icon={<RefreshCw className="w-4 h-4" />}
        >
          Rafraîchir
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Server Control */}
        <div className="lg:col-span-2 space-y-4">
          <Card title="Serveur" icon={<Rss className="w-4 h-4" />}>
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <span className="text-sm text-secondary">Status :</span>
                {status?.running ? (
                  <Badge color="green">En ligne</Badge>
                ) : (
                  <Badge color="muted">Hors ligne</Badge>
                )}
                {status?.running && status.episode_count > 0 && (
                  <span className="text-sm text-muted">
                    {status.episode_count} épisode{status.episode_count > 1 ? 's' : ''}
                  </span>
                )}
              </div>

              {status?.running ? (
                <div className="space-y-4">
                  {/* URL Info */}
                  <div className="p-4 bg-panel rounded-lg border border-border">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <p className="text-xs text-muted mb-1">URL du flux RSS</p>
                        <div className="flex items-center gap-2">
                          <code className="text-sm font-mono text-cyan">{status.url}/feed</code>
                          <a
                            href={`${status.url}/feed`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-accent hover:text-accent-hover"
                          >
                            <ExternalLink className="w-4 h-4" />
                          </a>
                        </div>
                      </div>
                      <div>
                        <p className="text-xs text-muted mb-1">Port</p>
                        <code className="text-sm font-mono text-primary">{status.port}</code>
                      </div>
                    </div>
                  </div>

                  <Button
                    variant="danger"
                    onClick={handleStop}
                    loading={loading}
                    icon={<Square className="w-4 h-4" />}
                  >
                    Arrêter le serveur
                  </Button>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <Input
                      label="Titre du podcast"
                      value={title}
                      onChange={(e) => setTitle(e.target.value)}
                    />
                    <Input
                      label="Port"
                      type="number"
                      value={port.toString()}
                      onChange={(e) => setPort(Number(e.target.value))}
                    />
                  </div>
                  <Button
                    onClick={handleStart}
                    loading={loading}
                    icon={<Play className="w-4 h-4" />}
                  >
                    Démarrer le serveur
                  </Button>
                </div>
              )}
            </div>
          </Card>

          {/* Episodes List */}
          {status?.running && (
            <Card
              title="Épisodes disponibles"
              icon={<FileAudio className="w-4 h-4" />}
              action={
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={loadEpisodes}
                  loading={loadingEpisodes}
                  icon={<RefreshCw className="w-3 h-3" />}
                >
                  Actualiser
                </Button>
              }
            >
              {episodes.length === 0 ? (
                <div className="py-8 text-center">
                  <FileAudio className="w-10 h-10 text-muted mx-auto mb-2" />
                  <p className="text-secondary text-sm">Aucun épisode</p>
                  <p className="text-xs text-muted mt-1">
                    Ajoutez des fichiers audio dans le dossier output/
                  </p>
                </div>
              ) : (
                <div className="divide-y divide-border/50 max-h-64 overflow-y-auto">
                  {episodes.map((ep, i) => (
                    <div key={ep.name} className="flex items-center gap-3 py-2">
                      <span className="w-6 text-xs text-muted text-center">{i + 1}</span>
                      <FileAudio className="w-4 h-4 text-cyan shrink-0" />
                      <div className="flex-1 min-w-0">
                        <p className="text-sm text-primary truncate">{ep.name}</p>
                        <div className="flex items-center gap-3 text-xs text-muted">
                          <span className="flex items-center gap-1">
                            <HardDrive className="w-3 h-3" />
                            {formatSize(ep.size_mb)}
                          </span>
                          <span className="flex items-center gap-1">
                            <Clock className="w-3 h-3" />
                            {formatDate(ep.modified)}
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </Card>
          )}
        </div>

        {/* QR Code & Instructions */}
        <div className="space-y-4">
          {/* QR Code */}
          {status?.running && status.qr_code_url && (
            <Card title="QR Code" icon={<QrCode className="w-4 h-4" />}>
              <div className="flex flex-col items-center py-2">
                <img
                  src={status.qr_code_url}
                  alt="QR Code du flux RSS"
                  className="w-40 h-40 rounded-lg bg-white p-2"
                />
                <p className="text-xs text-muted mt-3 text-center">
                  Scannez pour vous abonner au podcast
                </p>
              </div>
            </Card>
          )}

          {/* Instructions */}
          <Card title="Instructions" icon={<Smartphone className="w-4 h-4" />}>
            <div className="space-y-3 text-sm">
              <p className="text-secondary">
                Le serveur podcast expose un flux RSS local compatible avec :
              </p>
              <ul className="space-y-1.5">
                {['Apple Podcasts', 'Pocket Casts', 'AntennaPod', 'Overcast'].map((app) => (
                  <li key={app} className="flex items-center gap-2 text-muted">
                    <span className="w-1.5 h-1.5 rounded-full bg-accent" />
                    {app}
                  </li>
                ))}
              </ul>
              <div className="pt-3 border-t border-border space-y-2">
                <div className="flex items-start gap-2 text-xs text-muted">
                  <Wifi className="w-4 h-4 text-accent shrink-0 mt-0.5" />
                  <span>
                    Assurez-vous que votre appareil mobile est sur le même réseau Wi-Fi
                  </span>
                </div>
                <div className="flex items-start gap-2 text-xs text-muted">
                  <FileAudio className="w-4 h-4 text-accent shrink-0 mt-0.5" />
                  <span>
                    Les fichiers du dossier <code className="text-cyan">output/</code> sont
                    automatiquement ajoutés comme épisodes
                  </span>
                </div>
              </div>
            </div>
          </Card>

          {/* Quick Tips */}
          <Card title="Conseils">
            <ul className="space-y-2 text-xs text-muted">
              <li className="flex items-start gap-2">
                <span className="text-accent">1.</span>
                <span>Démarrez le serveur podcast</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-accent">2.</span>
                <span>Scannez le QR code avec votre téléphone</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-accent">3.</span>
                <span>Choisissez votre app de podcast préférée</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-accent">4.</span>
                <span>Écoutez vos audiobooks sans transfert de fichiers !</span>
              </li>
            </ul>
          </Card>
        </div>
      </div>

      <ToastContainer />
    </div>
  )
}
