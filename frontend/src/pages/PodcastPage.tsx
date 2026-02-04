import { useEffect, useState } from 'react'
import { Radio, Play, Square, ExternalLink } from 'lucide-react'
import Card from '../components/ui/Card'
import Button from '../components/ui/Button'
import Input from '../components/ui/Input'
import Badge from '../components/ui/Badge'
import { ToastContainer, toast } from '../components/ui/Toast'
import { usePodcastStore } from '../stores/usePodcastStore'

export default function PodcastPage() {
  const { status, loading, fetchStatus, start, stop } = usePodcastStore()
  const [port, setPort] = useState(8080)
  const [title, setTitle] = useState('AudioReader Podcast')

  useEffect(() => { fetchStatus() }, [fetchStatus])

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
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Erreur')
    }
  }

  return (
    <div className="space-y-6 max-w-4xl">
      <h1 className="text-xl font-semibold text-primary flex items-center gap-2">
        <Radio className="w-5 h-5 text-accent" />
        Podcast RSS
      </h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card title="Serveur">
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <span className="text-sm text-secondary">Status :</span>
              {status?.running ? (
                <Badge color="green">En ligne</Badge>
              ) : (
                <Badge color="muted">Hors ligne</Badge>
              )}
            </div>

            {status?.running ? (
              <div className="space-y-3">
                <div className="p-3 bg-panel rounded-lg border border-border">
                  <p className="text-xs text-muted">URL du flux RSS</p>
                  <div className="flex items-center gap-2 mt-1">
                    <code className="text-sm font-mono text-cyan flex-1">{status.url}/feed</code>
                    <a href={`${status.url}/feed`} target="_blank" rel="noopener"
                      className="text-accent hover:text-accent-hover">
                      <ExternalLink className="w-4 h-4" />
                    </a>
                  </div>
                </div>
                <p className="text-sm text-secondary">
                  Episodes : <span className="font-mono text-accent">{status.episode_count}</span>
                </p>
                <Button variant="danger" onClick={handleStop} loading={loading} icon={<Square className="w-4 h-4" />}>
                  Arrêter
                </Button>
              </div>
            ) : (
              <div className="space-y-4">
                <Input label="Titre du podcast" value={title}
                  onChange={(e) => setTitle(e.target.value)} />
                <Input label="Port" type="number" value={port.toString()}
                  onChange={(e) => setPort(Number(e.target.value))} />
                <Button onClick={handleStart} loading={loading} icon={<Play className="w-4 h-4" />}>
                  Démarrer le serveur
                </Button>
              </div>
            )}
          </div>
        </Card>

        <Card title="Instructions">
          <div className="space-y-3 text-sm text-secondary">
            <p>Le serveur podcast expose un flux RSS local compatible avec :</p>
            <ul className="list-disc list-inside space-y-1 text-muted">
              <li>Apple Podcasts</li>
              <li>Pocket Casts</li>
              <li>AntennaPod</li>
              <li>Tout lecteur RSS</li>
            </ul>
            <p>Les fichiers audio du dossier <code className="text-cyan">output/</code> sont automatiquement ajoutés comme épisodes.</p>
            <p className="text-xs text-muted mt-4">Assurez-vous que votre appareil est sur le même réseau Wi-Fi.</p>
          </div>
        </Card>
      </div>

      <ToastContainer />
    </div>
  )
}
