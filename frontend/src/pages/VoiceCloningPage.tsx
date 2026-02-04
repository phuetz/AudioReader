import { useState, useEffect } from 'react'
import { Mic, Trash2 } from 'lucide-react'
import Card from '../components/ui/Card'
import Button from '../components/ui/Button'
import Input from '../components/ui/Input'
import DropZone from '../components/ui/DropZone'
import Badge from '../components/ui/Badge'
import { ToastContainer, toast } from '../components/ui/Toast'
import { cloneVoice, getClonedVoices, deleteClonedVoice } from '../api/endpoints'

interface ClonedVoice {
  id: string
  name: string
  language: string
}

export default function VoiceCloningPage() {
  const [name, setName] = useState('')
  const [language, setLanguage] = useState('fr')
  const [file, setFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [clonedVoices, setClonedVoices] = useState<ClonedVoice[]>([])

  const loadVoices = () => {
    getClonedVoices().then(d => setClonedVoices(d.voices)).catch(() => {})
  }

  useEffect(loadVoices, [])

  const handleClone = async () => {
    if (!file || !name.trim()) { toast.error('Nom et fichier audio requis'); return }
    setLoading(true)
    try {
      const fd = new FormData()
      fd.append('name', name)
      fd.append('language', language)
      fd.append('audio', file)
      const res = await cloneVoice(fd)
      toast.success(`Voix "${res.name}" clonée (${res.voice_id})`)
      setName('')
      setFile(null)
      loadVoices()
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Erreur clonage')
    } finally {
      setLoading(false)
    }
  }

  const handleDelete = async (voiceId: string) => {
    try {
      await deleteClonedVoice(voiceId)
      toast.success('Voix supprimée')
      loadVoices()
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Erreur suppression')
    }
  }

  return (
    <div className="space-y-6 max-w-4xl">
      <h1 className="text-xl font-semibold text-primary flex items-center gap-2">
        <Mic className="w-5 h-5 text-accent" />
        Clonage vocal
      </h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card title="Nouvelle voix clonée">
          <div className="space-y-4">
            <Input label="Nom de la voix" value={name} onChange={(e) => setName(e.target.value)}
              placeholder="Ma voix" />

            <div className="space-y-1.5">
              <label className="text-xs font-medium text-secondary">Langue</label>
              <select
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
                className="w-full px-3 py-2 rounded-lg bg-panel border border-border text-primary text-sm cursor-pointer"
              >
                <option value="fr">Français</option>
                <option value="en">English</option>
              </select>
            </div>

            <DropZone
              accept=".wav,.mp3,.mp4,.mkv"
              onFile={setFile}
              label={file ? file.name : 'Fichier audio (min 6 secondes)'}
            />

            <Button onClick={handleClone} loading={loading} icon={<Mic className="w-4 h-4" />}
              disabled={!file || !name.trim()}>
              Cloner la voix
            </Button>
          </div>
        </Card>

        <Card title="Voix clonées">
          {clonedVoices.length === 0 ? (
            <p className="text-sm text-muted py-4 text-center">Aucune voix clonée</p>
          ) : (
            <div className="space-y-2">
              {clonedVoices.map((v) => (
                <div key={v.id} className="flex items-center justify-between px-3 py-2 bg-panel rounded-lg border border-border">
                  <div>
                    <span className="text-sm font-medium text-primary">{v.name}</span>
                    <div className="flex items-center gap-2 mt-0.5">
                      <Badge color="accent">cloned</Badge>
                      <span className="text-xs text-muted">{v.language?.toUpperCase()}</span>
                    </div>
                  </div>
                  <button
                    onClick={() => handleDelete(v.id)}
                    className="text-muted hover:text-red transition-colors cursor-pointer"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>

      <ToastContainer />
    </div>
  )
}
