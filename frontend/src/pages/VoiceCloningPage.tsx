import { useState, useEffect } from 'react'
import { Mic, Trash2, Sliders, Play, Volume2 } from 'lucide-react'
import Card from '../components/ui/Card'
import Button from '../components/ui/Button'
import Input from '../components/ui/Input'
import Slider from '../components/ui/Slider'
import Toggle from '../components/ui/Toggle'
import DropZone from '../components/ui/DropZone'
import Badge from '../components/ui/Badge'
import { ToastContainer, toast } from '../components/ui/Toast'
import { cloneVoice, getClonedVoices, deleteClonedVoice, previewVoice } from '../api/endpoints'

interface ClonedVoice {
  id: string
  name: string
  language: string
}

interface MorphSettings {
  pitch: number      // -12 to +12 semitones
  formant: number    // 0.5 to 2.0
  speed: number      // 0.5 to 2.0
  enabled: boolean
}

export default function VoiceCloningPage() {
  // Cloning state
  const [name, setName] = useState('')
  const [language, setLanguage] = useState('fr')
  const [file, setFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [clonedVoices, setClonedVoices] = useState<ClonedVoice[]>([])

  // Morphing state
  const [morphSettings, setMorphSettings] = useState<MorphSettings>({
    pitch: 0,
    formant: 1.0,
    speed: 1.0,
    enabled: false,
  })
  const [selectedVoice, setSelectedVoice] = useState<string | null>(null)
  const [previewText, setPreviewText] = useState('Bonjour, ceci est un test de la voix modifiée.')
  const [previewing, setPreviewing] = useState(false)
  const [audioUrl, setAudioUrl] = useState<string | null>(null)

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
      if (selectedVoice === voiceId) setSelectedVoice(null)
      loadVoices()
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Erreur suppression')
    }
  }

  const handlePreview = async () => {
    if (!selectedVoice) { toast.error('Sélectionnez une voix'); return }
    setPreviewing(true)
    setAudioUrl(null)
    try {
      const blob = await previewVoice(selectedVoice, previewText, morphSettings.speed, language)
      const url = URL.createObjectURL(blob)
      setAudioUrl(url)
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Erreur preview')
    } finally {
      setPreviewing(false)
    }
  }

  const updateMorph = (key: keyof MorphSettings, value: number | boolean) => {
    setMorphSettings(prev => ({ ...prev, [key]: value }))
  }

  const getPitchLabel = (pitch: number) => {
    if (pitch === 0) return 'Normal'
    if (pitch > 0) return `+${pitch} demi-tons`
    return `${pitch} demi-tons`
  }

  return (
    <div className="space-y-6 max-w-5xl">
      <h1 className="text-xl font-semibold text-primary flex items-center gap-2">
        <Mic className="w-5 h-5 text-accent" />
        Clonage & Morphing Vocal
      </h1>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Colonne 1: Clonage */}
        <div className="space-y-4">
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

          {/* Voix clonées */}
          <Card title="Voix clonées">
            {clonedVoices.length === 0 ? (
              <p className="text-sm text-muted py-4 text-center">Aucune voix clonée</p>
            ) : (
              <div className="space-y-2">
                {clonedVoices.map((v) => (
                  <div
                    key={v.id}
                    onClick={() => setSelectedVoice(v.id)}
                    className={`flex items-center justify-between px-3 py-2 rounded-lg border cursor-pointer transition-colors ${
                      selectedVoice === v.id
                        ? 'bg-accent/10 border-accent'
                        : 'bg-panel border-border hover:border-accent/50'
                    }`}
                  >
                    <div>
                      <span className="text-sm font-medium text-primary">{v.name}</span>
                      <div className="flex items-center gap-2 mt-0.5">
                        <Badge color="accent">cloned</Badge>
                        <span className="text-xs text-muted">{v.language?.toUpperCase()}</span>
                      </div>
                    </div>
                    <button
                      onClick={(e) => { e.stopPropagation(); handleDelete(v.id) }}
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

        {/* Colonne 2: Voice Morphing */}
        <div className="space-y-4">
          <Card title="Voice Morphing" icon={<Sliders className="w-4 h-4" />}>
            <div className="space-y-4">
              <Toggle
                label="Activer le morphing"
                checked={morphSettings.enabled}
                onChange={(v) => updateMorph('enabled', v)}
                description="Modifie la voix en temps réel"
              />

              {morphSettings.enabled && (
                <>
                  {/* Pitch */}
                  <div className="space-y-2">
                    <Slider
                      label="Pitch (hauteur)"
                      value={morphSettings.pitch}
                      min={-12}
                      max={12}
                      step={1}
                      onChange={(v) => updateMorph('pitch', v)}
                      displayValue={getPitchLabel}
                    />
                    <p className="text-xs text-muted">
                      Négatif = plus grave, Positif = plus aigu
                    </p>
                  </div>

                  {/* Formant */}
                  <div className="space-y-2">
                    <Slider
                      label="Formant (timbre)"
                      value={morphSettings.formant}
                      min={0.5}
                      max={2.0}
                      step={0.1}
                      onChange={(v) => updateMorph('formant', v)}
                      displayValue={(v) => `${v.toFixed(1)}x`}
                    />
                    <p className="text-xs text-muted">
                      &lt;1 = voix plus petite, &gt;1 = voix plus grande
                    </p>
                  </div>

                  {/* Speed */}
                  <div className="space-y-2">
                    <Slider
                      label="Vitesse"
                      value={morphSettings.speed}
                      min={0.5}
                      max={2.0}
                      step={0.1}
                      onChange={(v) => updateMorph('speed', v)}
                      displayValue={(v) => `${v.toFixed(1)}x`}
                    />
                    <p className="text-xs text-muted">
                      Ajuste la vitesse de parole
                    </p>
                  </div>

                  {/* Reset button */}
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setMorphSettings({
                      pitch: 0,
                      formant: 1.0,
                      speed: 1.0,
                      enabled: true,
                    })}
                  >
                    Réinitialiser
                  </Button>
                </>
              )}
            </div>
          </Card>

          {/* Presets rapides */}
          {morphSettings.enabled && (
            <Card title="Presets rapides">
              <div className="grid grid-cols-2 gap-2">
                {[
                  { label: 'Grave', pitch: -4, formant: 0.9 },
                  { label: 'Aigu', pitch: 4, formant: 1.1 },
                  { label: 'Robot', pitch: 0, formant: 0.5 },
                  { label: 'Géant', pitch: -8, formant: 1.5 },
                  { label: 'Enfant', pitch: 6, formant: 1.3 },
                  { label: 'Murmure', pitch: 2, formant: 0.8 },
                ].map((preset) => (
                  <button
                    key={preset.label}
                    onClick={() => setMorphSettings({
                      ...morphSettings,
                      pitch: preset.pitch,
                      formant: preset.formant,
                    })}
                    className="px-3 py-2 text-sm bg-panel border border-border rounded-lg hover:border-accent/50 transition-colors"
                  >
                    {preset.label}
                  </button>
                ))}
              </div>
            </Card>
          )}
        </div>

        {/* Colonne 3: Preview */}
        <div className="space-y-4">
          <Card title="Aperçu" icon={<Volume2 className="w-4 h-4" />}>
            <div className="space-y-4">
              {selectedVoice ? (
                <>
                  <div className="p-3 bg-accent/10 rounded-lg border border-accent/30">
                    <p className="text-xs text-muted">Voix sélectionnée</p>
                    <p className="text-sm font-medium text-accent">
                      {clonedVoices.find(v => v.id === selectedVoice)?.name || selectedVoice}
                    </p>
                  </div>

                  <div className="space-y-1.5">
                    <label className="text-xs font-medium text-secondary">Texte de test</label>
                    <textarea
                      value={previewText}
                      onChange={(e) => setPreviewText(e.target.value)}
                      rows={3}
                      className="w-full px-3 py-2 rounded-lg bg-panel border border-border text-primary text-sm resize-none"
                    />
                  </div>

                  <Button
                    onClick={handlePreview}
                    loading={previewing}
                    icon={<Play className="w-4 h-4" />}
                    disabled={!previewText.trim()}
                  >
                    Générer aperçu
                  </Button>

                  {audioUrl && (
                    <div className="mt-4">
                      <audio controls src={audioUrl} className="w-full" />
                    </div>
                  )}

                  {morphSettings.enabled && (
                    <div className="p-3 bg-panel rounded-lg border border-border">
                      <p className="text-xs text-muted mb-2">Paramètres de morphing actifs :</p>
                      <div className="grid grid-cols-3 gap-2 text-xs">
                        <div>
                          <span className="text-muted">Pitch:</span>
                          <span className="text-primary ml-1">{morphSettings.pitch > 0 ? '+' : ''}{morphSettings.pitch}</span>
                        </div>
                        <div>
                          <span className="text-muted">Formant:</span>
                          <span className="text-primary ml-1">{morphSettings.formant.toFixed(1)}x</span>
                        </div>
                        <div>
                          <span className="text-muted">Vitesse:</span>
                          <span className="text-primary ml-1">{morphSettings.speed.toFixed(1)}x</span>
                        </div>
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div className="flex flex-col items-center justify-center py-8 text-center">
                  <Volume2 className="w-10 h-10 text-muted mb-3" />
                  <p className="text-sm text-secondary">Sélectionnez une voix clonée</p>
                  <p className="text-xs text-muted mt-1">pour tester le morphing vocal</p>
                </div>
              )}
            </div>
          </Card>

          {/* Tips */}
          <Card title="Conseils">
            <ul className="space-y-2 text-xs text-muted">
              <li className="flex items-start gap-2">
                <span className="text-accent">•</span>
                <span>Utilisez un enregistrement de 6-15 secondes pour le clonage</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-accent">•</span>
                <span>Évitez le bruit de fond et la musique</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-accent">•</span>
                <span>Le morphing peut créer des effets vocaux uniques</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-accent">•</span>
                <span>Combinez pitch et formant pour des transformations naturelles</span>
              </li>
            </ul>
          </Card>
        </div>
      </div>

      <ToastContainer />
    </div>
  )
}
