import { useState, useEffect } from 'react'
import { Sliders, Play, Save, Trash2, Volume2 } from 'lucide-react'
import Card from '../components/ui/Card'
import Button from '../components/ui/Button'
import Select from '../components/ui/Select'
import { ToastContainer, toast } from '../components/ui/Toast'
import { useVoiceStore } from '../stores/useVoiceStore'
import VoiceBlender from '../components/voice/VoiceBlender'
import VoiceComparison from '../components/audio/VoiceComparison'
import { previewVoice } from '../api/endpoints'

interface VoicePreset {
  id: string
  name: string
  blend: string
  morph: { pitch: number; formant: number; speed: number }
  description: string
}

export default function VoiceLabPage() {
  const { voices, fetchVoices } = useVoiceStore()
  const [blend, setBlend] = useState('')
  const [pitch, setPitch] = useState(0)
  const [formant, setFormant] = useState(0)
  const [speed, setSpeed] = useState(1.0)
  const [presets, setPresets] = useState<VoicePreset[]>([])
  const [presetName, setPresetName] = useState('')
  const [previewText, setPreviewText] = useState('Bonjour, je teste cette voix hybride.')
  const [previewLoading, setPreviewLoading] = useState(false)
  const [testVoice, setTestVoice] = useState('ff_siwis')

  useEffect(() => { fetchVoices() }, [fetchVoices])

  // Load presets from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('audioreader_voice_presets')
    if (saved) setPresets(JSON.parse(saved))
  }, [])

  const savePreset = () => {
    if (!presetName.trim()) { toast.error('Entrez un nom de preset'); return }
    const preset: VoicePreset = {
      id: Date.now().toString(36),
      name: presetName,
      blend,
      morph: { pitch, formant, speed },
      description: `Pitch: ${pitch > 0 ? '+' : ''}${pitch}, Formant: ${formant > 0 ? '+' : ''}${formant}, Speed: ${speed}`,
    }
    const updated = [...presets, preset]
    setPresets(updated)
    localStorage.setItem('audioreader_voice_presets', JSON.stringify(updated))
    setPresetName('')
    toast.success(`Preset "${preset.name}" sauvegardé`)
  }

  const loadPreset = (preset: VoicePreset) => {
    setBlend(preset.blend)
    setPitch(preset.morph.pitch)
    setFormant(preset.morph.formant)
    setSpeed(preset.morph.speed)
    toast.info(`Preset "${preset.name}" chargé`)
  }

  const deletePreset = (id: string) => {
    const updated = presets.filter(p => p.id !== id)
    setPresets(updated)
    localStorage.setItem('audioreader_voice_presets', JSON.stringify(updated))
    toast.success('Preset supprimé')
  }

  const handlePreview = async () => {
    try {
      setPreviewLoading(true)
      const voiceId = blend || testVoice
      const blob = await previewVoice(voiceId, previewText, speed)
      const url = URL.createObjectURL(blob)
      const audio = new Audio(url)
      audio.play()
      audio.onended = () => URL.revokeObjectURL(url)
    } catch {
      toast.error('Erreur de prévisualisation')
    } finally {
      setPreviewLoading(false)
    }
  }

  return (
    <div className="space-y-6 max-w-5xl">
      <h1 className="text-xl font-semibold text-primary flex items-center gap-2">
        <Sliders className="w-5 h-5 text-accent" />
        Labo Voix
      </h1>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Voice Blender */}
        <div className="lg:col-span-2">
          <Card title="Mélangeur de voix">
            <VoiceBlender
              voices={voices}
              value={blend}
              onChange={setBlend}
            />
          </Card>
        </div>

        {/* Morph Controls */}
        <div className="space-y-6">
          <Card title="Morphing vocal" icon={<Volume2 className="w-4 h-4" />}>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-muted">Pitch</span>
                  <span className="text-primary font-mono">{pitch > 0 ? '+' : ''}{pitch.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="-1"
                  max="1"
                  step="0.05"
                  value={pitch}
                  onChange={(e) => setPitch(parseFloat(e.target.value))}
                  className="w-full accent-accent"
                />
              </div>
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-muted">Formant</span>
                  <span className="text-primary font-mono">{formant > 0 ? '+' : ''}{formant.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="-1"
                  max="1"
                  step="0.05"
                  value={formant}
                  onChange={(e) => setFormant(parseFloat(e.target.value))}
                  className="w-full accent-accent"
                />
              </div>
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-muted">Vitesse</span>
                  <span className="text-primary font-mono">{speed.toFixed(2)}x</span>
                </div>
                <input
                  type="range"
                  min="0.5"
                  max="2"
                  step="0.05"
                  value={speed}
                  onChange={(e) => setSpeed(parseFloat(e.target.value))}
                  className="w-full accent-accent"
                />
              </div>
            </div>
          </Card>

          {/* Quick test voice selector */}
          <Card title="Test rapide">
            <Select
              label="Voix"
              value={testVoice}
              onChange={(e) => setTestVoice(e.target.value)}
              options={voices.map(v => ({ value: v.id, label: `${v.name} (${v.gender})` }))}
            />
          </Card>
        </div>
      </div>

      {/* Preview */}
      <Card title="Prévisualisation">
        <div className="flex gap-3 items-end">
          <div className="flex-1">
            <textarea
              value={previewText}
              onChange={(e) => setPreviewText(e.target.value)}
              rows={2}
              className="w-full px-3 py-2 rounded-lg bg-panel border border-border text-primary text-sm resize-y"
              placeholder="Texte de test..."
            />
          </div>
          <Button onClick={handlePreview} loading={previewLoading} icon={<Play className="w-4 h-4" />}>
            Écouter
          </Button>
        </div>
      </Card>

      {/* Presets */}
      <Card title="Presets sauvegardés">
        <div className="flex gap-3 items-end mb-4">
          <div className="flex-1">
            <input
              type="text"
              value={presetName}
              onChange={(e) => setPresetName(e.target.value)}
              placeholder="Nom du preset"
              className="w-full px-3 py-2 rounded-lg bg-panel border border-border text-primary text-sm"
            />
          </div>
          <Button onClick={savePreset} icon={<Save className="w-4 h-4" />} variant="secondary">
            Sauvegarder
          </Button>
        </div>

        {presets.length === 0 ? (
          <p className="text-sm text-muted py-4 text-center">Aucun preset sauvegardé</p>
        ) : (
          <div className="space-y-1">
            {presets.map((preset) => (
              <div key={preset.id} className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-panel group">
                <span className="text-sm text-primary flex-1">{preset.name}</span>
                <span className="text-xs text-muted font-mono">{preset.blend || 'single'}</span>
                <span className="text-xs text-muted">{preset.description}</span>
                <button onClick={() => loadPreset(preset)} className="text-xs text-accent hover:underline">
                  Charger
                </button>
                <button
                  onClick={() => deletePreset(preset.id)}
                  className="opacity-0 group-hover:opacity-100 p-1 text-muted hover:text-red-400"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            ))}
          </div>
        )}
      </Card>

      {/* A/B Voice Comparison */}
      <VoiceComparison />

      <ToastContainer />
    </div>
  )
}
