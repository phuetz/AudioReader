import { useState } from 'react'
import { Play, Zap } from 'lucide-react'
import Card from '../components/ui/Card'
import Button from '../components/ui/Button'
import Slider from '../components/ui/Slider'
import Select from '../components/ui/Select'
import VoiceSelector from '../components/voices/VoiceSelector'
import AudioPlayer from '../components/audio/AudioPlayer'
import JobProgress from '../components/jobs/JobProgress'
import { ToastContainer, toast } from '../components/ui/Toast'
import { generateAudio, generatePreview } from '../api/endpoints'
import { useSettingsStore } from '../stores/useSettingsStore'

export default function QuickTextPage() {
  const { language, defaultVoice, speed, setSpeed } = useSettingsStore()
  const [text, setText] = useState('')
  const [voice, setVoice] = useState(defaultVoice)
  const [lang, setLang] = useState(language)
  const [jobId, setJobId] = useState<string | null>(null)
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const handleGenerate = async () => {
    if (!text.trim()) { toast.error('Entrez du texte'); return }
    setLoading(true)
    setAudioUrl(null)
    try {
      const res = await generateAudio({ text, voice, speed, language: lang })
      setJobId(res.job_id)
      toast.info(`Job ${res.job_id} démarré`)
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Erreur')
    } finally {
      setLoading(false)
    }
  }

  const handlePreview = async () => {
    if (!text.trim()) { toast.error('Entrez du texte'); return }
    setLoading(true)
    setAudioUrl(null)
    try {
      const res = await generatePreview({ text, voice, speed, language: lang, duration: 30 })
      setJobId(res.job_id)
      toast.info('Preview en cours...')
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Erreur')
    } finally {
      setLoading(false)
    }
  }

  const handleComplete = (result: Record<string, unknown>) => {
    const url = result.download_url as string
    if (url) setAudioUrl(url)
    toast.success('Audio généré !')
  }

  return (
    <div className="space-y-6 max-w-4xl">
      <h1 className="text-xl font-semibold text-primary">Texte rapide</h1>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Text input */}
        <div className="lg:col-span-2 space-y-4">
          <Card title="Texte">
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Entrez votre texte ici..."
              rows={10}
              className="w-full px-3 py-2 rounded-lg bg-panel border border-border text-primary text-sm
                placeholder:text-muted focus:border-accent resize-y"
            />
            <div className="flex items-center gap-2 mt-3">
              <span className="text-xs text-muted">{text.length} caractères</span>
              <span className="text-xs text-muted">{text.split(/\s+/).filter(Boolean).length} mots</span>
            </div>
          </Card>

          <div className="flex gap-2">
            <Button onClick={handleGenerate} loading={loading} icon={<Play className="w-4 h-4" />}>
              Générer
            </Button>
            <Button variant="secondary" onClick={handlePreview} loading={loading} icon={<Zap className="w-4 h-4" />}>
              Preview 30s
            </Button>
          </div>

          {jobId && <JobProgress jobId={jobId} onComplete={handleComplete} />}
          {audioUrl && <AudioPlayer url={audioUrl} title="Résultat" />}
        </div>

        {/* Settings panel */}
        <div className="space-y-4">
          <Card title="Voix">
            <VoiceSelector value={voice} onChange={setVoice} language={lang} />
          </Card>

          <Card title="Paramètres">
            <div className="space-y-4">
              <Select
                label="Langue"
                value={lang}
                onChange={(e) => setLang(e.target.value)}
                options={[
                  { value: 'fr', label: 'Français' },
                  { value: 'en', label: 'English' },
                ]}
              />
              <Slider label="Vitesse" value={speed} min={0.5} max={2.0} step={0.1}
                onChange={setSpeed} unit="x" />
            </div>
          </Card>
        </div>
      </div>

      <ToastContainer />
    </div>
  )
}
