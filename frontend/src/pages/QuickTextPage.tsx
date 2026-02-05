import { useState, useRef, useCallback } from 'react'
import { Play, Zap, Radio, FileText, Sparkles, Volume2 } from 'lucide-react'
import Card from '../components/ui/Card'
import Button from '../components/ui/Button'
import Slider from '../components/ui/Slider'
import Select from '../components/ui/Select'
import Toggle from '../components/ui/Toggle'
import VoiceSelector from '../components/voices/VoiceSelector'
import AudioPlayer from '../components/audio/AudioPlayer'
import JobProgress from '../components/jobs/JobProgress'
import { ToastContainer, toast } from '../components/ui/Toast'
import { generateAudio, generatePreview } from '../api/endpoints'
import { useSettingsStore } from '../stores/useSettingsStore'
import type { NarrationStyle } from '../api/types'

const SAMPLE_TEXTS = {
  greeting: 'Bonjour et bienvenue dans AudioReader. Cette voix de synthèse est capable de produire des audiobooks de qualité professionnelle.',
  emotion: 'Victor regarda Marie avec une profonde tristesse. "Je t\'aimais," murmura-t-il, "mais tu ne m\'as jamais compris." Elle détourna le regard, les yeux brillants de larmes.',
  action: 'L\'explosion retentit dans la nuit. Sans réfléchir, il plongea derrière le mur de béton, le souffle coupé. Des éclats de verre volèrent au-dessus de sa tête.',
  technical: 'Le processeur quantique utilise des qubits en superposition pour effectuer des calculs parallèles. Cette technologie révolutionnaire pourrait transformer l\'intelligence artificielle.',
}

const NARRATION_STYLES: { value: NarrationStyle; label: string }[] = [
  { value: 'storytelling', label: 'Storytelling (défaut)' },
  { value: 'conversational', label: 'Conversationnel' },
  { value: 'dramatic', label: 'Dramatique' },
  { value: 'formal', label: 'Formel' },
  { value: 'intimate', label: 'Intime' },
  { value: 'energetic', label: 'Énergique' },
  { value: 'documentary', label: 'Documentaire' },
]

export default function QuickTextPage() {
  const { language, defaultVoice, speed, setSpeed, enableTimingHumanization, enableIntonationContours } = useSettingsStore()
  const [text, setText] = useState('')
  const [voice, setVoice] = useState(defaultVoice)
  const [lang, setLang] = useState(language)
  const [style, setStyle] = useState<NarrationStyle>('storytelling')
  const [jobId, setJobId] = useState<string | null>(null)
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  // Streaming state
  const [isStreaming, setIsStreaming] = useState(false)
  const [streamProgress, setStreamProgress] = useState(0)
  const eventSourceRef = useRef<EventSource | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const audioBuffersRef = useRef<AudioBuffer[]>([])

  // Advanced options
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [enableEmotions, setEnableEmotions] = useState(true)
  const [enableHumanization, setEnableHumanization] = useState(enableTimingHumanization)
  const [enableIntonation, setEnableIntonation] = useState(enableIntonationContours)

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

  const handleStreaming = useCallback(async () => {
    if (!text.trim()) { toast.error('Entrez du texte'); return }

    // Stop existing stream
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
    }

    setIsStreaming(true)
    setStreamProgress(0)
    audioBuffersRef.current = []

    try {
      // Create AudioContext for playback
      if (!audioContextRef.current) {
        audioContextRef.current = new AudioContext({ sampleRate: 24000 })
      }

      // Use fetch for POST-based SSE
      const response = await fetch('/api/v2/synthesize-stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: text.slice(0, 500),
          voice,
          speed,
          chunk_size_ms: 300,
        }),
      })

      if (!response.ok) throw new Error('Streaming non disponible')

      const reader = response.body?.getReader()
      if (!reader) throw new Error('Stream non supporté')

      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const data = JSON.parse(line.slice(6))
            if (data.type === 'progress') {
              setStreamProgress(data.progress_percent)
            } else if (data.type === 'done') {
              toast.success('Streaming terminé')
            } else if (data.type === 'error') {
              throw new Error(data.message)
            }
          } catch {
            // Ignore parse errors
          }
        }
      }
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Erreur streaming')
    } finally {
      setIsStreaming(false)
    }
  }, [text, voice, speed])

  const handleComplete = (result: Record<string, unknown>) => {
    const url = result.download_url as string
    if (url) setAudioUrl(url)
    toast.success('Audio généré !')
  }

  const loadSampleText = (key: keyof typeof SAMPLE_TEXTS) => {
    setText(SAMPLE_TEXTS[key])
    toast.info('Texte exemple chargé')
  }

  return (
    <div className="space-y-6 max-w-5xl">
      <h1 className="text-xl font-semibold text-primary flex items-center gap-2">
        <Volume2 className="w-5 h-5 text-accent" />
        Texte rapide
      </h1>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Text input */}
        <div className="lg:col-span-2 space-y-4">
          <Card title="Texte">
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Entrez votre texte ici..."
              rows={12}
              className="w-full px-3 py-2 rounded-lg bg-panel border border-border text-primary text-sm
                placeholder:text-muted focus:border-accent resize-y"
            />
            <div className="flex items-center justify-between mt-3">
              <div className="flex items-center gap-3 text-xs text-muted">
                <span>{text.length} caractères</span>
                <span>{text.split(/\s+/).filter(Boolean).length} mots</span>
              </div>
              {/* Sample text buttons */}
              <div className="flex items-center gap-1">
                <span className="text-xs text-muted mr-1">Exemples:</span>
                {Object.keys(SAMPLE_TEXTS).map((key) => (
                  <button
                    key={key}
                    onClick={() => loadSampleText(key as keyof typeof SAMPLE_TEXTS)}
                    className="px-2 py-1 text-xs bg-panel border border-border rounded hover:border-accent/50 transition-colors"
                  >
                    {key}
                  </button>
                ))}
              </div>
            </div>
          </Card>

          {/* Action buttons */}
          <div className="flex flex-wrap gap-2">
            <Button onClick={handleGenerate} loading={loading} icon={<Play className="w-4 h-4" />}>
              Générer
            </Button>
            <Button variant="secondary" onClick={handlePreview} loading={loading} icon={<Zap className="w-4 h-4" />}>
              Preview 30s
            </Button>
            <Button
              variant="ghost"
              onClick={handleStreaming}
              loading={isStreaming}
              icon={<Radio className="w-4 h-4" />}
              disabled={!text.trim()}
            >
              Streaming
            </Button>
          </div>

          {/* Streaming progress */}
          {isStreaming && (
            <Card>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-secondary">Streaming en cours...</span>
                  <span className="text-accent font-mono">{streamProgress}%</span>
                </div>
                <div className="h-2 bg-surface rounded-full overflow-hidden">
                  <div
                    className="h-full bg-accent transition-all duration-300"
                    style={{ width: `${streamProgress}%` }}
                  />
                </div>
              </div>
            </Card>
          )}

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
              <Slider
                label="Vitesse"
                value={speed}
                min={0.5}
                max={2.0}
                step={0.1}
                onChange={setSpeed}
                displayValue={(v) => `${v.toFixed(1)}x`}
              />
              <Select
                label="Style de narration"
                value={style}
                onChange={(e) => setStyle(e.target.value as NarrationStyle)}
                options={NARRATION_STYLES}
              />
            </div>
          </Card>

          {/* Advanced options */}
          <Card
            title="Options avancées"
            icon={<Sparkles className="w-4 h-4" />}
          >
            <div className="space-y-4">
              <button
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="text-sm text-accent hover:underline"
              >
                {showAdvanced ? 'Masquer' : 'Afficher'} les options
              </button>

              {showAdvanced && (
                <div className="space-y-3 pt-2 border-t border-border">
                  <Toggle
                    label="Émotions automatiques"
                    checked={enableEmotions}
                    onChange={setEnableEmotions}
                    description="Détection et adaptation émotionnelle"
                  />
                  <Toggle
                    label="Humanisation du timing"
                    checked={enableHumanization}
                    onChange={setEnableHumanization}
                    description="Variations naturelles des pauses"
                  />
                  <Toggle
                    label="Contours d'intonation"
                    checked={enableIntonation}
                    onChange={setEnableIntonation}
                    description="Montée/descente mélodique"
                  />
                </div>
              )}
            </div>
          </Card>

          {/* Tips */}
          <Card title="Conseils">
            <ul className="space-y-2 text-xs text-muted">
              <li className="flex items-start gap-2">
                <FileText className="w-3.5 h-3.5 text-accent shrink-0 mt-0.5" />
                <span>Pour les dialogues, utilisez des guillemets français « »</span>
              </li>
              <li className="flex items-start gap-2">
                <Sparkles className="w-3.5 h-3.5 text-accent shrink-0 mt-0.5" />
                <span>Le style dramatique ajoute plus de variations prosodiques</span>
              </li>
              <li className="flex items-start gap-2">
                <Radio className="w-3.5 h-3.5 text-accent shrink-0 mt-0.5" />
                <span>Le streaming permet d'écouter en temps réel (limité à 500 caractères)</span>
              </li>
            </ul>
          </Card>
        </div>
      </div>

      <ToastContainer />
    </div>
  )
}
