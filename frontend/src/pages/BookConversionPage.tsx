import { useState } from 'react'
import { BookOpen, Play, ChevronDown, ChevronUp, Sparkles, Volume2, FileText, Shield, Clock } from 'lucide-react'
import Card from '../components/ui/Card'
import Button from '../components/ui/Button'
import DropZone from '../components/ui/DropZone'
import Select from '../components/ui/Select'
import Toggle from '../components/ui/Toggle'
import Slider from '../components/ui/Slider'
import ChapterList from '../components/book/ChapterList'
import FormatSelector from '../components/book/FormatSelector'
import BookMetadata from '../components/book/BookMetadata'
import VoiceSelector from '../components/voices/VoiceSelector'
import JobProgress from '../components/jobs/JobProgress'
import AudioPlayer from '../components/audio/AudioPlayer'
import { ToastContainer, toast } from '../components/ui/Toast'
import { useFileUpload } from '../hooks/useFileUpload'
import { useSettingsStore } from '../stores/useSettingsStore'
import { generateAudiobook } from '../api/endpoints'
import type { LLMProvider, NarrationStyle, SubtitleFormat } from '../api/types'

export default function BookConversionPage() {
  const { upload, uploading, result: uploadResult } = useFileUpload()
  const settings = useSettingsStore()

  // Basic options (from settings)
  const [voice, setVoice] = useState(settings.defaultVoice)
  const [style, setStyle] = useState<NarrationStyle>(settings.style)
  const [format, setFormat] = useState<'wav' | 'mp3' | 'm4b'>('wav')
  const [emotions, setEmotions] = useState(settings.enableEmotions)
  const [multiVoice, setMultiVoice] = useState(settings.enableMultiVoice)
  const [mastering, setMastering] = useState(settings.enableMastering)
  const [title, setTitle] = useState('audiobook')

  // Advanced options (v5.0)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [enableLLM, setEnableLLM] = useState(settings.enableLLMEnhance)
  const [llmProvider, setLLMProvider] = useState<LLMProvider>(settings.llmProvider)
  const [enableSoundFX, setEnableSoundFX] = useState(settings.enableSoundEffects)
  const [soundFXIntensity, setSoundFXIntensity] = useState(settings.soundEffectsIntensity)
  const [enableSubtitles, setEnableSubtitles] = useState(settings.enableSubtitles)
  const [subtitleFormat, setSubtitleFormat] = useState<SubtitleFormat>(settings.subtitleFormat)
  const [enableTiming, setEnableTiming] = useState(settings.enableTimingHumanization)
  const [pauseVariation, setPauseVariation] = useState(settings.pauseVariation)
  const [enableIntonation, setEnableIntonation] = useState(settings.enableIntonationContours)
  const [enableACX, setEnableACX] = useState(settings.enableACXCompliance)
  const [acxLufs, setACXLufs] = useState(settings.acxTargetLufs)

  // Job state
  const [jobId, setJobId] = useState<string | null>(null)
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const handleFile = async (file: File) => {
    const res = await upload(file)
    if (res) {
      toast.success(`Fichier "${res.original_name}" uploadé`)
      setTitle(file.name.replace(/\.[^.]+$/, ''))
    }
  }

  const handleGenerate = async () => {
    if (!uploadResult?.file_id) { toast.error('Uploadez un fichier d\'abord'); return }
    setLoading(true)
    setAudioUrl(null)
    try {
      const res = await generateAudiobook({
        file_id: uploadResult.file_id,
        title,
        narrator_voice: voice,
        style,
        enable_emotions: emotions,
        enable_multi_voice: multiVoice,
        enable_mastering: mastering,
        language: settings.language,
        // v5.0 options
        enable_llm_enhance: enableLLM,
        llm_provider: llmProvider,
        enable_sound_effects: enableSoundFX,
        sound_effects_intensity: soundFXIntensity,
        enable_subtitles: enableSubtitles,
        subtitle_format: subtitleFormat,
        enable_timing_humanization: enableTiming,
        pause_variation: pauseVariation,
        enable_intonation_contours: enableIntonation,
        enable_acx_compliance: enableACX,
        acx_target_lufs: acxLufs,
      })
      setJobId(res.job_id)
      toast.info(`Job ${res.job_id} démarré`)
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Erreur')
    } finally {
      setLoading(false)
    }
  }

  const handleComplete = (result: Record<string, unknown>) => {
    const url = result.download_url as string
    if (url) setAudioUrl(url)
    toast.success('Audiobook généré !')
  }

  return (
    <div className="space-y-6 max-w-5xl">
      <h1 className="text-xl font-semibold text-primary flex items-center gap-2">
        <BookOpen className="w-5 h-5 text-accent" />
        Conversion livre
      </h1>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-4">
          {/* Upload */}
          <Card title="Fichier source">
            <DropZone
              accept=".md,.txt,.pdf,.epub"
              onFile={handleFile}
              loading={uploading}
              label="Déposez un fichier MD, PDF, EPUB ou TXT"
            />
            {uploadResult && (
              <div className="mt-4 space-y-3">
                <BookMetadata
                  wordCount={uploadResult.text_preview?.split(/\s+/).length || 0}
                  chapterCount={uploadResult.chapters?.length || 0}
                  textPreview={uploadResult.text_preview}
                />
                {uploadResult.chapters && <ChapterList chapters={uploadResult.chapters} />}
              </div>
            )}
          </Card>

          {/* Generate */}
          <Button onClick={handleGenerate} loading={loading} icon={<Play className="w-4 h-4" />}
            disabled={!uploadResult}>
            Générer l'audiobook
          </Button>

          {jobId && <JobProgress jobId={jobId} onComplete={handleComplete} />}
          {audioUrl && <AudioPlayer url={audioUrl} title={title} />}
        </div>

        {/* Settings */}
        <div className="space-y-4">
          <Card title="Voix narrateur">
            <VoiceSelector value={voice} onChange={setVoice} language={settings.language} />
          </Card>

          <Card title="Options de base">
            <div className="space-y-4">
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-secondary">Titre</label>
                <input
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg bg-panel border border-border text-primary text-sm"
                />
              </div>
              <Select
                label="Style de narration"
                value={style}
                onChange={(e) => setStyle(e.target.value as NarrationStyle)}
                options={[
                  { value: 'storytelling', label: 'Storytelling' },
                  { value: 'dramatic', label: 'Dramatique' },
                  { value: 'formal', label: 'Formel' },
                  { value: 'conversational', label: 'Conversationnel' },
                  { value: 'documentary', label: 'Documentaire' },
                  { value: 'intimate', label: 'Intime' },
                  { value: 'energetic', label: 'Énergique' },
                ]}
              />
              <Toggle label="Analyse émotions" checked={emotions} onChange={setEmotions} />
              <Toggle label="Multi-voix" checked={multiVoice} onChange={setMultiVoice}
                description="Attribution automatique des voix aux personnages" />
              <Toggle label="Mastering audio" checked={mastering} onChange={setMastering}
                description="Normalisation et compression professionnelle" />
            </div>
          </Card>

          <Card title="Format de sortie">
            <FormatSelector value={format} onChange={setFormat} />
          </Card>

          {/* Advanced Options Toggle */}
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="w-full flex items-center justify-between px-4 py-3 bg-surface border border-border rounded-xl text-sm font-medium text-secondary hover:text-primary transition-colors"
          >
            <span className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-accent" />
              Options avancées (v5.0)
            </span>
            {showAdvanced ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>

          {/* Advanced Options */}
          {showAdvanced && (
            <div className="space-y-4">
              {/* LLM Enhancement */}
              <Card title="Amélioration LLM" icon={<Sparkles className="w-4 h-4" />}>
                <div className="space-y-3">
                  <Toggle label="Activer LLM" checked={enableLLM} onChange={setEnableLLM}
                    description="Validation des personnages et analyse contextuelle" />
                  {enableLLM && (
                    <Select
                      label="Provider"
                      value={llmProvider}
                      onChange={(e) => setLLMProvider(e.target.value as LLMProvider)}
                      options={[
                        { value: 'ollama', label: 'Ollama (local)' },
                        { value: 'gemini', label: 'Gemini 2.5 Flash' },
                        { value: 'openai', label: 'OpenAI' },
                        { value: 'anthropic', label: 'Anthropic' },
                      ]}
                    />
                  )}
                </div>
              </Card>

              {/* Sound Effects */}
              <Card title="Effets Sonores" icon={<Volume2 className="w-4 h-4" />}>
                <div className="space-y-3">
                  <Toggle label="Activer effets" checked={enableSoundFX} onChange={setEnableSoundFX}
                    description="Transitions, ambiances contextuelles" />
                  {enableSoundFX && (
                    <Slider label="Intensité" value={soundFXIntensity} min={0} max={1} step={0.1}
                      onChange={setSoundFXIntensity}
                      displayValue={(v) => v < 0.3 ? 'Subtil' : v < 0.6 ? 'Modéré' : 'Intense'} />
                  )}
                </div>
              </Card>

              {/* Subtitles */}
              <Card title="Sous-titres" icon={<FileText className="w-4 h-4" />}>
                <div className="space-y-3">
                  <Toggle label="Générer sous-titres" checked={enableSubtitles} onChange={setEnableSubtitles} />
                  {enableSubtitles && (
                    <Select
                      label="Format"
                      value={subtitleFormat}
                      onChange={(e) => setSubtitleFormat(e.target.value as SubtitleFormat)}
                      options={[
                        { value: 'srt', label: 'SRT' },
                        { value: 'vtt', label: 'VTT' },
                        { value: 'json', label: 'JSON' },
                      ]}
                    />
                  )}
                </div>
              </Card>

              {/* Timing */}
              <Card title="Timing & Prosodie" icon={<Clock className="w-4 h-4" />}>
                <div className="space-y-3">
                  <Toggle label="Humanisation timing" checked={enableTiming} onChange={setEnableTiming} />
                  {enableTiming && (
                    <Slider label="Variation pauses" value={pauseVariation} min={0} max={0.5} step={0.05}
                      onChange={setPauseVariation}
                      displayValue={(v) => `±${Math.round(v * 100)}%`} />
                  )}
                  <Toggle label="Contours intonation" checked={enableIntonation} onChange={setEnableIntonation}
                    description="Questions, exclamations naturelles" />
                </div>
              </Card>

              {/* ACX Compliance */}
              <Card title="Conformité ACX" icon={<Shield className="w-4 h-4" />}>
                <div className="space-y-3">
                  <Toggle label="Activer conformité" checked={enableACX} onChange={setEnableACX}
                    description="Standards Audible professionnels" />
                  {enableACX && (
                    <Slider label="Loudness (LUFS)" value={acxLufs} min={-24} max={-14} step={0.5}
                      onChange={setACXLufs} unit=" LUFS" />
                  )}
                </div>
              </Card>
            </div>
          )}
        </div>
      </div>

      <ToastContainer />
    </div>
  )
}
