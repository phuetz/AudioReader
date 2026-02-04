import { useState } from 'react'
import { BookOpen, Play } from 'lucide-react'
import Card from '../components/ui/Card'
import Button from '../components/ui/Button'
import DropZone from '../components/ui/DropZone'
import Select from '../components/ui/Select'
import Toggle from '../components/ui/Toggle'
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
import type { NarrationStyle } from '../api/types'

export default function BookConversionPage() {
  const { upload, uploading, result: uploadResult } = useFileUpload()
  const settings = useSettingsStore()

  const [voice, setVoice] = useState(settings.defaultVoice)
  const [style, setStyle] = useState<NarrationStyle>(settings.style)
  const [format, setFormat] = useState<'wav' | 'mp3' | 'm4b'>('wav')
  const [emotions, setEmotions] = useState(settings.enableEmotions)
  const [multiVoice, setMultiVoice] = useState(settings.enableMultiVoice)
  const [mastering, setMastering] = useState(settings.enableMastering)
  const [title, setTitle] = useState('audiobook')
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

          <Card title="Options">
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
              <Toggle label="Mastering ACX" checked={mastering} onChange={setMastering}
                description="Normalisation audio professionnelle" />
            </div>
          </Card>

          <Card title="Format de sortie">
            <FormatSelector value={format} onChange={setFormat} />
          </Card>
        </div>
      </div>

      <ToastContainer />
    </div>
  )
}
