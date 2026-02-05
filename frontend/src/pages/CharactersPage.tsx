import { useState, useEffect } from 'react'
import { Users, Search, Sparkles } from 'lucide-react'
import Card from '../components/ui/Card'
import Button from '../components/ui/Button'
import Toggle from '../components/ui/Toggle'
import Select from '../components/ui/Select'
import CharacterTable from '../components/characters/CharacterTable'
import { ToastContainer, toast } from '../components/ui/Toast'
import { useCharacterStore } from '../stores/useCharacterStore'
import { useVoiceStore } from '../stores/useVoiceStore'
import { useSettingsStore } from '../stores/useSettingsStore'
import { analyzeText } from '../api/endpoints'
import type { AnalysisResult, LLMProvider } from '../api/types'

export default function CharactersPage() {
  const [text, setText] = useState('')
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null)
  const [loading, setLoading] = useState(false)
  const { setCharacters } = useCharacterStore()
  const { fetchVoices } = useVoiceStore()
  const settings = useSettingsStore()

  // LLM options
  const [useLLM, setUseLLM] = useState(settings.enableLLMEnhance)
  const [llmProvider, setLLMProvider] = useState<LLMProvider>(settings.llmProvider)

  useEffect(() => { fetchVoices() }, [fetchVoices])

  const handleAnalyze = async () => {
    if (!text.trim()) { toast.error('Entrez du texte à analyser'); return }
    setLoading(true)
    try {
      const result = await analyzeText({ text, language: settings.language })
      setAnalysis(result)
      setCharacters(result.characters)
      const msg = useLLM
        ? `${result.characters.length} personnages détectés (validation LLM activée)`
        : `${result.characters.length} personnages détectés`
      toast.success(msg)
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Erreur analyse')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6 max-w-5xl">
      <h1 className="text-xl font-semibold text-primary flex items-center gap-2">
        <Users className="w-5 h-5 text-accent" />
        Personnages
      </h1>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <Card title="Texte à analyser">
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Collez un extrait de votre livre avec des dialogues..."
              rows={8}
              className="w-full px-3 py-2 rounded-lg bg-panel border border-border text-primary text-sm
                placeholder:text-muted focus:border-accent resize-y"
            />
            <Button onClick={handleAnalyze} loading={loading} icon={<Search className="w-4 h-4" />} className="mt-3">
              Détecter les personnages
            </Button>
          </Card>
        </div>

        {/* Options */}
        <div>
          <Card title="Options" icon={<Sparkles className="w-4 h-4" />}>
            <div className="space-y-4">
              <Toggle label="Validation LLM" checked={useLLM} onChange={setUseLLM}
                description="Utilise un LLM pour filtrer les faux positifs" />
              {useLLM && (
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
              <p className="text-xs text-muted">
                La validation LLM filtre les mots incorrectement identifiés comme personnages
                (ex: "coupé", "pointé").
              </p>
            </div>
          </Card>
        </div>
      </div>

      {analysis && (
        <>
          {/* Stats */}
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-surface border border-border rounded-xl p-4">
              <p className="text-xs text-muted">Personnages</p>
              <p className="text-xl font-semibold font-mono text-accent">{analysis.characters.length}</p>
            </div>
            <div className="bg-surface border border-border rounded-xl p-4">
              <p className="text-xs text-muted">Dialogues</p>
              <p className="text-xl font-semibold font-mono text-cyan">{analysis.dialogues.length}</p>
            </div>
            <div className="bg-surface border border-border rounded-xl p-4">
              <p className="text-xs text-muted">Mots</p>
              <p className="text-xl font-semibold font-mono text-primary">{analysis.word_count.toLocaleString()}</p>
            </div>
          </div>

          {/* Character table with voice assignment */}
          <Card title="Attribution des voix">
            <CharacterTable characters={analysis.characters} />
          </Card>

          {/* Emotions */}
          {analysis.emotions.length > 0 && (
            <Card title="Analyse émotionnelle">
              <div className="space-y-1">
                {analysis.emotions.map((e, i) => (
                  <div key={i} className="flex items-center gap-3 px-3 py-1.5 rounded hover:bg-panel text-sm">
                    <span className="text-xs font-mono text-cyan w-20">{e.emotion}</span>
                    <div className="flex-1 h-1 bg-border rounded-full">
                      <div className="h-full bg-accent rounded-full" style={{ width: `${e.intensity * 100}%` }} />
                    </div>
                    <span className="text-xs text-muted truncate max-w-48">{e.text}</span>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </>
      )}

      <ToastContainer />
    </div>
  )
}
