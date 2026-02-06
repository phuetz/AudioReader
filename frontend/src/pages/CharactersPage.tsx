import { useState, useEffect } from 'react'
import { Users, Search, Sparkles, Save, Trash2 } from 'lucide-react'
import Card from '../components/ui/Card'
import Button from '../components/ui/Button'
import Toggle from '../components/ui/Toggle'
import Select from '../components/ui/Select'
import CharacterTable from '../components/characters/CharacterTable'
import { ToastContainer, toast } from '../components/ui/Toast'
import { useCharacterStore } from '../stores/useCharacterStore'
import { useVoiceStore } from '../stores/useVoiceStore'
import { useSettingsStore } from '../stores/useSettingsStore'
import { analyzeText, getCharacterProfiles, createCharacterProfile, deleteCharacterProfile } from '../api/endpoints'
import type { AnalysisResult, CharacterProfile, LLMProvider } from '../api/types'

type TabId = 'detection' | 'profiles'

export default function CharactersPage() {
  const [activeTab, setActiveTab] = useState<TabId>('detection')
  const [text, setText] = useState('')
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null)
  const [loading, setLoading] = useState(false)
  const { setCharacters } = useCharacterStore()
  const { fetchVoices } = useVoiceStore()
  const settings = useSettingsStore()

  // LLM options
  const [useLLM, setUseLLM] = useState(settings.enableLLMEnhance)
  const [llmProvider, setLLMProvider] = useState<LLMProvider>(settings.llmProvider)

  // Profiles state
  const [profiles, setProfiles] = useState<CharacterProfile[]>([])
  const [profilesLoading, setProfilesLoading] = useState(false)
  const [savingProfile, setSavingProfile] = useState(false)

  useEffect(() => { fetchVoices() }, [fetchVoices])

  useEffect(() => {
    if (activeTab === 'profiles') fetchProfiles()
  }, [activeTab])

  const fetchProfiles = async () => {
    try {
      setProfilesLoading(true)
      const data = await getCharacterProfiles()
      setProfiles(data.profiles)
    } catch {
      toast.error('Erreur chargement des profils')
    } finally {
      setProfilesLoading(false)
    }
  }

  const saveAsProfile = async (name: string, voiceId: string, gender: string) => {
    try {
      setSavingProfile(true)
      await createCharacterProfile({ name, voice_id: voiceId, gender })
      toast.success(`Profil "${name}" sauvegardé`)
      fetchProfiles()
    } catch {
      toast.error('Erreur sauvegarde du profil')
    } finally {
      setSavingProfile(false)
    }
  }

  const handleDeleteProfile = async (id: string) => {
    try {
      await deleteCharacterProfile(id)
      setProfiles(prev => prev.filter(p => p.id !== id))
      toast.success('Profil supprimé')
    } catch {
      toast.error('Erreur suppression du profil')
    }
  }

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

  const tabs: { id: TabId; label: string }[] = [
    { id: 'detection', label: 'Détection' },
    { id: 'profiles', label: 'Profils sauvegardés' },
  ]

  return (
    <div className="space-y-6 max-w-5xl">
      <h1 className="text-xl font-semibold text-primary flex items-center gap-2">
        <Users className="w-5 h-5 text-accent" />
        Personnages
      </h1>

      {/* Tabs */}
      <div className="flex gap-1 border-b border-border">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors -mb-px ${
              activeTab === tab.id
                ? 'text-accent border-accent'
                : 'text-muted border-transparent hover:text-primary'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {activeTab === 'detection' && (
        <>
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
                {analysis.characters.length > 0 && (
                  <div className="mt-4 pt-4 border-t border-border">
                    <Button
                      variant="secondary"
                      size="sm"
                      loading={savingProfile}
                      icon={<Save className="w-4 h-4" />}
                      onClick={() => {
                        analysis.characters.forEach(c => {
                          saveAsProfile(c.name, c.suggested_voice || 'ff_siwis', c.gender)
                        })
                      }}
                    >
                      Sauvegarder tous comme profils
                    </Button>
                  </div>
                )}
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
        </>
      )}

      {activeTab === 'profiles' && (
        <>
          {profilesLoading ? (
            <div className="flex items-center justify-center h-32">
              <div className="w-8 h-8 border-2 border-accent border-t-transparent rounded-full animate-spin" />
            </div>
          ) : profiles.length === 0 ? (
            <div className="text-center py-12 text-muted">
              <Users className="w-12 h-12 mx-auto mb-3 opacity-30" />
              <p>Aucun profil sauvegardé</p>
              <p className="text-xs mt-1">Détectez des personnages puis sauvegardez-les comme profils.</p>
            </div>
          ) : (
            <div className="space-y-2">
              {profiles.map(profile => (
                <div key={profile.id} className="flex items-center gap-4 p-4 bg-surface border border-border rounded-xl">
                  <div className="w-10 h-10 rounded-full bg-accent/20 flex items-center justify-center text-accent font-semibold text-sm shrink-0">
                    {profile.name.charAt(0).toUpperCase()}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-primary">{profile.name}</span>
                      {profile.gender && (
                        <span className="text-xs px-1.5 py-0.5 rounded bg-panel text-muted">{profile.gender}</span>
                      )}
                    </div>
                    <div className="flex items-center gap-3 mt-1">
                      <span className="text-xs text-muted font-mono">{profile.voice_id}</span>
                      {profile.aliases.length > 0 && (
                        <span className="text-xs text-muted">Alias: {profile.aliases.join(', ')}</span>
                      )}
                      {profile.morph && (
                        <span className="text-xs text-muted">
                          P:{profile.morph.pitch > 0 ? '+' : ''}{profile.morph.pitch}
                          {' '}F:{profile.morph.formant > 0 ? '+' : ''}{profile.morph.formant}
                          {' '}S:{profile.morph.speed}x
                        </span>
                      )}
                    </div>
                    {profile.personality_notes && (
                      <p className="text-xs text-muted mt-1 truncate">{profile.personality_notes}</p>
                    )}
                  </div>
                  <button
                    onClick={() => handleDeleteProfile(profile.id)}
                    className="p-1.5 text-muted hover:text-red-400 transition-colors"
                    title="Supprimer le profil"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </>
      )}

      <ToastContainer />
    </div>
  )
}
