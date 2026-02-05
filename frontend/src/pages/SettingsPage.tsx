import { Settings, Sparkles, Volume2, FileText, Clock, Shield } from 'lucide-react'
import Card from '../components/ui/Card'
import Select from '../components/ui/Select'
import Slider from '../components/ui/Slider'
import Toggle from '../components/ui/Toggle'
import Input from '../components/ui/Input'
import VoiceSelector from '../components/voices/VoiceSelector'
import { ToastContainer } from '../components/ui/Toast'
import { useSettingsStore } from '../stores/useSettingsStore'
import type { LLMProvider, NarrationStyle, SubtitleFormat } from '../api/types'

export default function SettingsPage() {
  const s = useSettingsStore()

  return (
    <div className="space-y-6 max-w-4xl">
      <h1 className="text-xl font-semibold text-primary flex items-center gap-2">
        <Settings className="w-5 h-5 text-accent" />
        Paramètres
      </h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Colonne gauche */}
        <div className="space-y-6">
          {/* Langue et voix */}
          <Card title="Langue et voix">
            <div className="space-y-4">
              <Select
                label="Langue par défaut"
                value={s.language}
                onChange={(e) => s.setLanguage(e.target.value)}
                options={[
                  { value: 'fr', label: 'Français' },
                  { value: 'en', label: 'English' },
                ]}
              />
              <div>
                <label className="block text-xs font-medium text-secondary mb-1.5">Voix par défaut</label>
                <VoiceSelector value={s.defaultVoice} onChange={s.setDefaultVoice} language={s.language} />
              </div>
            </div>
          </Card>

          {/* Audio */}
          <Card title="Audio">
            <div className="space-y-4">
              <Slider label="Vitesse" value={s.speed} min={0.5} max={2.0} step={0.1}
                onChange={s.setSpeed} unit="x" />

              <Select
                label="Style de narration"
                value={s.style}
                onChange={(e) => s.setStyle(e.target.value as NarrationStyle)}
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
            </div>
          </Card>

          {/* Pipeline HQ */}
          <Card title="Pipeline HQ">
            <div className="space-y-4">
              <Toggle label="Analyse des émotions" checked={s.enableEmotions}
                onChange={s.setEnableEmotions}
                description="Détecte et adapte la voix selon les émotions du texte" />
              <Toggle label="Multi-voix automatique" checked={s.enableMultiVoice}
                onChange={s.setEnableMultiVoice}
                description="Attribue automatiquement des voix différentes aux personnages" />
              <Toggle label="Mastering audio" checked={s.enableMastering}
                onChange={s.setEnableMastering}
                description="Normalisation et compression professionnelle" />
            </div>
          </Card>

          {/* Timing & Prosodie */}
          <Card title="Timing & Prosodie" icon={<Clock className="w-4 h-4" />}>
            <div className="space-y-4">
              <Toggle label="Humanisation du timing" checked={s.enableTimingHumanization}
                onChange={s.setEnableTimingHumanization}
                description="Variations naturelles des pauses pour éviter l'effet robotique" />

              {s.enableTimingHumanization && (
                <Slider label="Variation des pauses" value={s.pauseVariation} min={0} max={0.5} step={0.05}
                  onChange={s.setPauseVariation} unit="%" displayValue={(v) => `±${Math.round(v * 100)}%`} />
              )}

              <Toggle label="Contours d'intonation" checked={s.enableIntonationContours}
                onChange={s.setEnableIntonationContours}
                description="Montée/descente naturelle pour questions, exclamations, etc." />
            </div>
          </Card>
        </div>

        {/* Colonne droite */}
        <div className="space-y-6">
          {/* LLM Enhancement */}
          <Card title="Amélioration LLM" icon={<Sparkles className="w-4 h-4" />}>
            <div className="space-y-4">
              <Toggle label="Activer l'amélioration LLM" checked={s.enableLLMEnhance}
                onChange={s.setEnableLLMEnhance}
                description="Utilise un LLM pour améliorer l'analyse du texte" />

              {s.enableLLMEnhance && (
                <>
                  <Select
                    label="Provider LLM"
                    value={s.llmProvider}
                    onChange={(e) => s.setLLMProvider(e.target.value as LLMProvider)}
                    options={[
                      { value: 'ollama', label: 'Ollama (local)' },
                      { value: 'gemini', label: 'Gemini 2.5 Flash' },
                      { value: 'openai', label: 'OpenAI' },
                      { value: 'anthropic', label: 'Anthropic Claude' },
                    ]}
                  />
                  <Input
                    label="Modèle (optionnel)"
                    value={s.llmModel}
                    onChange={(e) => s.setLLMModel(e.target.value)}
                    placeholder="Auto-sélection si vide"
                  />
                  <p className="text-xs text-muted">
                    Ollama: llama3.2 | Gemini: gemini-2.5-flash | OpenAI: gpt-4o-mini
                  </p>
                </>
              )}
            </div>
          </Card>

          {/* Effets Sonores */}
          <Card title="Effets Sonores" icon={<Volume2 className="w-4 h-4" />}>
            <div className="space-y-4">
              <Toggle label="Activer les effets sonores" checked={s.enableSoundEffects}
                onChange={s.setEnableSoundEffects}
                description="Ajoute des effets de transition et d'ambiance" />

              {s.enableSoundEffects && (
                <Slider label="Intensité" value={s.soundEffectsIntensity} min={0} max={1} step={0.1}
                  onChange={s.setSoundEffectsIntensity} unit=""
                  displayValue={(v) => v < 0.3 ? 'Subtil' : v < 0.6 ? 'Modéré' : 'Intense'} />
              )}
            </div>
          </Card>

          {/* Sous-titres */}
          <Card title="Sous-titres" icon={<FileText className="w-4 h-4" />}>
            <div className="space-y-4">
              <Toggle label="Générer les sous-titres" checked={s.enableSubtitles}
                onChange={s.setEnableSubtitles}
                description="Crée un fichier de sous-titres synchronisé avec l'audio" />

              {s.enableSubtitles && (
                <Select
                  label="Format"
                  value={s.subtitleFormat}
                  onChange={(e) => s.setSubtitleFormat(e.target.value as SubtitleFormat)}
                  options={[
                    { value: 'srt', label: 'SRT (SubRip)' },
                    { value: 'vtt', label: 'VTT (WebVTT)' },
                    { value: 'json', label: 'JSON (timestamps)' },
                  ]}
                />
              )}
            </div>
          </Card>

          {/* ACX Compliance */}
          <Card title="Conformité ACX/Audible" icon={<Shield className="w-4 h-4" />}>
            <div className="space-y-4">
              <Toggle label="Activer la conformité ACX" checked={s.enableACXCompliance}
                onChange={s.setEnableACXCompliance}
                description="Normalise l'audio selon les standards Audible" />

              {s.enableACXCompliance && (
                <>
                  <Slider label="Loudness cible (LUFS)" value={s.acxTargetLufs} min={-24} max={-14} step={0.5}
                    onChange={s.setACXTargetLufs} unit=" LUFS" />
                  <div className="text-xs text-muted space-y-1">
                    <p>Standards ACX :</p>
                    <ul className="list-disc list-inside pl-2">
                      <li>RMS : -23 dB à -18 dB</li>
                      <li>Peak : ≤ -3 dB</li>
                      <li>True Peak : ≤ -1 dB</li>
                      <li>Noise floor : ≤ -60 dB</li>
                    </ul>
                  </div>
                </>
              )}
            </div>
          </Card>
        </div>
      </div>

      <p className="text-xs text-muted">Les paramètres sont sauvegardés automatiquement dans le navigateur.</p>

      <ToastContainer />
    </div>
  )
}
