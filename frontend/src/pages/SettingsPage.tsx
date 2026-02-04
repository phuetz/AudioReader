import { Settings } from 'lucide-react'
import Card from '../components/ui/Card'
import Select from '../components/ui/Select'
import Slider from '../components/ui/Slider'
import Toggle from '../components/ui/Toggle'
import VoiceSelector from '../components/voices/VoiceSelector'
import { ToastContainer } from '../components/ui/Toast'
import { useSettingsStore } from '../stores/useSettingsStore'
import type { NarrationStyle } from '../api/types'

export default function SettingsPage() {
  const s = useSettingsStore()

  return (
    <div className="space-y-6 max-w-3xl">
      <h1 className="text-xl font-semibold text-primary flex items-center gap-2">
        <Settings className="w-5 h-5 text-accent" />
        Paramètres
      </h1>

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

      <Card title="Pipeline HQ">
        <div className="space-y-4">
          <Toggle label="Analyse des émotions" checked={s.enableEmotions}
            onChange={s.setEnableEmotions}
            description="Détecte et adapte la voix selon les émotions du texte" />
          <Toggle label="Multi-voix automatique" checked={s.enableMultiVoice}
            onChange={s.setEnableMultiVoice}
            description="Attribue automatiquement des voix différentes aux personnages" />
          <Toggle label="Mastering ACX" checked={s.enableMastering}
            onChange={s.setEnableMastering}
            description="Normalisation audio conforme aux standards Audible" />
        </div>
      </Card>

      <p className="text-xs text-muted">Les paramètres sont sauvegardés automatiquement dans le navigateur.</p>

      <ToastContainer />
    </div>
  )
}
