import { useState, useRef } from 'react'
import { Play, Pause } from 'lucide-react'
import Button from '../ui/Button'
import VoiceSelector from '../voices/VoiceSelector'
import { previewVoice } from '../../api/endpoints'
import { useSettingsStore } from '../../stores/useSettingsStore'

export default function VoiceComparison() {
  const language = useSettingsStore(s => s.language)
  const [voiceA, setVoiceA] = useState('ff_siwis')
  const [voiceB, setVoiceB] = useState('am_adam')
  const [text, setText] = useState("Bonjour, je suis une voix de synthèse. Comment trouvez-vous le résultat ?")
  const [loadingA, setLoadingA] = useState(false)
  const [loadingB, setLoadingB] = useState(false)
  const [playingA, setPlayingA] = useState(false)
  const [playingB, setPlayingB] = useState(false)
  const audioRefA = useRef<HTMLAudioElement>(null)
  const audioRefB = useRef<HTMLAudioElement>(null)

  const handlePlay = async (side: 'A' | 'B') => {
    const voice = side === 'A' ? voiceA : voiceB
    const setLoading = side === 'A' ? setLoadingA : setLoadingB
    const setPlaying = side === 'A' ? setPlayingA : setPlayingB
    const audioRef = side === 'A' ? audioRefA : audioRefB
    const otherRef = side === 'A' ? audioRefB : audioRefA
    const setOtherPlaying = side === 'A' ? setPlayingB : setPlayingA

    // Stop the other
    otherRef.current?.pause()
    setOtherPlaying(false)

    if (audioRef.current?.src && !audioRef.current.paused) {
      audioRef.current.pause()
      setPlaying(false)
      return
    }

    setLoading(true)
    try {
      const blob = await previewVoice(voice, text, 1.0, language)
      const url = URL.createObjectURL(blob)
      if (audioRef.current) {
        audioRef.current.src = url
        audioRef.current.onended = () => setPlaying(false)
        await audioRef.current.play()
        setPlaying(true)
      }
    } catch {
      // ignore
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-4">
      <audio ref={audioRefA} className="hidden" />
      <audio ref={audioRefB} className="hidden" />

      <div>
        <label className="block text-xs text-muted mb-1">Texte de test</label>
        <textarea
          value={text}
          onChange={e => setText(e.target.value)}
          rows={2}
          className="w-full px-3 py-2 rounded-lg bg-panel border border-border text-primary text-sm resize-y"
        />
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* Voice A */}
        <div className="space-y-3 p-4 bg-panel rounded-xl border border-border">
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold text-accent px-2 py-0.5 rounded-full bg-accent/20">A</span>
            <span className="text-sm font-medium text-primary">Voix A</span>
          </div>
          <VoiceSelector value={voiceA} onChange={setVoiceA} language={language} />
          <Button
            onClick={() => handlePlay('A')}
            loading={loadingA}
            icon={playingA ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            variant="primary"
            className="w-full"
          >
            {playingA ? 'Pause A' : 'Jouer A'}
          </Button>
        </div>

        {/* Voice B */}
        <div className="space-y-3 p-4 bg-panel rounded-xl border border-border">
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold text-cyan px-2 py-0.5 rounded-full bg-cyan/20">B</span>
            <span className="text-sm font-medium text-primary">Voix B</span>
          </div>
          <VoiceSelector value={voiceB} onChange={setVoiceB} language={language} />
          <Button
            onClick={() => handlePlay('B')}
            loading={loadingB}
            icon={playingB ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            variant="secondary"
            className="w-full"
          >
            {playingB ? 'Pause B' : 'Jouer B'}
          </Button>
        </div>
      </div>
    </div>
  )
}
