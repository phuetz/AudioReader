import { useState } from 'react'
import { Play, Loader2, Volume2 } from 'lucide-react'
import type { CharacterInfo } from '../../api/types'
import { useVoiceStore } from '../../stores/useVoiceStore'
import { useCharacterStore } from '../../stores/useCharacterStore'
import { previewVoice } from '../../api/endpoints'
import GenderBadge from './GenderBadge'

interface CharacterTableProps {
  characters: CharacterInfo[]
}

const SAMPLE_DIALOGUES: Record<string, string> = {
  M: 'Bonjour, je suis un personnage masculin de cette histoire.',
  F: 'Bonjour, je suis un personnage féminin de cette histoire.',
  '?': 'Bonjour, je suis un personnage de cette histoire.',
}

export default function CharacterTable({ characters }: CharacterTableProps) {
  const { voices } = useVoiceStore()
  const { voiceAssignments, assignVoice } = useCharacterStore()
  const [previewingChar, setPreviewingChar] = useState<string | null>(null)
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [playingChar, setPlayingChar] = useState<string | null>(null)

  const handlePreview = async (char: CharacterInfo) => {
    const voiceId = voiceAssignments[char.name] || char.suggested_voice
    if (!voiceId) return

    setPreviewingChar(char.name)
    try {
      const text = SAMPLE_DIALOGUES[char.gender] || SAMPLE_DIALOGUES['?']
      const blob = await previewVoice(voiceId, text, 1.0, 'fr')
      const url = URL.createObjectURL(blob)

      // Clean up previous audio
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl)
      }

      setAudioUrl(url)
      setPlayingChar(char.name)

      // Play audio
      const audio = new Audio(url)
      audio.onended = () => {
        setPlayingChar(null)
      }
      audio.play()
    } catch (e) {
      console.error('Preview error:', e)
    } finally {
      setPreviewingChar(null)
    }
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border text-left">
            <th className="py-2 pr-4 text-xs font-medium text-muted">Personnage</th>
            <th className="py-2 pr-4 text-xs font-medium text-muted">Genre</th>
            <th className="py-2 pr-4 text-xs font-medium text-muted">Dialogues</th>
            <th className="py-2 pr-4 text-xs font-medium text-muted">Voix</th>
            <th className="py-2 text-xs font-medium text-muted w-20">Aperçu</th>
          </tr>
        </thead>
        <tbody>
          {characters.map((char) => {
            const currentVoice = voiceAssignments[char.name] || char.suggested_voice
            const isPreviewing = previewingChar === char.name
            const isPlaying = playingChar === char.name

            return (
              <tr key={char.name} className="border-b border-border/50 hover:bg-panel/50">
                <td className="py-2.5 pr-4">
                  <span className="font-medium text-primary">{char.name}</span>
                </td>
                <td className="py-2.5 pr-4">
                  <GenderBadge gender={char.gender} />
                </td>
                <td className="py-2.5 pr-4">
                  <span className="font-mono text-cyan">{char.dialogue_count}</span>
                </td>
                <td className="py-2.5 pr-4">
                  <select
                    value={currentVoice || ''}
                    onChange={(e) => assignVoice(char.name, e.target.value)}
                    className="bg-panel border border-border rounded px-2 py-1 text-xs text-primary
                      focus:border-accent cursor-pointer min-w-[140px]"
                  >
                    <option value="">Auto</option>
                    {voices.map((v) => (
                      <option key={v.id} value={v.id}>
                        {v.name} ({v.engine})
                      </option>
                    ))}
                  </select>
                </td>
                <td className="py-2.5">
                  <button
                    onClick={() => handlePreview(char)}
                    disabled={!currentVoice || isPreviewing}
                    className={`flex items-center justify-center w-8 h-8 rounded-lg transition-colors ${
                      isPlaying
                        ? 'bg-accent/20 text-accent'
                        : currentVoice
                        ? 'hover:bg-panel text-muted hover:text-accent'
                        : 'text-border cursor-not-allowed'
                    }`}
                    title={currentVoice ? 'Écouter un aperçu' : 'Sélectionnez une voix'}
                  >
                    {isPreviewing ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : isPlaying ? (
                      <Volume2 className="w-4 h-4" />
                    ) : (
                      <Play className="w-4 h-4" />
                    )}
                  </button>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>

      {characters.length === 0 && (
        <div className="py-8 text-center text-muted text-sm">
          Aucun personnage détecté
        </div>
      )}

      {/* Tip */}
      <p className="text-xs text-muted mt-3">
        Cliquez sur <Play className="w-3 h-3 inline" /> pour écouter un aperçu de la voix sélectionnée.
      </p>
    </div>
  )
}
