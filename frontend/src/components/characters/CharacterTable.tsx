import type { CharacterInfo } from '../../api/types'
import { useVoiceStore } from '../../stores/useVoiceStore'
import { useCharacterStore } from '../../stores/useCharacterStore'
import GenderBadge from './GenderBadge'

interface CharacterTableProps {
  characters: CharacterInfo[]
}

export default function CharacterTable({ characters }: CharacterTableProps) {
  const { voices } = useVoiceStore()
  const { voiceAssignments, assignVoice } = useCharacterStore()

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border text-left">
            <th className="py-2 pr-4 text-xs font-medium text-muted">Personnage</th>
            <th className="py-2 pr-4 text-xs font-medium text-muted">Genre</th>
            <th className="py-2 pr-4 text-xs font-medium text-muted">Dialogues</th>
            <th className="py-2 text-xs font-medium text-muted">Voix</th>
          </tr>
        </thead>
        <tbody>
          {characters.map((char) => (
            <tr key={char.name} className="border-b border-border/50 hover:bg-panel/50">
              <td className="py-2.5 pr-4 font-medium text-primary">{char.name}</td>
              <td className="py-2.5 pr-4"><GenderBadge gender={char.gender} /></td>
              <td className="py-2.5 pr-4 font-mono text-cyan">{char.dialogue_count}</td>
              <td className="py-2.5">
                <select
                  value={voiceAssignments[char.name] || char.suggested_voice || ''}
                  onChange={(e) => assignVoice(char.name, e.target.value)}
                  className="bg-panel border border-border rounded px-2 py-1 text-xs text-primary
                    focus:border-accent cursor-pointer"
                >
                  <option value="">Auto</option>
                  {voices.map(v => (
                    <option key={v.id} value={v.id}>{v.name} ({v.engine})</option>
                  ))}
                </select>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
