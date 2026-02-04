import type { VoiceInfo } from '../../api/types'
import Badge from '../ui/Badge'

interface VoiceCardProps {
  voice: VoiceInfo
  selected?: boolean
  onSelect?: () => void
  onPreview?: () => void
}

export default function VoiceCard({ voice, selected, onSelect, onPreview }: VoiceCardProps) {
  const engineColor = voice.engine === 'kokoro' ? 'green' : voice.engine === 'cloned' ? 'accent' : 'cyan'

  return (
    <div
      onClick={onSelect}
      className={`flex items-center gap-3 px-4 py-3 rounded-lg border cursor-pointer transition-colors ${
        selected
          ? 'border-accent bg-accent/5'
          : 'border-border hover:border-accent/50 bg-panel'
      }`}
    >
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-primary">{voice.name}</span>
          <Badge color={engineColor}>{voice.engine}</Badge>
        </div>
        <div className="flex items-center gap-2 mt-0.5">
          <span className="text-xs text-muted">{voice.gender === 'F' ? 'Féminine' : voice.gender === 'M' ? 'Masculine' : '?'}</span>
          <span className="text-xs text-muted">{voice.language.toUpperCase()}</span>
          <span className="text-xs text-muted">{voice.style}</span>
        </div>
      </div>
      {onPreview && (
        <button
          onClick={(e) => { e.stopPropagation(); onPreview() }}
          className="text-xs text-cyan hover:text-cyan-dim transition-colors cursor-pointer"
        >
          Preview
        </button>
      )}
    </div>
  )
}
