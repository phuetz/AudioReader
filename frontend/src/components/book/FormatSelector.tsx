type Format = 'wav' | 'mp3' | 'm4b'

interface FormatSelectorProps {
  value: Format
  onChange: (f: Format) => void
}

const FORMATS: { value: Format; label: string; desc: string }[] = [
  { value: 'wav', label: 'WAV', desc: 'Non compressé, haute qualité' },
  { value: 'mp3', label: 'MP3', desc: 'Compressé, compatible partout' },
  { value: 'm4b', label: 'M4B', desc: 'Audiobook Apple, chapitres intégrés' },
]

export default function FormatSelector({ value, onChange }: FormatSelectorProps) {
  return (
    <div className="flex gap-2">
      {FORMATS.map((f) => (
        <button
          key={f.value}
          onClick={() => onChange(f.value)}
          className={`flex-1 px-3 py-2 rounded-lg border text-left transition-colors cursor-pointer ${
            value === f.value
              ? 'border-accent bg-accent/5'
              : 'border-border hover:border-accent/50 bg-panel'
          }`}
        >
          <span className="text-sm font-medium text-primary">{f.label}</span>
          <p className="text-xs text-muted mt-0.5">{f.desc}</p>
        </button>
      ))}
    </div>
  )
}
