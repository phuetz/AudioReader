interface ToggleProps {
  label: string
  checked: boolean
  onChange: (val: boolean) => void
  description?: string
}

export default function Toggle({ label, checked, onChange, description }: ToggleProps) {
  return (
    <label className="flex items-center justify-between gap-4 cursor-pointer group">
      <div>
        <span className="text-sm text-primary group-hover:text-accent transition-colors">{label}</span>
        {description && <p className="text-xs text-muted mt-0.5">{description}</p>}
      </div>
      <button
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={`relative w-10 h-5.5 rounded-full transition-colors cursor-pointer ${
          checked ? 'bg-accent' : 'bg-border'
        }`}
      >
        <span
          className={`absolute top-0.5 left-0.5 w-4.5 h-4.5 bg-deep rounded-full transition-transform ${
            checked ? 'translate-x-[18px]' : ''
          }`}
        />
      </button>
    </label>
  )
}
