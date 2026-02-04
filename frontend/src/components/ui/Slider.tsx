interface SliderProps {
  label?: string
  value: number
  min?: number
  max?: number
  step?: number
  onChange: (val: number) => void
  unit?: string
}

export default function Slider({ label, value, min = 0, max = 100, step = 1, onChange, unit }: SliderProps) {
  return (
    <div className="space-y-1.5">
      {label && (
        <div className="flex items-center justify-between">
          <label className="text-xs font-medium text-secondary">{label}</label>
          <span className="text-xs font-mono text-accent">{value}{unit}</span>
        </div>
      )}
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full h-1.5 bg-border rounded-full appearance-none cursor-pointer
          [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3.5
          [&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:bg-accent
          [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:cursor-pointer"
      />
    </div>
  )
}
