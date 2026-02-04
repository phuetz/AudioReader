interface ProgressProps {
  value: number  // 0-100
  label?: string
  color?: 'accent' | 'cyan' | 'green'
}

const COLORS = {
  accent: 'bg-accent',
  cyan: 'bg-cyan',
  green: 'bg-green',
}

export default function Progress({ value, label, color = 'cyan' }: ProgressProps) {
  return (
    <div className="space-y-1">
      {label && (
        <div className="flex justify-between text-xs">
          <span className="text-secondary">{label}</span>
          <span className="font-mono text-accent">{Math.round(value)}%</span>
        </div>
      )}
      <div className="h-1.5 bg-border rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-300 ${COLORS[color]}`}
          style={{ width: `${Math.min(100, Math.max(0, value))}%` }}
        />
      </div>
    </div>
  )
}
