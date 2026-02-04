import type { ReactNode } from 'react'

type Color = 'accent' | 'cyan' | 'green' | 'red' | 'muted'

const COLORS: Record<Color, string> = {
  accent: 'bg-accent/15 text-accent border-accent/30',
  cyan: 'bg-cyan/15 text-cyan border-cyan/30',
  green: 'bg-green/15 text-green border-green/30',
  red: 'bg-red/15 text-red border-red/30',
  muted: 'bg-border text-muted border-border',
}

interface BadgeProps {
  color?: Color
  children: ReactNode
}

export default function Badge({ color = 'muted', children }: BadgeProps) {
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-md text-xs font-medium border ${COLORS[color]}`}>
      {children}
    </span>
  )
}
