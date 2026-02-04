import Badge from '../ui/Badge'

export default function GenderBadge({ gender }: { gender: string }) {
  if (gender === 'F') return <Badge color="accent">F</Badge>
  if (gender === 'M') return <Badge color="cyan">M</Badge>
  return <Badge color="muted">?</Badge>
}
