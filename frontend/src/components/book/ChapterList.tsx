import { BookOpen } from 'lucide-react'

interface ChapterListProps {
  chapters: string[]
}

export default function ChapterList({ chapters }: ChapterListProps) {
  if (chapters.length === 0) return null

  return (
    <div className="space-y-1">
      <p className="text-xs font-medium text-secondary mb-2">Chapitres detectes ({chapters.length})</p>
      <div className="max-h-48 overflow-y-auto space-y-0.5">
        {chapters.map((ch, i) => (
          <div key={i} className="flex items-center gap-2 px-3 py-1.5 rounded-md hover:bg-panel text-sm">
            <BookOpen className="w-3.5 h-3.5 text-muted shrink-0" />
            <span className="text-xs font-mono text-cyan w-6">{i + 1}</span>
            <span className="text-secondary truncate">{ch}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
