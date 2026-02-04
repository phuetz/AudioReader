import { FileText, Hash, BookOpen } from 'lucide-react'

interface BookMetadataProps {
  wordCount: number
  chapterCount: number
  textPreview?: string
}

export default function BookMetadata({ wordCount, chapterCount, textPreview }: BookMetadataProps) {
  return (
    <div className="space-y-3">
      <div className="flex gap-4">
        <div className="flex items-center gap-2 text-sm">
          <Hash className="w-4 h-4 text-cyan" />
          <span className="text-secondary">{wordCount.toLocaleString()} mots</span>
        </div>
        <div className="flex items-center gap-2 text-sm">
          <BookOpen className="w-4 h-4 text-accent" />
          <span className="text-secondary">{chapterCount} chapitres</span>
        </div>
      </div>
      {textPreview && (
        <div className="p-3 rounded-lg bg-panel border border-border">
          <div className="flex items-center gap-1.5 mb-1.5">
            <FileText className="w-3.5 h-3.5 text-muted" />
            <span className="text-xs text-muted">Aperçu</span>
          </div>
          <p className="text-xs text-secondary line-clamp-3">{textPreview}</p>
        </div>
      )}
    </div>
  )
}
