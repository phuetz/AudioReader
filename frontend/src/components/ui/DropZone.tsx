import { useCallback, useState, type DragEvent } from 'react'
import { Upload } from 'lucide-react'

interface DropZoneProps {
  accept?: string
  onFile: (file: File) => void
  label?: string
  loading?: boolean
}

export default function DropZone({ accept, onFile, label = 'Déposez un fichier ou cliquez', loading }: DropZoneProps) {
  const [dragOver, setDragOver] = useState(false)

  const handleDrop = useCallback((e: DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) onFile(file)
  }, [onFile])

  const handleClick = useCallback(() => {
    const input = document.createElement('input')
    input.type = 'file'
    if (accept) input.accept = accept
    input.onchange = () => {
      const file = input.files?.[0]
      if (file) onFile(file)
    }
    input.click()
  }, [accept, onFile])

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      onClick={handleClick}
      className={`flex flex-col items-center justify-center gap-3 p-8 rounded-xl border-2 border-dashed
        cursor-pointer transition-colors ${
          dragOver
            ? 'border-accent bg-accent/5'
            : 'border-border hover:border-accent/50 hover:bg-panel'
        } ${loading ? 'opacity-50 pointer-events-none' : ''}`}
    >
      {loading ? (
        <div className="w-8 h-8 border-2 border-accent border-t-transparent rounded-full animate-spin" />
      ) : (
        <Upload className="w-8 h-8 text-muted" />
      )}
      <p className="text-sm text-secondary">{label}</p>
      {accept && <p className="text-xs text-muted">Formats : {accept}</p>}
    </div>
  )
}
