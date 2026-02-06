import { useState, useEffect, useRef, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { Search, X, FileAudio, Briefcase, FolderKanban } from 'lucide-react'
import { useJobStore } from '../../stores/useJobStore'

interface SearchResult {
  type: 'job' | 'file' | 'page'
  label: string
  sublabel?: string
  route: string
}

const PAGES: SearchResult[] = [
  { type: 'page', label: 'Dashboard', route: '/' },
  { type: 'page', label: 'Texte rapide', route: '/text' },
  { type: 'page', label: 'Conversion livre', route: '/book' },
  { type: 'page', label: 'Personnages', route: '/characters' },
  { type: 'page', label: 'Clonage voix', route: '/cloning' },
  { type: 'page', label: 'ACX Compliance', route: '/acx' },
  { type: 'page', label: 'Corrections', route: '/corrections' },
  { type: 'page', label: 'Projets', route: '/projects' },
  { type: 'page', label: "File d'attente", route: '/queue' },
  { type: 'page', label: 'Labo Voix', route: '/voice-lab' },
  { type: 'page', label: 'Podcast', route: '/podcast' },
  { type: 'page', label: 'Fichiers', route: '/files' },
  { type: 'page', label: 'Paramètres', route: '/settings' },
]

export default function SearchBar() {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)
  const navigate = useNavigate()
  const jobs = useJobStore(s => s.jobs)

  // Cmd+K shortcut
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        setOpen(prev => !prev)
      }
      if (e.key === 'Escape') setOpen(false)
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  useEffect(() => {
    if (open) {
      inputRef.current?.focus()
      setQuery('')
    }
  }, [open])

  const results = useMemo(() => {
    if (!query.trim()) return []
    const q = query.toLowerCase()
    const matched: SearchResult[] = []

    // Pages
    for (const p of PAGES) {
      if (p.label.toLowerCase().includes(q)) {
        matched.push(p)
      }
    }

    // Jobs
    for (const j of jobs) {
      if (
        j.job_id.includes(q) ||
        (j.phase && j.phase.toLowerCase().includes(q)) ||
        j.status.includes(q)
      ) {
        matched.push({
          type: 'job',
          label: `Job ${j.job_id}`,
          sublabel: `${j.status} — ${j.phase || ''}`,
          route: `/review/${j.job_id}`,
        })
      }
    }

    return matched.slice(0, 10)
  }, [query, jobs])

  const handleSelect = (result: SearchResult) => {
    navigate(result.route)
    setOpen(false)
  }

  const iconForType = (type: string) => {
    switch (type) {
      case 'job': return <Briefcase className="w-4 h-4 text-accent" />
      case 'file': return <FileAudio className="w-4 h-4 text-cyan" />
      default: return <FolderKanban className="w-4 h-4 text-muted" />
    }
  }

  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-24" onClick={() => setOpen(false)}>
      <div className="absolute inset-0 bg-black/50" />
      <div
        className="relative w-full max-w-lg bg-surface border border-border rounded-xl shadow-2xl overflow-hidden"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-center gap-3 px-4 py-3 border-b border-border">
          <Search className="w-5 h-5 text-muted shrink-0" />
          <input
            ref={inputRef}
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="Rechercher pages, jobs..."
            className="flex-1 bg-transparent text-primary text-sm outline-none placeholder:text-muted"
          />
          <button onClick={() => setOpen(false)} className="text-muted hover:text-primary">
            <X className="w-4 h-4" />
          </button>
        </div>

        {results.length > 0 && (
          <div className="max-h-72 overflow-y-auto p-2">
            {results.map((r, i) => (
              <button
                key={i}
                onClick={() => handleSelect(r)}
                className="flex items-center gap-3 w-full px-3 py-2 rounded-lg text-left text-sm hover:bg-panel transition-colors"
              >
                {iconForType(r.type)}
                <div className="min-w-0">
                  <p className="text-primary truncate">{r.label}</p>
                  {r.sublabel && <p className="text-xs text-muted truncate">{r.sublabel}</p>}
                </div>
              </button>
            ))}
          </div>
        )}

        {query && results.length === 0 && (
          <div className="p-6 text-center text-sm text-muted">Aucun résultat</div>
        )}

        <div className="px-4 py-2 border-t border-border text-xs text-muted">
          <kbd className="px-1.5 py-0.5 rounded bg-panel border border-border font-mono">Esc</kbd> pour fermer
        </div>
      </div>
    </div>
  )
}
