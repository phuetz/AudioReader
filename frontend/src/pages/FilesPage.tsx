import { useEffect, useState, useMemo } from 'react'
import {
  FolderOpen,
  Download,
  Play,
  Search,
  SortAsc,
  SortDesc,
  Filter,
  Grid,
  List,
  ListPlus,
  Clock,
  HardDrive,
  FileAudio,
} from 'lucide-react'
import Card from '../components/ui/Card'
import Badge from '../components/ui/Badge'
import AudioPlayer from '../components/audio/AudioPlayer'
import { ToastContainer, toast } from '../components/ui/Toast'
import { getFiles } from '../api/endpoints'
import type { FileInfo } from '../api/types'
import { usePlaylistStore } from '../stores/usePlaylistStore'
import { useAudioStore } from '../stores/useAudioStore'

type SortField = 'name' | 'size' | 'date'
type SortOrder = 'asc' | 'desc'
type ViewMode = 'list' | 'grid'

const FILE_TYPES = [
  { value: '', label: 'Tous' },
  { value: 'wav', label: 'WAV' },
  { value: 'mp3', label: 'MP3' },
  { value: 'm4a', label: 'M4A' },
  { value: 'm4b', label: 'M4B' },
]

export default function FilesPage() {
  const [files, setFiles] = useState<FileInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedFile, setSelectedFile] = useState<string | null>(null)

  // Filtering & sorting
  const [search, setSearch] = useState('')
  const [filterType, setFilterType] = useState('')
  const [sortField, setSortField] = useState<SortField>('date')
  const [sortOrder, setSortOrder] = useState<SortOrder>('desc')
  const [viewMode, setViewMode] = useState<ViewMode>('list')

  // Playlist integration
  const { addTrack } = usePlaylistStore()
  const { setCurrentUrl, setIsPlaying } = useAudioStore()

  useEffect(() => {
    setLoading(true)
    getFiles()
      .then((d) => {
        setFiles(d.files)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  const filteredFiles = useMemo(() => {
    let result = files

    // Filter by search
    if (search) {
      const lowerSearch = search.toLowerCase()
      result = result.filter((f) => f.name.toLowerCase().includes(lowerSearch))
    }

    // Filter by type
    if (filterType) {
      result = result.filter((f) => f.name.toLowerCase().endsWith(`.${filterType}`))
    }

    // Sort
    result = [...result].sort((a, b) => {
      let cmp = 0
      switch (sortField) {
        case 'name':
          cmp = a.name.localeCompare(b.name)
          break
        case 'size':
          cmp = a.size_mb - b.size_mb
          break
        case 'date':
          cmp = new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
          break
      }
      return sortOrder === 'asc' ? cmp : -cmp
    })

    return result
  }, [files, search, filterType, sortField, sortOrder])

  const formatSize = (mb: number) =>
    mb < 1 ? `${(mb * 1024).toFixed(0)} KB` : `${mb.toFixed(1)} MB`

  const formatDate = (iso: string) =>
    new Date(iso).toLocaleDateString('fr-FR', {
      day: '2-digit',
      month: 'short',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })

  const toggleSort = (field: SortField) => {
    if (sortField === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')
    } else {
      setSortField(field)
      setSortOrder('desc')
    }
  }

  const handlePlay = (file: FileInfo) => {
    setCurrentUrl(file.download_url)
    setIsPlaying(true)
    setSelectedFile(file.download_url)
  }

  const handleAddToPlaylist = (file: FileInfo) => {
    addTrack({ url: file.download_url, title: file.name })
    toast.success(`"${file.name}" ajouté à la playlist`)
  }

  const totalSize = useMemo(
    () => files.reduce((sum, f) => sum + f.size_mb, 0),
    [files]
  )

  const SortButton = ({
    field,
    label,
  }: {
    field: SortField
    label: string
  }) => (
    <button
      onClick={() => toggleSort(field)}
      className={`flex items-center gap-1 px-2 py-1 text-xs rounded transition-colors ${
        sortField === field
          ? 'bg-accent/20 text-accent'
          : 'text-muted hover:text-primary'
      }`}
    >
      {label}
      {sortField === field &&
        (sortOrder === 'asc' ? (
          <SortAsc className="w-3 h-3" />
        ) : (
          <SortDesc className="w-3 h-3" />
        ))}
    </button>
  )

  return (
    <div className="space-y-6 max-w-6xl">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold text-primary flex items-center gap-2">
          <FolderOpen className="w-5 h-5 text-accent" />
          Fichiers générés
        </h1>
        <div className="flex items-center gap-4 text-sm text-muted">
          <span className="flex items-center gap-1">
            <FileAudio className="w-4 h-4" />
            {files.length} fichiers
          </span>
          <span className="flex items-center gap-1">
            <HardDrive className="w-4 h-4" />
            {totalSize.toFixed(1)} MB
          </span>
        </div>
      </div>

      {/* Current player */}
      {selectedFile && <AudioPlayer url={selectedFile} title="Lecture en cours" />}

      {/* Toolbar */}
      <div className="flex flex-wrap items-center gap-3">
        {/* Search */}
        <div className="relative flex-1 min-w-[200px]">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Rechercher..."
            className="w-full pl-10 pr-4 py-2 rounded-lg bg-panel border border-border text-primary text-sm placeholder:text-muted"
          />
        </div>

        {/* Type filter */}
        <div className="flex items-center gap-1">
          <Filter className="w-4 h-4 text-muted" />
          <select
            value={filterType}
            onChange={(e) => setFilterType(e.target.value)}
            className="px-2 py-2 rounded-lg bg-panel border border-border text-primary text-sm cursor-pointer"
          >
            {FILE_TYPES.map((t) => (
              <option key={t.value} value={t.value}>
                {t.label}
              </option>
            ))}
          </select>
        </div>

        {/* Sort buttons */}
        <div className="flex items-center gap-1 border-l border-border pl-3">
          <SortButton field="date" label="Date" />
          <SortButton field="name" label="Nom" />
          <SortButton field="size" label="Taille" />
        </div>

        {/* View toggle */}
        <div className="flex items-center gap-1 border-l border-border pl-3">
          <button
            onClick={() => setViewMode('list')}
            className={`p-1.5 rounded transition-colors ${
              viewMode === 'list' ? 'text-accent' : 'text-muted hover:text-primary'
            }`}
          >
            <List className="w-4 h-4" />
          </button>
          <button
            onClick={() => setViewMode('grid')}
            className={`p-1.5 rounded transition-colors ${
              viewMode === 'grid' ? 'text-accent' : 'text-muted hover:text-primary'
            }`}
          >
            <Grid className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Files */}
      <Card>
        {loading ? (
          <div className="py-12 text-center">
            <div className="w-8 h-8 border-2 border-accent border-t-transparent rounded-full animate-spin mx-auto" />
            <p className="text-muted text-sm mt-3">Chargement...</p>
          </div>
        ) : filteredFiles.length === 0 ? (
          <div className="py-12 text-center">
            <FolderOpen className="w-12 h-12 text-muted mx-auto mb-3" />
            <p className="text-secondary">
              {search || filterType ? 'Aucun fichier trouvé' : 'Aucun fichier généré'}
            </p>
            <p className="text-xs text-muted mt-1">
              {search || filterType
                ? 'Essayez avec d\'autres critères'
                : 'Générez votre premier audiobook !'}
            </p>
          </div>
        ) : viewMode === 'list' ? (
          /* List View */
          <div className="divide-y divide-border/50">
            {filteredFiles.map((f) => (
              <div
                key={f.id}
                className="flex items-center gap-4 py-3 hover:bg-panel/50 px-2 rounded group"
              >
                <button
                  onClick={() => handlePlay(f)}
                  className="text-cyan hover:text-accent transition-colors cursor-pointer"
                  title="Lire"
                >
                  <Play className="w-4 h-4" />
                </button>

                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-primary truncate">{f.name}</p>
                  <div className="flex items-center gap-3 mt-0.5">
                    <span className="text-xs text-muted flex items-center gap-1">
                      <HardDrive className="w-3 h-3" />
                      {formatSize(f.size_mb)}
                    </span>
                    <span className="text-xs text-muted flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      {formatDate(f.created_at)}
                    </span>
                  </div>
                </div>

                <Badge color="cyan">{f.name.split('.').pop()?.toUpperCase()}</Badge>

                <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button
                    onClick={() => handleAddToPlaylist(f)}
                    className="p-1.5 text-muted hover:text-accent transition-colors"
                    title="Ajouter à la playlist"
                  >
                    <ListPlus className="w-4 h-4" />
                  </button>
                  <a
                    href={f.download_url}
                    download
                    className="p-1.5 text-muted hover:text-green transition-colors"
                    title="Télécharger"
                  >
                    <Download className="w-4 h-4" />
                  </a>
                </div>
              </div>
            ))}
          </div>
        ) : (
          /* Grid View */
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 p-2">
            {filteredFiles.map((f) => (
              <div
                key={f.id}
                className="bg-panel border border-border rounded-lg p-3 hover:border-accent/50 transition-colors group"
              >
                <div className="flex items-center justify-between mb-2">
                  <Badge color="cyan">
                    {f.name.split('.').pop()?.toUpperCase()}
                  </Badge>
                  <span className="text-xs text-muted">{formatSize(f.size_mb)}</span>
                </div>
                <p className="text-sm font-medium text-primary truncate mb-1" title={f.name}>
                  {f.name}
                </p>
                <p className="text-xs text-muted mb-3">{formatDate(f.created_at)}</p>
                <div className="flex items-center gap-1">
                  <button
                    onClick={() => handlePlay(f)}
                    className="flex-1 flex items-center justify-center gap-1 py-1.5 bg-accent/10 text-accent rounded text-xs hover:bg-accent/20 transition-colors"
                  >
                    <Play className="w-3 h-3" />
                    Lire
                  </button>
                  <button
                    onClick={() => handleAddToPlaylist(f)}
                    className="p-1.5 text-muted hover:text-accent transition-colors"
                    title="Playlist"
                  >
                    <ListPlus className="w-4 h-4" />
                  </button>
                  <a
                    href={f.download_url}
                    download
                    className="p-1.5 text-muted hover:text-green transition-colors"
                    title="Télécharger"
                  >
                    <Download className="w-4 h-4" />
                  </a>
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>

      {/* Stats */}
      {!loading && filteredFiles.length > 0 && filteredFiles.length !== files.length && (
        <p className="text-sm text-muted text-center">
          {filteredFiles.length} fichier{filteredFiles.length > 1 ? 's' : ''} sur {files.length}
        </p>
      )}

      <ToastContainer />
    </div>
  )
}
