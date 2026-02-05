import { useState, useEffect } from 'react'
import {
  FolderKanban,
  Plus,
  Trash2,
  Edit3,
  Calendar,
  FileText,
  Settings2,
  ChevronRight,
  Search,
} from 'lucide-react'
import Card from '../components/ui/Card'
import Button from '../components/ui/Button'
import Input from '../components/ui/Input'
import { ToastContainer, toast } from '../components/ui/Toast'
import { getProjects, createProject, deleteProject } from '../api/endpoints'
import type { ProjectInfo } from '../api/types'

export default function ProjectsPage() {
  const [projects, setProjects] = useState<ProjectInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')

  // Create project form
  const [showCreateForm, setShowCreateForm] = useState(false)
  const [newName, setNewName] = useState('')
  const [newDescription, setNewDescription] = useState('')
  const [creating, setCreating] = useState(false)

  const loadProjects = async () => {
    setLoading(true)
    try {
      const data = await getProjects()
      setProjects(data.projects)
    } catch (e) {
      toast.error('Erreur chargement des projets')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadProjects()
  }, [])

  const handleCreate = async () => {
    if (!newName.trim()) {
      toast.error('Nom du projet requis')
      return
    }
    setCreating(true)
    try {
      const project = await createProject(newName, newDescription || undefined)
      toast.success(`Projet "${project.name}" créé`)
      setNewName('')
      setNewDescription('')
      setShowCreateForm(false)
      loadProjects()
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Erreur création')
    } finally {
      setCreating(false)
    }
  }

  const handleDelete = async (project: ProjectInfo) => {
    if (!confirm(`Supprimer le projet "${project.name}" ?`)) return
    try {
      await deleteProject(project.id)
      toast.success('Projet supprimé')
      loadProjects()
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Erreur suppression')
    }
  }

  const formatDate = (iso: string) => {
    const d = new Date(iso)
    return d.toLocaleDateString('fr-FR', {
      day: 'numeric',
      month: 'short',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  const filteredProjects = projects.filter(
    (p) =>
      p.name.toLowerCase().includes(search.toLowerCase()) ||
      (p.description && p.description.toLowerCase().includes(search.toLowerCase()))
  )

  return (
    <div className="space-y-6 max-w-5xl">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold text-primary flex items-center gap-2">
          <FolderKanban className="w-5 h-5 text-accent" />
          Projets
        </h1>
        <Button
          onClick={() => setShowCreateForm(!showCreateForm)}
          icon={<Plus className="w-4 h-4" />}
        >
          Nouveau projet
        </Button>
      </div>

      {/* Create Form */}
      {showCreateForm && (
        <Card title="Nouveau projet">
          <div className="space-y-4">
            <Input
              label="Nom du projet"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder="Mon audiobook"
            />
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-secondary">Description</label>
              <textarea
                value={newDescription}
                onChange={(e) => setNewDescription(e.target.value)}
                placeholder="Description optionnelle..."
                rows={2}
                className="w-full px-3 py-2 rounded-lg bg-panel border border-border text-primary text-sm resize-none"
              />
            </div>
            <div className="flex gap-2">
              <Button onClick={handleCreate} loading={creating}>
                Créer
              </Button>
              <Button variant="ghost" onClick={() => setShowCreateForm(false)}>
                Annuler
              </Button>
            </div>
          </div>
        </Card>
      )}

      {/* Search */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted" />
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Rechercher un projet..."
          className="w-full pl-10 pr-4 py-2 rounded-lg bg-panel border border-border text-primary text-sm placeholder:text-muted"
        />
      </div>

      {/* Projects List */}
      {loading ? (
        <div className="flex items-center justify-center h-40">
          <div className="w-8 h-8 border-2 border-accent border-t-transparent rounded-full animate-spin" />
        </div>
      ) : filteredProjects.length === 0 ? (
        <Card>
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <FolderKanban className="w-12 h-12 text-muted mb-4" />
            <p className="text-secondary">
              {search ? 'Aucun projet trouvé' : 'Aucun projet créé'}
            </p>
            <p className="text-xs text-muted mt-1">
              {search
                ? 'Essayez avec d\'autres termes'
                : 'Créez un projet pour organiser vos audiobooks'}
            </p>
          </div>
        </Card>
      ) : (
        <div className="grid gap-4">
          {filteredProjects.map((project) => (
            <div
              key={project.id}
              className="bg-surface border border-border rounded-xl p-4 hover:border-accent/50 transition-colors group"
            >
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <FolderKanban className="w-5 h-5 text-accent shrink-0" />
                    <h3 className="font-medium text-primary truncate">{project.name}</h3>
                  </div>
                  {project.description && (
                    <p className="text-sm text-secondary mt-1 line-clamp-2">
                      {project.description}
                    </p>
                  )}
                  <div className="flex items-center gap-4 mt-3 text-xs text-muted">
                    <span className="flex items-center gap-1">
                      <Calendar className="w-3.5 h-3.5" />
                      {formatDate(project.updated_at)}
                    </span>
                    {project.files && project.files.length > 0 && (
                      <span className="flex items-center gap-1">
                        <FileText className="w-3.5 h-3.5" />
                        {project.files.length} fichier{project.files.length > 1 ? 's' : ''}
                      </span>
                    )}
                    {project.settings && Object.keys(project.settings).length > 0 && (
                      <span className="flex items-center gap-1">
                        <Settings2 className="w-3.5 h-3.5" />
                        Configuré
                      </span>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button
                    onClick={() => toast.info('Édition bientôt disponible')}
                    className="p-2 text-muted hover:text-primary hover:bg-panel rounded-lg transition-colors"
                    title="Modifier"
                  >
                    <Edit3 className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => handleDelete(project)}
                    className="p-2 text-muted hover:text-red hover:bg-panel rounded-lg transition-colors"
                    title="Supprimer"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => toast.info('Ouverture bientôt disponible')}
                    className="p-2 text-muted hover:text-accent hover:bg-panel rounded-lg transition-colors"
                    title="Ouvrir"
                  >
                    <ChevronRight className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Stats Footer */}
      {!loading && projects.length > 0 && (
        <div className="text-sm text-muted text-center">
          {filteredProjects.length} projet{filteredProjects.length > 1 ? 's' : ''} sur {projects.length}
        </div>
      )}

      <ToastContainer />
    </div>
  )
}
