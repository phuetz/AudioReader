import { useState, useEffect } from 'react'
import {
  BookA,
  Plus,
  Trash2,
  Edit3,
  Search,
  Check,
  X,
  Sparkles,
  FileText,
  ArrowRight,
} from 'lucide-react'
import Card from '../components/ui/Card'
import Button from '../components/ui/Button'
import Input from '../components/ui/Input'
import Badge from '../components/ui/Badge'
import { ToastContainer, toast } from '../components/ui/Toast'
import {
  getCorrections,
  createCorrection,
  updateCorrection,
  deleteCorrection,
  applyCorrections,
} from '../api/endpoints'
import type { CorrectionRule } from '../api/types'

const CONFIDENCE_COLORS: Record<string, 'green' | 'accent' | 'red'> = {
  high: 'green',
  medium: 'accent',
  low: 'red',
}

export default function CorrectionsPage() {
  const [corrections, setCorrections] = useState<CorrectionRule[]>([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')

  // Form state
  const [showForm, setShowForm] = useState(false)
  const [editingId, setEditingId] = useState<string | null>(null)
  const [formPattern, setFormPattern] = useState('')
  const [formReplacement, setFormReplacement] = useState('')
  const [formConfidence, setFormConfidence] = useState<'high' | 'medium' | 'low'>('high')
  const [formNotes, setFormNotes] = useState('')
  const [saving, setSaving] = useState(false)

  // Test state
  const [testText, setTestText] = useState('')
  const [testResult, setTestResult] = useState<{ corrected: string; changes: number } | null>(null)
  const [testing, setTesting] = useState(false)

  const loadCorrections = async () => {
    setLoading(true)
    try {
      const data = await getCorrections(search || undefined)
      setCorrections(data.corrections)
    } catch (e) {
      toast.error('Erreur chargement des corrections')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadCorrections()
  }, [search])

  const resetForm = () => {
    setFormPattern('')
    setFormReplacement('')
    setFormConfidence('high')
    setFormNotes('')
    setEditingId(null)
    setShowForm(false)
  }

  const handleEdit = (correction: CorrectionRule) => {
    setFormPattern(correction.pattern)
    setFormReplacement(correction.replacement)
    setFormConfidence(correction.confidence)
    setFormNotes(correction.notes)
    setEditingId(correction.id)
    setShowForm(true)
  }

  const handleSubmit = async () => {
    if (!formPattern.trim()) {
      toast.error('Le pattern est requis')
      return
    }

    setSaving(true)
    try {
      if (editingId) {
        await updateCorrection(editingId, {
          pattern: formPattern,
          replacement: formReplacement,
          confidence: formConfidence,
          notes: formNotes,
        })
        toast.success('Correction mise à jour')
      } else {
        await createCorrection({
          pattern: formPattern,
          replacement: formReplacement,
          confidence: formConfidence,
          notes: formNotes,
        })
        toast.success('Correction ajoutée')
      }
      resetForm()
      loadCorrections()
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Erreur')
    } finally {
      setSaving(false)
    }
  }

  const handleDelete = async (id: string) => {
    if (!confirm('Supprimer cette correction ?')) return
    try {
      await deleteCorrection(id)
      toast.success('Correction supprimée')
      loadCorrections()
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Erreur')
    }
  }

  const handleTest = async () => {
    if (!testText.trim()) {
      toast.error('Entrez du texte à tester')
      return
    }
    setTesting(true)
    setTestResult(null)
    try {
      const result = await applyCorrections(testText)
      setTestResult({ corrected: result.corrected, changes: result.changes_count })
      if (result.changes_count > 0) {
        toast.success(`${result.changes_count} correction(s) appliquée(s)`)
      } else {
        toast.info('Aucune correction applicable')
      }
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Erreur')
    } finally {
      setTesting(false)
    }
  }

  return (
    <div className="space-y-6 max-w-5xl">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold text-primary flex items-center gap-2">
          <BookA className="w-5 h-5 text-accent" />
          Corrections de prononciation
        </h1>
        <Button
          onClick={() => setShowForm(!showForm)}
          icon={<Plus className="w-4 h-4" />}
        >
          Nouvelle correction
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main content */}
        <div className="lg:col-span-2 space-y-4">
          {/* Add/Edit Form */}
          {showForm && (
            <Card title={editingId ? 'Modifier la correction' : 'Nouvelle correction'}>
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Input
                    label="Pattern (texte à remplacer)"
                    value={formPattern}
                    onChange={(e) => setFormPattern(e.target.value)}
                    placeholder="ex: M."
                  />
                  <Input
                    label="Remplacement"
                    value={formReplacement}
                    onChange={(e) => setFormReplacement(e.target.value)}
                    placeholder="ex: Monsieur"
                  />
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-1.5">
                    <label className="text-xs font-medium text-secondary">Confiance</label>
                    <select
                      value={formConfidence}
                      onChange={(e) => setFormConfidence(e.target.value as 'high' | 'medium' | 'low')}
                      className="w-full px-3 py-2 rounded-lg bg-panel border border-border text-primary text-sm cursor-pointer"
                    >
                      <option value="high">Haute (toujours appliquer)</option>
                      <option value="medium">Moyenne (généralement appliquer)</option>
                      <option value="low">Basse (vérifier manuellement)</option>
                    </select>
                  </div>
                  <Input
                    label="Notes (optionnel)"
                    value={formNotes}
                    onChange={(e) => setFormNotes(e.target.value)}
                    placeholder="ex: Abréviation courante"
                  />
                </div>
                <div className="flex gap-2">
                  <Button onClick={handleSubmit} loading={saving} icon={<Check className="w-4 h-4" />}>
                    {editingId ? 'Mettre à jour' : 'Ajouter'}
                  </Button>
                  <Button variant="ghost" onClick={resetForm} icon={<X className="w-4 h-4" />}>
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
              placeholder="Rechercher une correction..."
              className="w-full pl-10 pr-4 py-2 rounded-lg bg-panel border border-border text-primary text-sm placeholder:text-muted"
            />
          </div>

          {/* Corrections List */}
          <Card title={`Corrections (${corrections.length})`}>
            {loading ? (
              <div className="py-8 text-center">
                <div className="w-8 h-8 border-2 border-accent border-t-transparent rounded-full animate-spin mx-auto" />
              </div>
            ) : corrections.length === 0 ? (
              <div className="py-8 text-center">
                <BookA className="w-10 h-10 text-muted mx-auto mb-2" />
                <p className="text-secondary text-sm">
                  {search ? 'Aucune correction trouvée' : 'Aucune correction définie'}
                </p>
                <p className="text-xs text-muted mt-1">
                  {search ? 'Essayez avec d\'autres termes' : 'Ajoutez des corrections pour améliorer la prononciation'}
                </p>
              </div>
            ) : (
              <div className="divide-y divide-border/50">
                {corrections.map((c) => (
                  <div key={c.id} className="flex items-center gap-3 py-3 group">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <code className="text-sm font-mono text-cyan bg-cyan/10 px-1.5 py-0.5 rounded">
                          {c.pattern}
                        </code>
                        <ArrowRight className="w-3 h-3 text-muted" />
                        <code className="text-sm font-mono text-green bg-green/10 px-1.5 py-0.5 rounded">
                          {c.replacement || '(supprimer)'}
                        </code>
                      </div>
                      {c.notes && (
                        <p className="text-xs text-muted truncate">{c.notes}</p>
                      )}
                    </div>
                    <Badge color={CONFIDENCE_COLORS[c.confidence]}>{c.confidence}</Badge>
                    <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button
                        onClick={() => handleEdit(c)}
                        className="p-1.5 text-muted hover:text-accent transition-colors"
                        title="Modifier"
                      >
                        <Edit3 className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => handleDelete(c.id)}
                        className="p-1.5 text-muted hover:text-red transition-colors"
                        title="Supprimer"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </Card>
        </div>

        {/* Sidebar */}
        <div className="space-y-4">
          {/* Test Panel */}
          <Card title="Tester les corrections" icon={<Sparkles className="w-4 h-4" />}>
            <div className="space-y-3">
              <textarea
                value={testText}
                onChange={(e) => setTestText(e.target.value)}
                placeholder="Entrez du texte pour tester les corrections..."
                rows={4}
                className="w-full px-3 py-2 rounded-lg bg-panel border border-border text-primary text-sm resize-none"
              />
              <Button
                onClick={handleTest}
                loading={testing}
                disabled={!testText.trim()}
                className="w-full"
                icon={<Sparkles className="w-4 h-4" />}
              >
                Appliquer les corrections
              </Button>

              {testResult && (
                <div className="space-y-2 pt-3 border-t border-border">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-muted">Résultat</span>
                    <Badge color={testResult.changes > 0 ? 'green' : 'muted'}>
                      {testResult.changes} modification{testResult.changes > 1 ? 's' : ''}
                    </Badge>
                  </div>
                  <div className="p-2 bg-panel rounded-lg text-sm text-primary whitespace-pre-wrap">
                    {testResult.corrected}
                  </div>
                </div>
              )}
            </div>
          </Card>

          {/* Tips */}
          <Card title="Conseils" icon={<FileText className="w-4 h-4" />}>
            <ul className="space-y-2 text-xs text-muted">
              <li className="flex items-start gap-2">
                <span className="text-accent">•</span>
                <span>Les patterns sont sensibles à la casse</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-accent">•</span>
                <span>Utilisez les abréviations courantes (M., Mme, Dr.)</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-accent">•</span>
                <span>La confiance "haute" est toujours appliquée</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-accent">•</span>
                <span>Testez vos corrections avant de les utiliser</span>
              </li>
            </ul>
          </Card>

          {/* Common Corrections */}
          <Card title="Corrections courantes">
            <div className="space-y-2">
              {[
                { p: 'M.', r: 'Monsieur' },
                { p: 'Mme', r: 'Madame' },
                { p: 'Dr.', r: 'Docteur' },
                { p: 'etc.', r: 'et cetera' },
                { p: 'n°', r: 'numéro' },
              ].map((ex) => (
                <button
                  key={ex.p}
                  onClick={() => {
                    setFormPattern(ex.p)
                    setFormReplacement(ex.r)
                    setFormConfidence('high')
                    setShowForm(true)
                  }}
                  className="w-full flex items-center gap-2 px-2 py-1.5 text-xs rounded hover:bg-panel transition-colors text-left"
                >
                  <code className="text-cyan">{ex.p}</code>
                  <ArrowRight className="w-3 h-3 text-muted" />
                  <code className="text-green">{ex.r}</code>
                </button>
              ))}
            </div>
          </Card>
        </div>
      </div>

      <ToastContainer />
    </div>
  )
}
