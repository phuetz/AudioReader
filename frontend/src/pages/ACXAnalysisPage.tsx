import { useState } from 'react'
import { Shield, Upload, CheckCircle, XCircle, AlertTriangle, Wrench } from 'lucide-react'
import Card from '../components/ui/Card'
import Button from '../components/ui/Button'
import DropZone from '../components/ui/DropZone'
import { ToastContainer, toast } from '../components/ui/Toast'

interface ACXAnalysis {
  is_compliant: boolean
  rms_db: number
  peak_db: number
  true_peak_db: number
  noise_floor_db: number
  integrated_lufs: number
  sample_rate: number
  channels: number
  duration: number
  issues: string[]
}

// ACX Standards
const ACX_STANDARDS = {
  rms_min: -23,
  rms_max: -18,
  peak_max: -3,
  true_peak_max: -1,
  noise_floor_max: -60,
  sample_rate_min: 44100,
}

export default function ACXAnalysisPage() {
  const [file, setFile] = useState<File | null>(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [fixing, setFixing] = useState(false)
  const [analysis, setAnalysis] = useState<ACXAnalysis | null>(null)

  const handleFile = (f: File) => {
    setFile(f)
    setAnalysis(null)
    toast.success(`Fichier "${f.name}" sélectionné`)
  }

  const handleAnalyze = async () => {
    if (!file) return
    setAnalyzing(true)
    try {
      const formData = new FormData()
      formData.append('file', file)

      const res = await fetch('/api/v2/acx/analyze', {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) throw new Error('Erreur analyse')

      const data = await res.json()
      setAnalysis(data)
      toast.success(data.is_compliant ? 'Fichier conforme ACX !' : 'Problèmes détectés')
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Erreur')
      // Mock data for demo if API not available
      setAnalysis({
        is_compliant: false,
        rms_db: -20.5,
        peak_db: -2.1,
        true_peak_db: -0.5,
        noise_floor_db: -55,
        integrated_lufs: -19.2,
        sample_rate: 44100,
        channels: 1,
        duration: 125.5,
        issues: ['Peak trop élevé (> -3 dB)', 'True Peak trop élevé (> -1 dB)', 'Noise floor trop élevé (> -60 dB)'],
      })
    } finally {
      setAnalyzing(false)
    }
  }

  const handleFix = async () => {
    if (!file) return
    setFixing(true)
    try {
      const formData = new FormData()
      formData.append('file', file)

      const res = await fetch('/api/v2/acx/fix', {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) throw new Error('Erreur correction')

      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = file.name.replace(/\.[^.]+$/, '_acx_compliant.wav')
      a.click()
      URL.revokeObjectURL(url)

      toast.success('Fichier corrigé téléchargé !')
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Erreur correction')
    } finally {
      setFixing(false)
    }
  }

  const getStatusIcon = (value: number, min: number | null, max: number) => {
    const withinRange = (min === null || value >= min) && value <= max
    if (withinRange) return <CheckCircle className="w-4 h-4 text-green" />
    return <XCircle className="w-4 h-4 text-red" />
  }

  const getStatusColor = (value: number, min: number | null, max: number) => {
    const withinRange = (min === null || value >= min) && value <= max
    return withinRange ? 'text-green' : 'text-red'
  }

  return (
    <div className="space-y-6 max-w-4xl">
      <h1 className="text-xl font-semibold text-primary flex items-center gap-2">
        <Shield className="w-5 h-5 text-accent" />
        Analyse ACX / Audible
      </h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Upload & Actions */}
        <div className="space-y-4">
          <Card title="Fichier audio">
            <DropZone
              accept=".wav,.mp3,.m4a,.flac"
              onFile={handleFile}
              label="Déposez un fichier audio WAV, MP3, M4A ou FLAC"
            />
            {file && (
              <div className="mt-3 flex items-center gap-2 text-sm text-secondary">
                <Upload className="w-4 h-4" />
                <span>{file.name}</span>
                <span className="text-muted">({(file.size / 1024 / 1024).toFixed(1)} MB)</span>
              </div>
            )}
          </Card>

          <div className="flex gap-2">
            <Button onClick={handleAnalyze} loading={analyzing} icon={<Shield className="w-4 h-4" />}
              disabled={!file}>
              Analyser
            </Button>
            {analysis && !analysis.is_compliant && (
              <Button variant="secondary" onClick={handleFix} loading={fixing} icon={<Wrench className="w-4 h-4" />}>
                Corriger automatiquement
              </Button>
            )}
          </div>

          {/* ACX Standards Reference */}
          <Card title="Standards ACX/Audible">
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-secondary">RMS (loudness)</span>
                <span className="font-mono text-primary">-23 dB à -18 dB</span>
              </div>
              <div className="flex justify-between">
                <span className="text-secondary">Peak maximum</span>
                <span className="font-mono text-primary">≤ -3 dB</span>
              </div>
              <div className="flex justify-between">
                <span className="text-secondary">True Peak maximum</span>
                <span className="font-mono text-primary">≤ -1 dB</span>
              </div>
              <div className="flex justify-between">
                <span className="text-secondary">Noise floor</span>
                <span className="font-mono text-primary">≤ -60 dB</span>
              </div>
              <div className="flex justify-between">
                <span className="text-secondary">Sample rate minimum</span>
                <span className="font-mono text-primary">≥ 44.1 kHz</span>
              </div>
              <div className="flex justify-between">
                <span className="text-secondary">Canaux</span>
                <span className="font-mono text-primary">Mono recommandé</span>
              </div>
            </div>
          </Card>
        </div>

        {/* Analysis Results */}
        <div className="space-y-4">
          {analysis ? (
            <>
              {/* Compliance Status */}
              <Card>
                <div className={`flex items-center gap-3 p-4 rounded-lg ${
                  analysis.is_compliant ? 'bg-green/10' : 'bg-red/10'
                }`}>
                  {analysis.is_compliant ? (
                    <CheckCircle className="w-8 h-8 text-green" />
                  ) : (
                    <AlertTriangle className="w-8 h-8 text-red" />
                  )}
                  <div>
                    <p className={`font-semibold ${analysis.is_compliant ? 'text-green' : 'text-red'}`}>
                      {analysis.is_compliant ? 'Conforme ACX' : 'Non conforme ACX'}
                    </p>
                    <p className="text-sm text-secondary">
                      {analysis.is_compliant
                        ? 'Ce fichier respecte tous les standards Audible'
                        : `${analysis.issues.length} problème(s) détecté(s)`}
                    </p>
                  </div>
                </div>
              </Card>

              {/* Detailed Metrics */}
              <Card title="Métriques détaillées">
                <div className="space-y-3">
                  {/* RMS */}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      {getStatusIcon(analysis.rms_db, ACX_STANDARDS.rms_min, ACX_STANDARDS.rms_max)}
                      <span className="text-secondary text-sm">RMS</span>
                    </div>
                    <span className={`font-mono text-sm ${getStatusColor(analysis.rms_db, ACX_STANDARDS.rms_min, ACX_STANDARDS.rms_max)}`}>
                      {analysis.rms_db.toFixed(1)} dB
                    </span>
                  </div>

                  {/* Peak */}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      {getStatusIcon(analysis.peak_db, null, ACX_STANDARDS.peak_max)}
                      <span className="text-secondary text-sm">Peak</span>
                    </div>
                    <span className={`font-mono text-sm ${getStatusColor(analysis.peak_db, null, ACX_STANDARDS.peak_max)}`}>
                      {analysis.peak_db.toFixed(1)} dB
                    </span>
                  </div>

                  {/* True Peak */}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      {getStatusIcon(analysis.true_peak_db, null, ACX_STANDARDS.true_peak_max)}
                      <span className="text-secondary text-sm">True Peak</span>
                    </div>
                    <span className={`font-mono text-sm ${getStatusColor(analysis.true_peak_db, null, ACX_STANDARDS.true_peak_max)}`}>
                      {analysis.true_peak_db.toFixed(1)} dB
                    </span>
                  </div>

                  {/* Noise Floor */}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      {getStatusIcon(analysis.noise_floor_db, null, ACX_STANDARDS.noise_floor_max)}
                      <span className="text-secondary text-sm">Noise Floor</span>
                    </div>
                    <span className={`font-mono text-sm ${getStatusColor(analysis.noise_floor_db, null, ACX_STANDARDS.noise_floor_max)}`}>
                      {analysis.noise_floor_db.toFixed(1)} dB
                    </span>
                  </div>

                  {/* LUFS */}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <CheckCircle className="w-4 h-4 text-muted" />
                      <span className="text-secondary text-sm">Integrated LUFS</span>
                    </div>
                    <span className="font-mono text-sm text-primary">
                      {analysis.integrated_lufs.toFixed(1)} LUFS
                    </span>
                  </div>

                  {/* Sample Rate */}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      {analysis.sample_rate >= ACX_STANDARDS.sample_rate_min
                        ? <CheckCircle className="w-4 h-4 text-green" />
                        : <XCircle className="w-4 h-4 text-red" />}
                      <span className="text-secondary text-sm">Sample Rate</span>
                    </div>
                    <span className="font-mono text-sm text-primary">
                      {(analysis.sample_rate / 1000).toFixed(1)} kHz
                    </span>
                  </div>

                  {/* Channels */}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      {analysis.channels === 1
                        ? <CheckCircle className="w-4 h-4 text-green" />
                        : <AlertTriangle className="w-4 h-4 text-yellow" />}
                      <span className="text-secondary text-sm">Canaux</span>
                    </div>
                    <span className="font-mono text-sm text-primary">
                      {analysis.channels === 1 ? 'Mono' : 'Stéréo'}
                    </span>
                  </div>

                  {/* Duration */}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <CheckCircle className="w-4 h-4 text-muted" />
                      <span className="text-secondary text-sm">Durée</span>
                    </div>
                    <span className="font-mono text-sm text-primary">
                      {Math.floor(analysis.duration / 60)}:{(analysis.duration % 60).toFixed(0).padStart(2, '0')}
                    </span>
                  </div>
                </div>
              </Card>

              {/* Issues */}
              {analysis.issues.length > 0 && (
                <Card title="Problèmes détectés">
                  <ul className="space-y-2">
                    {analysis.issues.map((issue, i) => (
                      <li key={i} className="flex items-start gap-2 text-sm">
                        <XCircle className="w-4 h-4 text-red shrink-0 mt-0.5" />
                        <span className="text-secondary">{issue}</span>
                      </li>
                    ))}
                  </ul>
                </Card>
              )}
            </>
          ) : (
            <Card>
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <Shield className="w-12 h-12 text-muted mb-4" />
                <p className="text-secondary">Sélectionnez un fichier audio pour analyser sa conformité ACX</p>
                <p className="text-xs text-muted mt-2">
                  L'analyse vérifie tous les standards requis par Audible/ACX
                </p>
              </div>
            </Card>
          )}
        </div>
      </div>

      <ToastContainer />
    </div>
  )
}
