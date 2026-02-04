import { useEffect } from 'react'
import { useVoiceStore } from '../../stores/useVoiceStore'
import VoiceCard from './VoiceCard'
import { previewVoice } from '../../api/endpoints'

interface VoiceSelectorProps {
  value: string
  onChange: (voiceId: string) => void
  language?: string
}

export default function VoiceSelector({ value, onChange, language }: VoiceSelectorProps) {
  const { voices, loading, fetchVoices } = useVoiceStore()

  useEffect(() => {
    fetchVoices(language)
  }, [fetchVoices, language])

  const handlePreview = async (voiceId: string) => {
    try {
      const blob = await previewVoice(voiceId, undefined, undefined, language)
      const url = URL.createObjectURL(blob)
      const audio = new Audio(url)
      audio.play()
      audio.onended = () => URL.revokeObjectURL(url)
    } catch {
      // ignore
    }
  }

  if (loading) {
    return <div className="text-sm text-muted">Chargement des voix...</div>
  }

  return (
    <div className="space-y-2 max-h-64 overflow-y-auto">
      {voices.map((v) => (
        <VoiceCard
          key={v.id}
          voice={v}
          selected={v.id === value}
          onSelect={() => onChange(v.id)}
          onPreview={() => handlePreview(v.id)}
        />
      ))}
      {voices.length === 0 && (
        <p className="text-sm text-muted py-4 text-center">Aucune voix disponible</p>
      )}
    </div>
  )
}
