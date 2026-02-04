import { useEffect } from 'react'
import { useVoiceStore } from '../../stores/useVoiceStore'
import VoiceCard from './VoiceCard'
import { previewVoice } from '../../api/endpoints'

export default function VoiceGallery() {
  const { voices, loading, fetchVoices } = useVoiceStore()

  useEffect(() => { fetchVoices() }, [fetchVoices])

  const handlePreview = async (voiceId: string) => {
    try {
      const blob = await previewVoice(voiceId)
      const url = URL.createObjectURL(blob)
      const audio = new Audio(url)
      audio.play()
      audio.onended = () => URL.revokeObjectURL(url)
    } catch { /* ignore */ }
  }

  if (loading) {
    return <div className="text-sm text-muted py-4">Chargement...</div>
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
      {voices.map((v) => (
        <VoiceCard key={v.id} voice={v} onPreview={() => handlePreview(v.id)} />
      ))}
    </div>
  )
}
