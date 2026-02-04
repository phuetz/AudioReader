import { Play, Pause, Loader } from 'lucide-react'
import { useState, useRef } from 'react'

interface MiniPlayerProps {
  url: string
}

export default function MiniPlayer({ url }: MiniPlayerProps) {
  const [playing, setPlaying] = useState(false)
  const [loading, setLoading] = useState(false)
  const audioRef = useRef<HTMLAudioElement | null>(null)

  const toggle = () => {
    if (!audioRef.current) {
      audioRef.current = new Audio(url)
      audioRef.current.onended = () => setPlaying(false)
      audioRef.current.oncanplay = () => setLoading(false)
    }

    if (playing) {
      audioRef.current.pause()
      setPlaying(false)
    } else {
      setLoading(true)
      audioRef.current.play().then(() => {
        setPlaying(true)
        setLoading(false)
      }).catch(() => setLoading(false))
    }
  }

  return (
    <button
      onClick={toggle}
      className="flex items-center justify-center w-7 h-7 rounded-full bg-panel border border-border
        hover:border-accent text-secondary hover:text-accent transition-colors cursor-pointer"
    >
      {loading ? (
        <Loader className="w-3.5 h-3.5 animate-spin" />
      ) : playing ? (
        <Pause className="w-3.5 h-3.5" />
      ) : (
        <Play className="w-3.5 h-3.5 ml-0.5" />
      )}
    </button>
  )
}
