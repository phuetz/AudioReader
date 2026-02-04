import { Play, Pause, SkipBack } from 'lucide-react'
import { useAudioStore } from '../../stores/useAudioStore'
import { useAudioPlayer } from '../../hooks/useAudioPlayer'
import { useWaveform } from '../../hooks/useWaveform'
import Waveform from './Waveform'

interface AudioPlayerProps {
  url: string
  title?: string
}

export default function AudioPlayer({ url, title }: AudioPlayerProps) {
  const { currentUrl, isPlaying, currentTime, duration, setCurrentUrl, setIsPlaying } = useAudioStore()
  const { seek } = useAudioPlayer()
  const { peaks } = useWaveform(url, 120)

  const isThisPlaying = currentUrl === url && isPlaying
  const progress = currentUrl === url && duration > 0 ? currentTime / duration : 0

  const handlePlay = () => {
    if (currentUrl !== url) {
      setCurrentUrl(url)
      setIsPlaying(true)
    } else {
      setIsPlaying(!isPlaying)
    }
  }

  const handleSeek = (ratio: number) => {
    if (currentUrl !== url) {
      setCurrentUrl(url)
    }
    seek(ratio * duration)
  }

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60)
    const sec = Math.floor(s % 60)
    return `${m}:${sec.toString().padStart(2, '0')}`
  }

  return (
    <div className="bg-panel border border-border rounded-xl p-4 space-y-3">
      {title && <p className="text-sm font-medium text-primary truncate">{title}</p>}

      <Waveform peaks={peaks} progress={progress} onSeek={handleSeek} />

      <div className="flex items-center gap-4">
        <button
          onClick={() => { if (currentUrl === url) seek(0) }}
          className="text-muted hover:text-primary transition-colors cursor-pointer"
        >
          <SkipBack className="w-4 h-4" />
        </button>

        <button
          onClick={handlePlay}
          className="flex items-center justify-center w-9 h-9 rounded-full bg-accent text-deep
            hover:bg-accent-hover transition-colors cursor-pointer"
        >
          {isThisPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
        </button>

        <div className="flex-1" />

        <span className="text-xs font-mono text-muted">
          {currentUrl === url ? formatTime(currentTime) : '0:00'} / {duration > 0 && currentUrl === url ? formatTime(duration) : '--:--'}
        </span>
      </div>
    </div>
  )
}
