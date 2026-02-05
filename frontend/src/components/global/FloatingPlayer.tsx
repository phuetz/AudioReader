import { useState } from 'react'
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  X,
  ChevronUp,
  ChevronDown,
  Repeat,
  Repeat1,
  Shuffle,
  List,
} from 'lucide-react'
import { useAudioStore } from '../../stores/useAudioStore'
import { usePlaylistStore } from '../../stores/usePlaylistStore'
import { useAudioPlayer } from '../../hooks/useAudioPlayer'
import Waveform from '../audio/Waveform'
import { useWaveform } from '../../hooks/useWaveform'

export default function FloatingPlayer() {
  const { currentUrl, isPlaying, currentTime, duration, setCurrentUrl, setIsPlaying } =
    useAudioStore()
  const {
    queue,
    currentIndex,
    repeat,
    shuffle,
    next,
    previous,
    setRepeat,
    toggleShuffle,
    hasNext,
    hasPrevious,
    getCurrentTrack,
    removeTrack,
    clearQueue,
  } = usePlaylistStore()

  const { seek, toggle } = useAudioPlayer()
  const [isExpanded, setIsExpanded] = useState(false)
  const [showQueue, setShowQueue] = useState(false)

  const currentTrack = getCurrentTrack()
  const { peaks } = useWaveform(currentUrl || '', 80)

  // Don't render if no audio
  if (!currentUrl && queue.length === 0) {
    return null
  }

  const progress = duration > 0 ? currentTime / duration : 0

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60)
    const sec = Math.floor(s % 60)
    return `${m}:${sec.toString().padStart(2, '0')}`
  }

  const handlePlayPause = () => {
    if (currentTrack && currentUrl !== currentTrack.url) {
      setCurrentUrl(currentTrack.url)
      setIsPlaying(true)
    } else {
      toggle()
    }
  }

  const handleNext = () => {
    next()
    const nextTrack = getCurrentTrack()
    if (nextTrack) {
      setCurrentUrl(nextTrack.url)
      setIsPlaying(true)
    }
  }

  const handlePrevious = () => {
    // If more than 3 seconds in, restart current track
    if (currentTime > 3) {
      seek(0)
    } else {
      previous()
      const prevTrack = getCurrentTrack()
      if (prevTrack) {
        setCurrentUrl(prevTrack.url)
        setIsPlaying(true)
      }
    }
  }

  const handleSeek = (ratio: number) => {
    seek(ratio * duration)
  }

  const repeatIcon = () => {
    switch (repeat) {
      case 'one':
        return <Repeat1 className="w-4 h-4" />
      case 'all':
        return <Repeat className="w-4 h-4 text-accent" />
      default:
        return <Repeat className="w-4 h-4" />
    }
  }

  const cycleRepeat = () => {
    const modes: ('none' | 'one' | 'all')[] = ['none', 'all', 'one']
    const currentIdx = modes.indexOf(repeat)
    setRepeat(modes[(currentIdx + 1) % modes.length])
  }

  const handleClose = () => {
    setIsPlaying(false)
    setCurrentUrl(null)
    clearQueue()
  }

  return (
    <div
      className={`fixed bottom-4 right-4 z-50 transition-all duration-300 ${
        isExpanded ? 'w-96' : 'w-80'
      }`}
    >
      {/* Queue Panel */}
      {showQueue && (
        <div className="absolute bottom-full right-0 mb-2 w-80 max-h-64 overflow-y-auto bg-panel border border-border rounded-lg shadow-xl">
          <div className="sticky top-0 bg-panel border-b border-border px-3 py-2 flex justify-between items-center">
            <span className="text-sm font-medium text-primary">
              File d'attente ({queue.length})
            </span>
            <button
              onClick={clearQueue}
              className="text-xs text-muted hover:text-danger transition-colors"
            >
              Vider
            </button>
          </div>
          <div className="p-2 space-y-1">
            {queue.map((track, index) => (
              <div
                key={track.id}
                className={`flex items-center gap-2 px-2 py-1.5 rounded text-sm ${
                  index === currentIndex
                    ? 'bg-accent/20 text-accent'
                    : 'hover:bg-surface text-secondary'
                }`}
              >
                <span className="w-5 text-xs text-muted">{index + 1}</span>
                <span className="flex-1 truncate">{track.title}</span>
                <button
                  onClick={() => removeTrack(track.id)}
                  className="text-muted hover:text-danger transition-colors"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            ))}
            {queue.length === 0 && (
              <p className="text-center text-muted text-sm py-4">File d'attente vide</p>
            )}
          </div>
        </div>
      )}

      {/* Main Player */}
      <div className="bg-panel border border-border rounded-xl shadow-xl overflow-hidden">
        {/* Expanded View */}
        {isExpanded && (
          <div className="p-4 space-y-3 border-b border-border">
            <Waveform peaks={peaks} progress={progress} onSeek={handleSeek} height={60} />
          </div>
        )}

        {/* Compact Controls */}
        <div className="p-3">
          {/* Track Info */}
          <div className="flex items-center gap-3 mb-2">
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-primary truncate">
                {currentTrack?.title || 'Aucune piste'}
              </p>
              {currentTrack?.metadata?.book && (
                <p className="text-xs text-muted truncate">{currentTrack.metadata.book}</p>
              )}
            </div>
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="text-muted hover:text-primary transition-colors"
            >
              {isExpanded ? (
                <ChevronDown className="w-4 h-4" />
              ) : (
                <ChevronUp className="w-4 h-4" />
              )}
            </button>
            <button
              onClick={handleClose}
              className="text-muted hover:text-danger transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Progress Bar (compact mode) */}
          {!isExpanded && (
            <div
              className="h-1 bg-surface rounded-full mb-2 cursor-pointer"
              onClick={(e) => {
                const rect = e.currentTarget.getBoundingClientRect()
                const ratio = (e.clientX - rect.left) / rect.width
                handleSeek(ratio)
              }}
            >
              <div
                className="h-full bg-accent rounded-full transition-all"
                style={{ width: `${progress * 100}%` }}
              />
            </div>
          )}

          {/* Controls */}
          <div className="flex items-center gap-2">
            {/* Shuffle */}
            <button
              onClick={toggleShuffle}
              className={`p-1.5 rounded transition-colors ${
                shuffle ? 'text-accent' : 'text-muted hover:text-primary'
              }`}
              title="Lecture aléatoire"
            >
              <Shuffle className="w-4 h-4" />
            </button>

            {/* Previous */}
            <button
              onClick={handlePrevious}
              disabled={!hasPrevious() && currentTime <= 3}
              className="p-1.5 text-muted hover:text-primary disabled:opacity-50 transition-colors"
            >
              <SkipBack className="w-4 h-4" />
            </button>

            {/* Play/Pause */}
            <button
              onClick={handlePlayPause}
              className="flex items-center justify-center w-9 h-9 rounded-full bg-accent text-deep hover:bg-accent-hover transition-colors"
            >
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
            </button>

            {/* Next */}
            <button
              onClick={handleNext}
              disabled={!hasNext()}
              className="p-1.5 text-muted hover:text-primary disabled:opacity-50 transition-colors"
            >
              <SkipForward className="w-4 h-4" />
            </button>

            {/* Repeat */}
            <button
              onClick={cycleRepeat}
              className={`p-1.5 rounded transition-colors ${
                repeat !== 'none' ? 'text-accent' : 'text-muted hover:text-primary'
              }`}
              title={`Répétition: ${repeat}`}
            >
              {repeatIcon()}
            </button>

            {/* Spacer */}
            <div className="flex-1" />

            {/* Time */}
            <span className="text-xs font-mono text-muted">
              {formatTime(currentTime)} / {duration > 0 ? formatTime(duration) : '--:--'}
            </span>

            {/* Queue toggle */}
            <button
              onClick={() => setShowQueue(!showQueue)}
              className={`p-1.5 rounded transition-colors ${
                showQueue ? 'text-accent' : 'text-muted hover:text-primary'
              }`}
              title="File d'attente"
            >
              <List className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
