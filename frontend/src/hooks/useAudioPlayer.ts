import { useEffect, useRef, useCallback } from 'react'
import { useAudioStore } from '../stores/useAudioStore'

export function useAudioPlayer() {
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const { currentUrl, isPlaying, setIsPlaying, setCurrentTime, setDuration } = useAudioStore()

  useEffect(() => {
    if (!audioRef.current) {
      audioRef.current = new Audio()
    }
    const audio = audioRef.current

    const onTime = () => setCurrentTime(audio.currentTime)
    const onDuration = () => setDuration(audio.duration || 0)
    const onEnded = () => setIsPlaying(false)

    audio.addEventListener('timeupdate', onTime)
    audio.addEventListener('loadedmetadata', onDuration)
    audio.addEventListener('ended', onEnded)

    return () => {
      audio.removeEventListener('timeupdate', onTime)
      audio.removeEventListener('loadedmetadata', onDuration)
      audio.removeEventListener('ended', onEnded)
    }
  }, [setCurrentTime, setDuration, setIsPlaying])

  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return
    if (currentUrl) {
      if (audio.src !== currentUrl) {
        audio.src = currentUrl
        audio.load()
      }
    } else {
      audio.pause()
      audio.src = ''
    }
  }, [currentUrl])

  useEffect(() => {
    const audio = audioRef.current
    if (!audio || !currentUrl) return
    if (isPlaying) {
      audio.play().catch(() => setIsPlaying(false))
    } else {
      audio.pause()
    }
  }, [isPlaying, currentUrl, setIsPlaying])

  const seek = useCallback((time: number) => {
    if (audioRef.current) {
      audioRef.current.currentTime = time
    }
  }, [])

  const toggle = useCallback(() => {
    setIsPlaying(!isPlaying)
  }, [isPlaying, setIsPlaying])

  return { toggle, seek, audioRef }
}
