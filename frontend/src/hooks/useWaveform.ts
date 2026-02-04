import { useState, useEffect } from 'react'

/**
 * Decode an audio URL and return downsampled waveform peaks.
 */
export function useWaveform(url: string | null, bars = 100) {
  const [peaks, setPeaks] = useState<number[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!url) {
      setPeaks([])
      return
    }

    let cancelled = false
    setLoading(true)

    const ctx = new AudioContext()
    fetch(url)
      .then(r => r.arrayBuffer())
      .then(buf => ctx.decodeAudioData(buf))
      .then(decoded => {
        if (cancelled) return
        const data = decoded.getChannelData(0)
        const step = Math.floor(data.length / bars)
        const result: number[] = []
        for (let i = 0; i < bars; i++) {
          let sum = 0
          for (let j = 0; j < step; j++) {
            sum += Math.abs(data[i * step + j])
          }
          result.push(sum / step)
        }
        // Normalize 0..1
        const max = Math.max(...result, 0.001)
        setPeaks(result.map(v => v / max))
        setLoading(false)
      })
      .catch(() => {
        if (!cancelled) setLoading(false)
      })
      .finally(() => ctx.close())

    return () => { cancelled = true }
  }, [url, bars])

  return { peaks, loading }
}
