import { useRef, useEffect } from 'react'

interface WaveformProps {
  peaks: number[]
  progress?: number  // 0..1
  onSeek?: (ratio: number) => void
  height?: number
  accentColor?: string
  baseColor?: string
}

export default function Waveform({
  peaks, progress = 0, onSeek, height = 48,
  accentColor = '#06b6d4', baseColor = '#2d2d44',
}: WaveformProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || peaks.length === 0) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    const w = canvas.clientWidth
    const h = canvas.clientHeight
    canvas.width = w * dpr
    canvas.height = h * dpr
    ctx.scale(dpr, dpr)

    ctx.clearRect(0, 0, w, h)

    const barW = Math.max(2, (w / peaks.length) - 1)
    const gap = 1
    const mid = h / 2

    peaks.forEach((peak, i) => {
      const x = i * (barW + gap)
      const barH = Math.max(2, peak * mid * 0.9)
      const playedRatio = i / peaks.length

      ctx.fillStyle = playedRatio <= progress ? accentColor : baseColor
      ctx.beginPath()
      ctx.roundRect(x, mid - barH, barW, barH * 2, 1)
      ctx.fill()
    })
  }, [peaks, progress, accentColor, baseColor, height])

  const handleClick = (e: React.MouseEvent) => {
    if (!onSeek || !canvasRef.current) return
    const rect = canvasRef.current.getBoundingClientRect()
    const ratio = (e.clientX - rect.left) / rect.width
    onSeek(Math.max(0, Math.min(1, ratio)))
  }

  return (
    <canvas
      ref={canvasRef}
      className="w-full cursor-pointer"
      style={{ height }}
      onClick={handleClick}
    />
  )
}
