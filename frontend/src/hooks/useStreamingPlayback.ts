import { useRef, useState, useCallback, useEffect } from 'react'

interface StreamChunk {
  type: 'chunk' | 'progress' | 'done' | 'error'
  audio?: string // base64 encoded PCM float32
  sample_rate?: number
  timestamp_ms?: number
  is_final?: boolean
  text_segment?: string
  progress_percent?: number
  total_duration_ms?: number
  message?: string
}

interface StreamingPlaybackState {
  isStreaming: boolean
  isBuffering: boolean
  progress: number
  currentSegment: string
  error: string | null
  totalDurationMs: number
}

export function useStreamingPlayback() {
  const audioContextRef = useRef<AudioContext | null>(null)
  const sourceNodesRef = useRef<AudioBufferSourceNode[]>([])
  const nextPlayTimeRef = useRef<number>(0)
  const eventSourceRef = useRef<EventSource | null>(null)

  const [state, setState] = useState<StreamingPlaybackState>({
    isStreaming: false,
    isBuffering: true,
    progress: 0,
    currentSegment: '',
    error: null,
    totalDurationMs: 0,
  })

  // Cleanup function
  const cleanup = useCallback(() => {
    // Close EventSource
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
      eventSourceRef.current = null
    }

    // Stop all playing sources
    sourceNodesRef.current.forEach((source) => {
      try {
        source.stop()
        source.disconnect()
      } catch {
        // Ignore errors if already stopped
      }
    })
    sourceNodesRef.current = []

    // Close audio context
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close()
      audioContextRef.current = null
    }

    nextPlayTimeRef.current = 0
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return cleanup
  }, [cleanup])

  // Convert base64 PCM float32 to AudioBuffer
  const decodeAudioChunk = useCallback(
    async (base64Audio: string, sampleRate: number): Promise<AudioBuffer | null> => {
      if (!audioContextRef.current) return null

      try {
        // Decode base64
        const binaryString = atob(base64Audio)
        const bytes = new Uint8Array(binaryString.length)
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i)
        }

        // Convert to Float32Array (PCM float32)
        const float32Array = new Float32Array(bytes.buffer)

        // Create AudioBuffer
        const audioBuffer = audioContextRef.current.createBuffer(
          1, // mono
          float32Array.length,
          sampleRate
        )
        audioBuffer.getChannelData(0).set(float32Array)

        return audioBuffer
      } catch (error) {
        console.error('Error decoding audio chunk:', error)
        return null
      }
    },
    []
  )

  // Schedule audio buffer for playback
  const scheduleAudioBuffer = useCallback((audioBuffer: AudioBuffer) => {
    if (!audioContextRef.current) return

    const source = audioContextRef.current.createBufferSource()
    source.buffer = audioBuffer
    source.connect(audioContextRef.current.destination)

    // Schedule at the next available time
    const currentTime = audioContextRef.current.currentTime
    const startTime = Math.max(currentTime, nextPlayTimeRef.current)

    source.start(startTime)
    nextPlayTimeRef.current = startTime + audioBuffer.duration

    // Keep track for cleanup
    sourceNodesRef.current.push(source)

    // Clean up finished sources
    source.onended = () => {
      const index = sourceNodesRef.current.indexOf(source)
      if (index > -1) {
        sourceNodesRef.current.splice(index, 1)
      }
      source.disconnect()
    }
  }, [])

  // Start streaming synthesis and playback
  const startStreaming = useCallback(
    async (
      text: string,
      options: {
        voice?: string
        speed?: number
        chunkSizeMs?: number
      } = {}
    ) => {
      // Cleanup any existing stream
      cleanup()

      // Initialize audio context
      audioContextRef.current = new AudioContext({ sampleRate: 24000 })

      setState({
        isStreaming: true,
        isBuffering: true,
        progress: 0,
        currentSegment: '',
        error: null,
        totalDurationMs: 0,
      })

      try {
        // Build URL with query parameters
        const baseUrl = '/api/v2/synthesize-stream'
        const body = JSON.stringify({
          text,
          voice: options.voice || 'ff_siwis',
          speed: options.speed || 1.0,
          chunk_size_ms: options.chunkSizeMs || 300,
        })

        // Use fetch with POST for streaming
        const response = await fetch(baseUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body,
        })

        if (!response.ok) {
          throw new Error(`HTTP error: ${response.status}`)
        }

        const reader = response.body?.getReader()
        if (!reader) {
          throw new Error('Response body is not readable')
        }

        const decoder = new TextDecoder()
        let buffer = ''

        // Process the stream
        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          buffer += decoder.decode(value, { stream: true })

          // Process complete SSE messages
          const lines = buffer.split('\n')
          buffer = lines.pop() || '' // Keep incomplete line in buffer

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data: StreamChunk = JSON.parse(line.slice(6))
                await handleStreamChunk(data)
              } catch {
                // Ignore parse errors
              }
            }
          }
        }
      } catch (error) {
        setState((prev) => ({
          ...prev,
          isStreaming: false,
          isBuffering: false,
          error: error instanceof Error ? error.message : 'Unknown error',
        }))
      }
    },
    [cleanup, decodeAudioChunk, scheduleAudioBuffer]
  )

  // Handle incoming stream chunks
  const handleStreamChunk = useCallback(
    async (chunk: StreamChunk) => {
      switch (chunk.type) {
        case 'chunk':
          if (chunk.audio && chunk.sample_rate) {
            const audioBuffer = await decodeAudioChunk(chunk.audio, chunk.sample_rate)
            if (audioBuffer) {
              scheduleAudioBuffer(audioBuffer)
              setState((prev) => ({ ...prev, isBuffering: false }))
            }
          }
          break

        case 'progress':
          setState((prev) => ({
            ...prev,
            progress: chunk.progress_percent || prev.progress,
            currentSegment: chunk.text_segment || prev.currentSegment,
          }))
          break

        case 'done':
          setState((prev) => ({
            ...prev,
            isStreaming: false,
            progress: 100,
            totalDurationMs: chunk.total_duration_ms || prev.totalDurationMs,
          }))
          break

        case 'error':
          setState((prev) => ({
            ...prev,
            isStreaming: false,
            isBuffering: false,
            error: chunk.message || 'Unknown error',
          }))
          break
      }
    },
    [decodeAudioChunk, scheduleAudioBuffer]
  )

  // Stop streaming
  const stopStreaming = useCallback(() => {
    cleanup()
    setState({
      isStreaming: false,
      isBuffering: false,
      progress: 0,
      currentSegment: '',
      error: null,
      totalDurationMs: 0,
    })
  }, [cleanup])

  return {
    ...state,
    startStreaming,
    stopStreaming,
  }
}
