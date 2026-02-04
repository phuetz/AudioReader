import { useState, useCallback } from 'react'
import type { UploadResponse } from '../api/types'
import { uploadFile } from '../api/endpoints'

export function useFileUpload() {
  const [uploading, setUploading] = useState(false)
  const [result, setResult] = useState<UploadResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const upload = useCallback(async (file: File) => {
    setUploading(true)
    setError(null)
    try {
      const res = await uploadFile(file)
      setResult(res)
      return res
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Upload failed'
      setError(msg)
      return null
    } finally {
      setUploading(false)
    }
  }, [])

  const reset = useCallback(() => {
    setResult(null)
    setError(null)
  }, [])

  return { upload, uploading, result, error, reset }
}
