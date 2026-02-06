import { useState } from 'react'
import { Plus, X } from 'lucide-react'
import type { VoiceInfo } from '../../api/types'

interface BlendSlot {
  voiceId: string
  weight: number
}

interface VoiceBlenderProps {
  voices: VoiceInfo[]
  value: string
  onChange: (blend: string) => void
}

export default function VoiceBlender({ voices, value, onChange }: VoiceBlenderProps) {
  const [slots, setSlots] = useState<BlendSlot[]>(() => {
    if (!value) return []
    return value.split(',').map(part => {
      const [voiceId, weight] = part.split(':')
      return { voiceId, weight: parseInt(weight) || 50 }
    }).filter(s => s.voiceId)
  })

  const updateSlots = (newSlots: BlendSlot[]) => {
    setSlots(newSlots)
    if (newSlots.length === 0) {
      onChange('')
    } else if (newSlots.length === 1) {
      onChange(newSlots[0].voiceId)
    } else {
      onChange(newSlots.map(s => `${s.voiceId}:${s.weight}`).join(','))
    }
  }

  const addSlot = () => {
    const available = voices.filter(v => !slots.some(s => s.voiceId === v.id))
    if (available.length === 0) return
    const equalWeight = Math.floor(100 / (slots.length + 1))
    const newSlots = slots.map(s => ({ ...s, weight: equalWeight }))
    newSlots.push({ voiceId: available[0].id, weight: equalWeight })
    updateSlots(newSlots)
  }

  const removeSlot = (index: number) => {
    const newSlots = slots.filter((_, i) => i !== index)
    if (newSlots.length > 0) {
      const equalWeight = Math.floor(100 / newSlots.length)
      newSlots.forEach(s => s.weight = equalWeight)
    }
    updateSlots(newSlots)
  }

  const changeVoice = (index: number, voiceId: string) => {
    const newSlots = [...slots]
    newSlots[index] = { ...newSlots[index], voiceId }
    updateSlots(newSlots)
  }

  const changeWeight = (index: number, weight: number) => {
    const newSlots = [...slots]
    newSlots[index] = { ...newSlots[index], weight }
    updateSlots(newSlots)
  }

  const totalWeight = slots.reduce((sum, s) => sum + s.weight, 0)

  return (
    <div className="space-y-4">
      {slots.length === 0 ? (
        <div className="text-center py-8 text-muted">
          <p className="text-sm mb-3">Ajoutez des voix pour créer un mélange</p>
          <button
            onClick={addSlot}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-accent/20 text-accent hover:bg-accent/30 transition-colors text-sm"
          >
            <Plus className="w-4 h-4" />
            Ajouter une voix
          </button>
        </div>
      ) : (
        <>
          {/* Blend visualization */}
          <div className="flex h-4 rounded-full overflow-hidden bg-panel">
            {slots.map((slot, i) => {
              const voice = voices.find(v => v.id === slot.voiceId)
              const colors = ['bg-accent', 'bg-cyan', 'bg-green-500', 'bg-yellow-500', 'bg-purple-500']
              const pct = totalWeight > 0 ? (slot.weight / totalWeight) * 100 : 0
              return (
                <div
                  key={i}
                  className={`${colors[i % colors.length]} transition-all duration-300`}
                  style={{ width: `${pct}%` }}
                  title={`${voice?.name || slot.voiceId}: ${Math.round(pct)}%`}
                />
              )
            })}
          </div>

          {/* Slot editors */}
          {slots.map((slot, i) => (
              <div key={i} className="flex items-center gap-3 p-3 rounded-lg bg-panel">
                <select
                  value={slot.voiceId}
                  onChange={(e) => changeVoice(i, e.target.value)}
                  className="flex-1 px-2 py-1.5 rounded bg-surface border border-border text-primary text-sm"
                >
                  {voices.map(v => (
                    <option key={v.id} value={v.id}>{v.name} ({v.gender}) — {v.engine}</option>
                  ))}
                </select>

                <div className="flex items-center gap-2 w-40">
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={slot.weight}
                    onChange={(e) => changeWeight(i, parseInt(e.target.value))}
                    className="flex-1 accent-accent"
                  />
                  <span className="text-xs font-mono text-muted w-8 text-right">
                    {totalWeight > 0 ? Math.round((slot.weight / totalWeight) * 100) : 0}%
                  </span>
                </div>

                <button
                  onClick={() => removeSlot(i)}
                  className="p-1 text-muted hover:text-red-400 transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
          ))}

          {/* Add button */}
          <button
            onClick={addSlot}
            className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-muted hover:text-primary hover:bg-panel transition-colors w-full justify-center"
          >
            <Plus className="w-4 h-4" />
            Ajouter une voix
          </button>
        </>
      )}

      {/* Output info */}
      {slots.length > 0 && (
        <div className="text-xs text-muted border-t border-border pt-3">
          <span className="font-mono">
            {slots.length === 1
              ? slots[0].voiceId
              : slots.map(s => `${s.voiceId}:${totalWeight > 0 ? Math.round((s.weight / totalWeight) * 100) : 0}`).join(', ')
            }
          </span>
        </div>
      )}
    </div>
  )
}
