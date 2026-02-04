import { create } from 'zustand'
import type { CharacterInfo } from '../api/types'

interface CharacterState {
  characters: CharacterInfo[]
  voiceAssignments: Record<string, string>  // character name -> voice_id
  setCharacters: (chars: CharacterInfo[]) => void
  assignVoice: (character: string, voiceId: string) => void
}

export const useCharacterStore = create<CharacterState>((set) => ({
  characters: [],
  voiceAssignments: {},
  setCharacters: (chars) => set({ characters: chars }),
  assignVoice: (character, voiceId) =>
    set((s) => ({ voiceAssignments: { ...s.voiceAssignments, [character]: voiceId } })),
}))
