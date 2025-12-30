"""
Dynamic Voice Blending.

Calcule des mélanges de voix dynamiques basés sur l'émotion
pour simuler des variations de style (acting).
"""
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from .emotion_analyzer import Emotion, Intensity

@dataclass
class DynamicVoiceConfig:
    """Configuration du mélange dynamique."""
    # Mélange de voix (Target Voice ID, Weight)
    # Ex: ("am_adam", 0.3) signifie ajouter 30% d'Adam
    blend_voice: Optional[str] = None
    blend_weight: float = 0.0
    
    # Prosodie
    speed_mult: float = 1.0
    pitch_shift: float = 0.0  # Semi-tones
    energy_mult: float = 1.0

class DynamicVoiceManager:
    """Gère le calcul des voix dynamiques."""
    
    # Voix de référence pour le blending (Kokoro)
    REF_VOICES = {
        "deep_male": "am_adam",     # Pour colère/autorité
        "soft_female": "af_bella",  # Pour tristesse/douceur
        "bright_female": "af_nicole", # Pour joie/énergie
        "shaky_female": "af_sky",   # Pour peur/instabilité
        "calm_male": "am_michael",  # Pour calme/réconfort
    }

    def get_voice_config(
        self, 
        base_voice: str, 
        emotion: Emotion, 
        intensity: Intensity
    ) -> str:
        """
        Retourne la configuration de voix pour Kokoro (format string).
        Ex: "af_sarah:0.7,am_adam:0.3"
        """
        # Facteur d'intensité (0.0 à 1.0)
        intensity_map = {
            Intensity.LOW: 0.2,
            Intensity.MEDIUM: 0.5,
            Intensity.HIGH: 0.8,
            Intensity.EXTREME: 1.0
        }
        factor = intensity_map.get(intensity, 0.5)
        
        target_blend = None
        
        # Logique de mélange selon l'émotion
        if emotion == Emotion.ANGER:
            target_blend = self.REF_VOICES["deep_male"]
            weight = 0.3 * factor
            
        elif emotion == Emotion.SADNESS:
            target_blend = self.REF_VOICES["soft_female"]
            weight = 0.25 * factor
            
        elif emotion == Emotion.FEAR:
            target_blend = self.REF_VOICES["shaky_female"]
            weight = 0.3 * factor
            
        elif emotion == Emotion.JOY or emotion == Emotion.EXCITEMENT:
            target_blend = self.REF_VOICES["bright_female"]
            weight = 0.2 * factor
            
        elif emotion == Emotion.SUSPENSE:
            target_blend = self.REF_VOICES["calm_male"]
            weight = 0.15 * factor
            
        else:
            # Neutre ou autre : pas de mélange
            return base_voice

        # Si la voix de base est déjà la voix cible, ne pas mélanger
        if base_voice == target_blend:
            return base_voice

        # Construire la chaîne de mélange
        # Kokoro format: "voice_a:weight_a,voice_b:weight_b"
        base_weight = 1.0 - weight
        
        # Arrondir pour propreté
        return f"{base_voice}:{base_weight:.2f},{target_blend}:{weight:.2f}"

    def get_prosody_modifiers(self, emotion: Emotion, intensity: Intensity) -> dict:
        """Retourne les modificateurs de prosodie (speed, pitch)."""
        intensity_map = {
            Intensity.LOW: 1,
            Intensity.MEDIUM: 2,
            Intensity.HIGH: 3,
            Intensity.EXTREME: 4
        }
        level = intensity_map.get(intensity, 2)
        
        modifiers = {"speed": 1.0, "pitch": 0.0}
        
        if emotion == Emotion.ANGER:
            modifiers["speed"] = 1.0 + (0.05 * level)
            modifiers["pitch"] = 0.0  # La voix grave vient du blending
            
        elif emotion == Emotion.FEAR:
            modifiers["speed"] = 1.0 + (0.08 * level)
            modifiers["pitch"] = 0.2 * level
            
        elif emotion == Emotion.SADNESS:
            modifiers["speed"] = 1.0 - (0.05 * level)
            modifiers["pitch"] = -0.1 * level
            
        elif emotion == Emotion.JOY:
            modifiers["speed"] = 1.0 + (0.05 * level)
            modifiers["pitch"] = 0.15 * level
            
        elif emotion == Emotion.SUSPENSE:
            modifiers["speed"] = 0.9 - (0.03 * level)
            
        return modifiers
