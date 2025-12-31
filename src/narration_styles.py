"""
Styles de Narration v2.4.

Gere differents styles de narration pour adapter la prosodie:
- Formel: discours pose, professionnel
- Conversationnel: naturel, decontracte
- Dramatique: intense, emotionnel
- Storytelling: captivant, variations dynamiques

Chaque style ajuste: vitesse, pitch, pauses, variations.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict
import random


class NarrationStyle(Enum):
    """Styles de narration disponibles."""
    FORMAL = "formal"              # Discours pose, professionnel
    CONVERSATIONAL = "conversational"  # Naturel, decontracte
    DRAMATIC = "dramatic"          # Intense, emotionnel
    STORYTELLING = "storytelling"  # Captivant, variations
    DOCUMENTARY = "documentary"    # Informatif, neutre
    INTIMATE = "intimate"          # Proche, confidentiel
    ENERGETIC = "energetic"        # Dynamique, enthousiaste


@dataclass
class StyleProsodyProfile:
    """Profil prosodique pour un style de narration."""
    # Vitesse de base (1.0 = normal)
    base_speed: float = 1.0
    speed_variation: float = 0.05  # Variation aleatoire +/-

    # Pitch de base (0.0 = normal, en demi-tons)
    base_pitch: float = 0.0
    pitch_variation: float = 0.0

    # Volume (1.0 = normal)
    base_volume: float = 1.0
    volume_variation: float = 0.05

    # Pauses (multiplicateur)
    pause_multiplier: float = 1.0
    sentence_pause: float = 0.4  # Pause apres phrase (secondes)
    paragraph_pause: float = 0.8  # Pause apres paragraphe

    # Respirations
    breath_frequency: float = 1.0  # Frequence des respirations
    breath_intensity: float = 0.5  # Intensite des respirations

    # Emphase
    emphasis_strength: float = 1.0  # Force de l'emphase sur mots importants

    # Variations dynamiques
    enable_dynamic_variation: bool = True
    crescendo_on_climax: bool = False

    # Description
    description: str = ""


# Profils predefinis pour chaque style
STYLE_PROFILES: Dict[NarrationStyle, StyleProsodyProfile] = {
    NarrationStyle.FORMAL: StyleProsodyProfile(
        base_speed=0.95,
        speed_variation=0.02,  # Tres stable
        base_pitch=0.0,
        pitch_variation=0.1,
        base_volume=1.0,
        volume_variation=0.02,
        pause_multiplier=1.2,  # Pauses plus longues
        sentence_pause=0.5,
        paragraph_pause=1.0,
        breath_frequency=0.8,
        breath_intensity=0.3,
        emphasis_strength=0.7,  # Emphase moderee
        enable_dynamic_variation=False,
        description="Discours pose et professionnel, rythme stable"
    ),

    NarrationStyle.CONVERSATIONAL: StyleProsodyProfile(
        base_speed=1.05,
        speed_variation=0.08,  # Plus de variation naturelle
        base_pitch=0.3,  # Legerement plus aigu
        pitch_variation=0.5,
        base_volume=0.95,
        volume_variation=0.08,
        pause_multiplier=0.9,  # Pauses plus courtes
        sentence_pause=0.3,
        paragraph_pause=0.6,
        breath_frequency=1.2,
        breath_intensity=0.4,
        emphasis_strength=1.0,
        enable_dynamic_variation=True,
        description="Naturel et decontracte, comme une conversation"
    ),

    NarrationStyle.DRAMATIC: StyleProsodyProfile(
        base_speed=0.92,
        speed_variation=0.12,  # Grande variation pour le drame
        base_pitch=0.0,
        pitch_variation=1.0,  # Grandes variations de pitch
        base_volume=1.05,
        volume_variation=0.15,  # Variations de volume importantes
        pause_multiplier=1.3,  # Pauses dramatiques
        sentence_pause=0.6,
        paragraph_pause=1.2,
        breath_frequency=1.0,
        breath_intensity=0.7,  # Respirations plus audibles
        emphasis_strength=1.5,  # Forte emphase
        enable_dynamic_variation=True,
        crescendo_on_climax=True,
        description="Intense et emotionnel, pour les moments forts"
    ),

    NarrationStyle.STORYTELLING: StyleProsodyProfile(
        base_speed=1.0,
        speed_variation=0.1,
        base_pitch=0.2,
        pitch_variation=0.7,  # Variations melodiques
        base_volume=1.0,
        volume_variation=0.1,
        pause_multiplier=1.1,
        sentence_pause=0.45,
        paragraph_pause=0.9,
        breath_frequency=1.0,
        breath_intensity=0.5,
        emphasis_strength=1.2,
        enable_dynamic_variation=True,
        crescendo_on_climax=True,
        description="Captivant avec variations dynamiques, ideal pour les histoires"
    ),

    NarrationStyle.DOCUMENTARY: StyleProsodyProfile(
        base_speed=1.0,
        speed_variation=0.03,
        base_pitch=-0.2,  # Legerement plus grave
        pitch_variation=0.2,
        base_volume=1.0,
        volume_variation=0.03,
        pause_multiplier=1.0,
        sentence_pause=0.4,
        paragraph_pause=0.8,
        breath_frequency=0.7,
        breath_intensity=0.3,
        emphasis_strength=0.8,
        enable_dynamic_variation=False,
        description="Informatif et neutre, style documentaire"
    ),

    NarrationStyle.INTIMATE: StyleProsodyProfile(
        base_speed=0.9,
        speed_variation=0.05,
        base_pitch=-0.3,
        pitch_variation=0.3,
        base_volume=0.85,  # Plus doux
        volume_variation=0.05,
        pause_multiplier=1.2,
        sentence_pause=0.5,
        paragraph_pause=1.0,
        breath_frequency=1.3,  # Respirations plus frequentes
        breath_intensity=0.6,
        emphasis_strength=0.9,
        enable_dynamic_variation=True,
        description="Proche et confidentiel, comme un secret partage"
    ),

    NarrationStyle.ENERGETIC: StyleProsodyProfile(
        base_speed=1.12,
        speed_variation=0.1,
        base_pitch=0.5,
        pitch_variation=0.8,
        base_volume=1.1,
        volume_variation=0.12,
        pause_multiplier=0.8,  # Pauses courtes
        sentence_pause=0.25,
        paragraph_pause=0.5,
        breath_frequency=0.9,
        breath_intensity=0.4,
        emphasis_strength=1.3,
        enable_dynamic_variation=True,
        description="Dynamique et enthousiaste, plein d'energie"
    ),
}


class NarrationStyleManager:
    """
    Gere l'application des styles de narration.

    Permet de:
    - Selectionner un style global
    - Ajuster dynamiquement selon le contexte
    - Mixer les styles pour des effets specifiques
    """

    def __init__(self, default_style: NarrationStyle = NarrationStyle.STORYTELLING):
        """
        Initialise le gestionnaire de styles.

        Args:
            default_style: Style par defaut
        """
        self.default_style = default_style
        self.current_style = default_style
        self._segment_counter = 0

    def get_profile(self, style: NarrationStyle = None) -> StyleProsodyProfile:
        """
        Retourne le profil prosodique pour un style.

        Args:
            style: Style demande (None = style courant)

        Returns:
            Profil prosodique du style
        """
        style = style or self.current_style
        return STYLE_PROFILES.get(style, STYLE_PROFILES[NarrationStyle.STORYTELLING])

    def apply_style_to_prosody(
        self,
        base_speed: float = 1.0,
        base_pitch: float = 0.0,
        base_volume: float = 1.0,
        style: NarrationStyle = None
    ) -> Dict[str, float]:
        """
        Applique un style a des parametres prosodiques de base.

        Args:
            base_speed: Vitesse de base
            base_pitch: Pitch de base
            base_volume: Volume de base
            style: Style a appliquer

        Returns:
            Dict avec speed, pitch, volume ajustes
        """
        profile = self.get_profile(style)
        self._segment_counter += 1

        # Appliquer les valeurs de base du style
        speed = base_speed * profile.base_speed
        pitch = base_pitch + profile.base_pitch
        volume = base_volume * profile.base_volume

        # Ajouter la variation si activee
        if profile.enable_dynamic_variation:
            speed += random.uniform(-profile.speed_variation, profile.speed_variation)
            pitch += random.uniform(-profile.pitch_variation, profile.pitch_variation)
            volume += random.uniform(-profile.volume_variation, profile.volume_variation)

        # Limiter les valeurs
        speed = max(0.5, min(1.5, speed))
        pitch = max(-3.0, min(3.0, pitch))
        volume = max(0.3, min(1.5, volume))

        return {
            "speed": speed,
            "pitch": pitch,
            "volume": volume
        }

    def get_pause_duration(
        self,
        pause_type: str = "sentence",
        style: NarrationStyle = None
    ) -> float:
        """
        Retourne la duree de pause pour un type donne.

        Args:
            pause_type: Type de pause (sentence, paragraph, breath)
            style: Style a utiliser

        Returns:
            Duree de la pause en secondes
        """
        profile = self.get_profile(style)

        base_pauses = {
            "sentence": profile.sentence_pause,
            "paragraph": profile.paragraph_pause,
            "comma": profile.sentence_pause * 0.5,
            "breath": 0.3,
            "dramatic": profile.sentence_pause * 2.0,
        }

        base = base_pauses.get(pause_type, profile.sentence_pause)

        # Appliquer le multiplicateur et variation
        result = base * profile.pause_multiplier

        if profile.enable_dynamic_variation:
            result *= random.uniform(0.9, 1.1)

        return result

    def get_emphasis_strength(self, style: NarrationStyle = None) -> float:
        """Retourne la force d'emphase pour le style."""
        return self.get_profile(style).emphasis_strength

    def get_breath_settings(self, style: NarrationStyle = None) -> Dict[str, float]:
        """Retourne les parametres de respiration pour le style."""
        profile = self.get_profile(style)
        return {
            "frequency": profile.breath_frequency,
            "intensity": profile.breath_intensity
        }

    def suggest_style_for_context(
        self,
        narrative_type: str,
        emotion: str = "neutral",
        is_dialogue: bool = False
    ) -> NarrationStyle:
        """
        Suggere un style base sur le contexte.

        Args:
            narrative_type: Type narratif (action, description, etc.)
            emotion: Emotion dominante
            is_dialogue: Est-ce un dialogue?

        Returns:
            Style suggere
        """
        # Dialogues -> conversationnel
        if is_dialogue:
            if emotion in ["anger", "fear", "excitement"]:
                return NarrationStyle.DRAMATIC
            return NarrationStyle.CONVERSATIONAL

        # Selon le type narratif
        narrative_mapping = {
            "action": NarrationStyle.ENERGETIC,
            "description": NarrationStyle.DOCUMENTARY,
            "introspection": NarrationStyle.INTIMATE,
            "flashback": NarrationStyle.INTIMATE,
            "suspense": NarrationStyle.DRAMATIC,
        }

        if narrative_type in narrative_mapping:
            return narrative_mapping[narrative_type]

        # Selon l'emotion
        emotion_mapping = {
            "fear": NarrationStyle.DRAMATIC,
            "anger": NarrationStyle.DRAMATIC,
            "sadness": NarrationStyle.INTIMATE,
            "joy": NarrationStyle.ENERGETIC,
            "excitement": NarrationStyle.ENERGETIC,
            "tenderness": NarrationStyle.INTIMATE,
            "suspense": NarrationStyle.DRAMATIC,
        }

        if emotion in emotion_mapping:
            return emotion_mapping[emotion]

        # Default
        return self.default_style

    def blend_styles(
        self,
        style1: NarrationStyle,
        style2: NarrationStyle,
        blend_factor: float = 0.5
    ) -> StyleProsodyProfile:
        """
        Melange deux styles.

        Args:
            style1: Premier style
            style2: Deuxieme style
            blend_factor: Facteur de melange (0 = style1, 1 = style2)

        Returns:
            Profil melange
        """
        p1 = self.get_profile(style1)
        p2 = self.get_profile(style2)

        def lerp(a, b, t):
            return a + (b - a) * t

        return StyleProsodyProfile(
            base_speed=lerp(p1.base_speed, p2.base_speed, blend_factor),
            speed_variation=lerp(p1.speed_variation, p2.speed_variation, blend_factor),
            base_pitch=lerp(p1.base_pitch, p2.base_pitch, blend_factor),
            pitch_variation=lerp(p1.pitch_variation, p2.pitch_variation, blend_factor),
            base_volume=lerp(p1.base_volume, p2.base_volume, blend_factor),
            volume_variation=lerp(p1.volume_variation, p2.volume_variation, blend_factor),
            pause_multiplier=lerp(p1.pause_multiplier, p2.pause_multiplier, blend_factor),
            sentence_pause=lerp(p1.sentence_pause, p2.sentence_pause, blend_factor),
            paragraph_pause=lerp(p1.paragraph_pause, p2.paragraph_pause, blend_factor),
            breath_frequency=lerp(p1.breath_frequency, p2.breath_frequency, blend_factor),
            breath_intensity=lerp(p1.breath_intensity, p2.breath_intensity, blend_factor),
            emphasis_strength=lerp(p1.emphasis_strength, p2.emphasis_strength, blend_factor),
            enable_dynamic_variation=p1.enable_dynamic_variation or p2.enable_dynamic_variation,
            crescendo_on_climax=p1.crescendo_on_climax or p2.crescendo_on_climax,
            description=f"Blend: {style1.value} + {style2.value}"
        )

    def set_style(self, style: NarrationStyle):
        """Definit le style courant."""
        self.current_style = style

    def reset(self):
        """Reinitialise le gestionnaire."""
        self.current_style = self.default_style
        self._segment_counter = 0


def get_style_description(style: NarrationStyle) -> str:
    """Retourne la description d'un style."""
    return STYLE_PROFILES[style].description


def list_available_styles() -> list[Dict]:
    """Liste tous les styles disponibles avec leurs descriptions."""
    return [
        {
            "name": style.value,
            "description": profile.description,
            "base_speed": profile.base_speed,
            "base_pitch": profile.base_pitch,
        }
        for style, profile in STYLE_PROFILES.items()
    ]


if __name__ == "__main__":
    print("=== Styles de Narration Disponibles ===\n")

    for style_info in list_available_styles():
        print(f"- {style_info['name']:15} : {style_info['description']}")

    print("\n=== Test Application de Style ===\n")

    manager = NarrationStyleManager(NarrationStyle.STORYTELLING)

    # Tester chaque style
    for style in NarrationStyle:
        manager.set_style(style)
        prosody = manager.apply_style_to_prosody()
        pause = manager.get_pause_duration("sentence")

        print(f"{style.value:15}: speed={prosody['speed']:.2f}, "
              f"pitch={prosody['pitch']:.2f}, pause={pause:.2f}s")
