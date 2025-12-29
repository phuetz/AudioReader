"""
Voice Morphing - Modification de voix en temps reel.

Fonctionnalites:
- Pitch shifting (hauteur de la voix)
- Formant shifting (timbre vocal)
- Time stretching (vitesse sans changer le pitch)
- Variation de stabilite (plus ou moins expressif)
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path


@dataclass
class VoiceMorphSettings:
    """Parametres de morphing vocal."""
    pitch_shift: float = 0.0       # Demi-tons (-12 a +12)
    formant_shift: float = 1.0     # Ratio (0.5 a 2.0, 1.0 = normal)
    time_stretch: float = 1.0      # Ratio (0.5 a 2.0, 1.0 = normal)
    stability: float = 0.7         # 0.0 = tres expressif, 1.0 = stable
    breathiness: float = 0.0       # 0.0 a 1.0 (souffle dans la voix)
    roughness: float = 0.0         # 0.0 a 1.0 (voix rauque)


class VoiceMorpher:
    """
    Applique des transformations vocales a l'audio.

    Utilise des techniques DSP pour modifier:
    - La hauteur (pitch)
    - Le timbre (formants)
    - La duree (time stretch)
    """

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self._has_librosa = None
        self._has_soundfile = None

    def _check_dependencies(self) -> bool:
        """Verifie les dependances."""
        if self._has_librosa is None:
            try:
                import librosa
                self._has_librosa = True
            except ImportError:
                self._has_librosa = False

        if self._has_soundfile is None:
            try:
                import soundfile
                self._has_soundfile = True
            except ImportError:
                self._has_soundfile = False

        return self._has_librosa and self._has_soundfile

    def pitch_shift(
        self,
        audio: np.ndarray,
        semitones: float
    ) -> np.ndarray:
        """
        Decale le pitch de l'audio.

        Args:
            audio: Signal audio numpy
            semitones: Nombre de demi-tons (+/- 12)

        Returns:
            Audio avec pitch modifie
        """
        if not self._check_dependencies():
            print("Warning: librosa non installe, pitch shift ignore")
            return audio

        if abs(semitones) < 0.1:
            return audio

        import librosa

        # Limiter les valeurs extremes
        semitones = max(-12, min(12, semitones))

        # Appliquer le pitch shift
        return librosa.effects.pitch_shift(
            audio,
            sr=self.sample_rate,
            n_steps=semitones
        )

    def time_stretch(
        self,
        audio: np.ndarray,
        rate: float
    ) -> np.ndarray:
        """
        Modifie la vitesse sans changer le pitch.

        Args:
            audio: Signal audio
            rate: Facteur de vitesse (>1 = plus rapide)

        Returns:
            Audio avec duree modifiee
        """
        if not self._check_dependencies():
            print("Warning: librosa non installe, time stretch ignore")
            return audio

        if abs(rate - 1.0) < 0.05:
            return audio

        import librosa

        # Limiter les valeurs
        rate = max(0.5, min(2.0, rate))

        return librosa.effects.time_stretch(audio, rate=rate)

    def formant_shift(
        self,
        audio: np.ndarray,
        shift_ratio: float
    ) -> np.ndarray:
        """
        Modifie les formants (timbre) de la voix.

        Args:
            audio: Signal audio
            shift_ratio: Ratio de decalage (1.0 = normal)
                        <1.0 = voix plus grave/masculine
                        >1.0 = voix plus aigue/feminine

        Returns:
            Audio avec formants modifies
        """
        if not self._check_dependencies():
            return audio

        if abs(shift_ratio - 1.0) < 0.05:
            return audio

        import librosa

        # Limiter les valeurs
        shift_ratio = max(0.5, min(2.0, shift_ratio))

        # Technique: pitch shift + time stretch inverse
        # Cela preserve le pitch mais change les formants
        semitones = 12 * np.log2(shift_ratio)

        # Shift pitch
        shifted = librosa.effects.pitch_shift(
            audio,
            sr=self.sample_rate,
            n_steps=semitones
        )

        # Compenser avec time stretch
        return librosa.effects.time_stretch(shifted, rate=shift_ratio)

    def add_breathiness(
        self,
        audio: np.ndarray,
        amount: float
    ) -> np.ndarray:
        """
        Ajoute du souffle a la voix.

        Args:
            audio: Signal audio
            amount: Intensite (0.0 a 1.0)

        Returns:
            Audio avec souffle ajoute
        """
        if amount < 0.01:
            return audio

        amount = max(0.0, min(1.0, amount))

        # Generer du bruit filtre (simule le souffle)
        noise = np.random.randn(len(audio)) * 0.1

        # Filtre passe-haut pour le bruit (souffle = hautes frequences)
        # Approximation simple avec moyenne mobile
        window = int(self.sample_rate * 0.001)  # 1ms
        if window > 1:
            kernel = np.ones(window) / window
            noise_lp = np.convolve(noise, kernel, mode='same')
            noise = noise - noise_lp  # Garde les hautes frequences

        # Moduler par l'enveloppe du signal original
        envelope = np.abs(audio)
        window = int(self.sample_rate * 0.02)  # 20ms
        if window > 1:
            kernel = np.ones(window) / window
            envelope = np.convolve(envelope, kernel, mode='same')

        # Mixer
        breathiness = noise * envelope * amount * 3
        return audio + breathiness.astype(audio.dtype)

    def add_roughness(
        self,
        audio: np.ndarray,
        amount: float
    ) -> np.ndarray:
        """
        Ajoute de la rugosite a la voix.

        Args:
            audio: Signal audio
            amount: Intensite (0.0 a 1.0)

        Returns:
            Audio avec voix plus rauque
        """
        if amount < 0.01:
            return audio

        amount = max(0.0, min(1.0, amount))

        # Distorsion douce (simule la voix rauque)
        threshold = 1.0 - (amount * 0.5)

        # Soft clipping
        clipped = np.tanh(audio / threshold) * threshold

        # Mixer avec l'original
        return audio * (1 - amount * 0.5) + clipped * (amount * 0.5)

    def apply_variation(
        self,
        audio: np.ndarray,
        stability: float
    ) -> np.ndarray:
        """
        Applique une variation pour rendre la voix plus expressive.

        Args:
            audio: Signal audio
            stability: 0.0 = tres variable, 1.0 = stable

        Returns:
            Audio avec variation appliquee
        """
        if stability > 0.95:
            return audio

        # Variation = micro-fluctuations de pitch
        variation_amount = (1.0 - stability) * 0.3

        # Generer une courbe de variation lente
        num_points = max(10, len(audio) // (self.sample_rate // 10))
        variation_curve = np.random.randn(num_points) * variation_amount

        # Interpoler pour tout le signal
        x_orig = np.linspace(0, 1, num_points)
        x_new = np.linspace(0, 1, len(audio))
        variation = np.interp(x_new, x_orig, variation_curve)

        # Appliquer comme modulation de phase (subtil)
        # Ceci cree une variation naturelle sans artefacts evidents
        modulated = audio * (1.0 + variation * 0.1)

        return np.clip(modulated, -1.0, 1.0).astype(audio.dtype)

    def morph(
        self,
        audio: np.ndarray,
        settings: VoiceMorphSettings
    ) -> np.ndarray:
        """
        Applique toutes les transformations de morphing.

        Args:
            audio: Signal audio original
            settings: Parametres de morphing

        Returns:
            Audio transforme
        """
        result = audio.copy()

        # Ordre important des operations:

        # 1. Time stretch (avant pitch pour qualite)
        if abs(settings.time_stretch - 1.0) > 0.05:
            result = self.time_stretch(result, settings.time_stretch)

        # 2. Pitch shift
        if abs(settings.pitch_shift) > 0.1:
            result = self.pitch_shift(result, settings.pitch_shift)

        # 3. Formant shift
        if abs(settings.formant_shift - 1.0) > 0.05:
            result = self.formant_shift(result, settings.formant_shift)

        # 4. Variation de stabilite
        if settings.stability < 0.95:
            result = self.apply_variation(result, settings.stability)

        # 5. Effets additifs
        if settings.breathiness > 0.01:
            result = self.add_breathiness(result, settings.breathiness)

        if settings.roughness > 0.01:
            result = self.add_roughness(result, settings.roughness)

        return result


class VoicePresets:
    """Presets de voix predefinies."""

    PRESETS = {
        # Modifications de genre
        "more_masculine": VoiceMorphSettings(
            pitch_shift=-3.0,
            formant_shift=0.85,
        ),
        "more_feminine": VoiceMorphSettings(
            pitch_shift=3.0,
            formant_shift=1.15,
        ),

        # Ages
        "younger": VoiceMorphSettings(
            pitch_shift=2.0,
            formant_shift=1.1,
            stability=0.6,
        ),
        "older": VoiceMorphSettings(
            pitch_shift=-2.0,
            formant_shift=0.9,
            breathiness=0.2,
            roughness=0.1,
        ),

        # Styles
        "whisper": VoiceMorphSettings(
            breathiness=0.6,
            stability=0.9,
            formant_shift=1.0,
        ),
        "rough": VoiceMorphSettings(
            roughness=0.4,
            pitch_shift=-1.0,
        ),
        "expressive": VoiceMorphSettings(
            stability=0.3,
        ),
        "robotic": VoiceMorphSettings(
            stability=1.0,
            formant_shift=0.95,
        ),

        # Emotions via morphing
        "excited_morph": VoiceMorphSettings(
            pitch_shift=1.5,
            time_stretch=0.9,
            stability=0.5,
        ),
        "sad_morph": VoiceMorphSettings(
            pitch_shift=-1.0,
            time_stretch=1.15,
            breathiness=0.15,
            stability=0.8,
        ),
        "angry_morph": VoiceMorphSettings(
            pitch_shift=0.5,
            roughness=0.2,
            time_stretch=0.95,
            stability=0.6,
        ),
        "fearful_morph": VoiceMorphSettings(
            pitch_shift=2.0,
            time_stretch=0.85,
            stability=0.4,
            breathiness=0.1,
        ),
    }

    @classmethod
    def get(cls, name: str) -> Optional[VoiceMorphSettings]:
        """Recupere un preset par son nom."""
        return cls.PRESETS.get(name)

    @classmethod
    def list_presets(cls) -> None:
        """Affiche tous les presets disponibles."""
        print("\n=== Presets de Voice Morphing ===\n")
        for name, settings in cls.PRESETS.items():
            mods = []
            if settings.pitch_shift != 0:
                mods.append(f"pitch:{settings.pitch_shift:+.1f}")
            if settings.formant_shift != 1.0:
                mods.append(f"formant:{settings.formant_shift:.2f}")
            if settings.time_stretch != 1.0:
                mods.append(f"time:{settings.time_stretch:.2f}")
            if settings.stability != 0.7:
                mods.append(f"stab:{settings.stability:.1f}")
            if settings.breathiness > 0:
                mods.append(f"breath:{settings.breathiness:.1f}")
            if settings.roughness > 0:
                mods.append(f"rough:{settings.roughness:.1f}")

            print(f"  {name:20}: {', '.join(mods)}")


def morph_audio_file(
    input_path: Path,
    output_path: Path,
    settings: VoiceMorphSettings
) -> bool:
    """
    Applique le morphing a un fichier audio.

    Args:
        input_path: Fichier d'entree
        output_path: Fichier de sortie
        settings: Parametres de morphing

    Returns:
        True si succes
    """
    try:
        import soundfile as sf

        # Charger
        audio, sr = sf.read(str(input_path))

        # Morphing
        morpher = VoiceMorpher(sample_rate=sr)
        morphed = morpher.morph(audio, settings)

        # Sauvegarder
        sf.write(str(output_path), morphed, sr)

        return True

    except Exception as e:
        print(f"Erreur morphing: {e}")
        return False


if __name__ == "__main__":
    VoicePresets.list_presets()

    print("\n=== Test Voice Morphing ===")
    print("Necessite librosa: pip install librosa")
