"""
Breath Sample Manager v2.3.

Gestionnaire de samples audio pour les respirations.
Permet d'utiliser des enregistrements réels au lieu de synthèse.

Si le dossier de samples n'existe pas ou est vide, le système
retourne automatiquement vers la synthèse (fallback).
"""
import numpy as np
import random
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class BreathSampleManager:
    """
    Gère le chargement et la sélection de samples audio pour les respirations.

    Structure attendue du dossier samples:
    samples/breaths/
    ├── soft/
    │   ├── breath_soft_01.wav
    │   └── breath_soft_02.wav
    ├── gasp/
    ├── sigh/
    └── deep/
    """

    def __init__(
        self,
        samples_dir: Optional[Path] = None,
        sample_rate: int = 24000
    ):
        """
        Initialise le gestionnaire de samples.

        Args:
            samples_dir: Chemin vers le dossier de samples (None = synthèse uniquement)
            sample_rate: Taux d'échantillonnage cible
        """
        self.sample_rate = sample_rate
        self.samples: dict[str, list[np.ndarray]] = {}
        self.samples_dir = samples_dir

        if samples_dir and Path(samples_dir).exists():
            self._load_samples(Path(samples_dir))

    def _load_samples(self, samples_dir: Path) -> None:
        """
        Charge tous les samples depuis le dossier.

        Supporte les formats WAV et FLAC.
        Les samples sont resampleés si nécessaire.
        """
        try:
            import soundfile as sf
        except ImportError:
            logger.warning("soundfile non disponible, samples désactivés")
            return

        # Types de respiration supportés
        breath_types = ["soft", "sharp", "deep", "gasp", "sigh"]

        for breath_type in breath_types:
            type_dir = samples_dir / breath_type
            if not type_dir.exists():
                continue

            self.samples[breath_type] = []

            # Charger tous les fichiers audio du dossier
            for audio_file in type_dir.glob("*.wav"):
                try:
                    audio, sr = sf.read(audio_file)

                    # Convertir en mono si nécessaire
                    if len(audio.shape) > 1:
                        audio = np.mean(audio, axis=1)

                    # Resample si nécessaire
                    if sr != self.sample_rate:
                        audio = self._resample(audio, sr, self.sample_rate)

                    # Normaliser
                    if np.max(np.abs(audio)) > 0:
                        audio = audio / np.max(np.abs(audio))

                    self.samples[breath_type].append(audio.astype(np.float32))
                    logger.debug(f"Sample chargé: {audio_file}")

                except Exception as e:
                    logger.warning(f"Erreur chargement {audio_file}: {e}")

            # Charger aussi les fichiers FLAC
            for audio_file in type_dir.glob("*.flac"):
                try:
                    audio, sr = sf.read(audio_file)

                    if len(audio.shape) > 1:
                        audio = np.mean(audio, axis=1)

                    if sr != self.sample_rate:
                        audio = self._resample(audio, sr, self.sample_rate)

                    if np.max(np.abs(audio)) > 0:
                        audio = audio / np.max(np.abs(audio))

                    self.samples[breath_type].append(audio.astype(np.float32))

                except Exception as e:
                    logger.warning(f"Erreur chargement {audio_file}: {e}")

            if self.samples[breath_type]:
                logger.info(
                    f"Chargé {len(self.samples[breath_type])} samples pour '{breath_type}'"
                )

    def _resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """
        Resample l'audio vers le taux d'échantillonnage cible.

        Utilise librosa si disponible, sinon interpolation simple.
        """
        try:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            # Interpolation linéaire simple (moins précise)
            duration = len(audio) / orig_sr
            target_length = int(duration * target_sr)
            indices = np.linspace(0, len(audio) - 1, target_length)
            return np.interp(indices, np.arange(len(audio)), audio)

    def get_breath(
        self,
        breath_type: str,
        intensity: float = 0.5
    ) -> Optional[np.ndarray]:
        """
        Retourne un sample de respiration aléatoire du type demandé.

        Args:
            breath_type: Type de respiration (soft, gasp, sigh, etc.)
            intensity: Intensité (0.0 à 1.0), ajuste le volume

        Returns:
            Audio numpy array ou None si aucun sample disponible
        """
        if breath_type not in self.samples or not self.samples[breath_type]:
            return None

        # Sélectionner un sample aléatoire
        sample = random.choice(self.samples[breath_type]).copy()

        # Ajuster l'intensité (volume)
        sample = sample * intensity

        return sample

    def has_samples(self, breath_type: str = None) -> bool:
        """
        Vérifie si des samples sont disponibles.

        Args:
            breath_type: Type spécifique à vérifier (None = n'importe quel type)

        Returns:
            True si des samples sont disponibles
        """
        if breath_type:
            return breath_type in self.samples and len(self.samples[breath_type]) > 0

        return any(len(samples) > 0 for samples in self.samples.values())

    def list_available_types(self) -> list[str]:
        """
        Liste les types de respiration avec des samples disponibles.

        Returns:
            Liste des types disponibles
        """
        return [t for t, s in self.samples.items() if len(s) > 0]

    def get_sample_count(self, breath_type: str = None) -> int:
        """
        Compte le nombre de samples disponibles.

        Args:
            breath_type: Type spécifique (None = total)

        Returns:
            Nombre de samples
        """
        if breath_type:
            return len(self.samples.get(breath_type, []))

        return sum(len(s) for s in self.samples.values())


class HybridBreathGenerator:
    """
    Générateur hybride: samples réels + synthèse.

    Utilise des samples quand disponibles, sinon synthèse avancée.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        samples_dir: Optional[Path] = None,
        prefer_samples: bool = True
    ):
        """
        Initialise le générateur hybride.

        Args:
            sample_rate: Taux d'échantillonnage
            samples_dir: Dossier de samples (optionnel)
            prefer_samples: Préférer les samples à la synthèse quand disponibles
        """
        from .bio_acoustics import BioAudioGenerator

        self.sample_rate = sample_rate
        self.prefer_samples = prefer_samples

        # Gestionnaire de samples
        self.sample_manager = BreathSampleManager(samples_dir, sample_rate)

        # Générateur synthétique (fallback)
        self.synth_generator = BioAudioGenerator(sample_rate, use_advanced_breaths=True)

    def generate_breath(
        self,
        breath_type: str = "soft",
        duration: float = 0.4,
        intensity: float = 0.5
    ) -> np.ndarray:
        """
        Génère une respiration (sample ou synthèse).

        Args:
            breath_type: Type de respiration
            duration: Durée cible (pour synthèse)
            intensity: Intensité du son

        Returns:
            Audio numpy array
        """
        # Essayer les samples d'abord si préféré
        if self.prefer_samples and self.sample_manager.has_samples(breath_type):
            sample = self.sample_manager.get_breath(breath_type, intensity)
            if sample is not None:
                return sample

        # Fallback vers synthèse avancée
        return self.synth_generator.generate_breath(
            duration=duration,
            intensity=intensity,
            type=breath_type
        )

    def generate_for_tag(self, tag: str, intensity: float = 0.5) -> Optional[np.ndarray]:
        """
        Génère l'audio pour un tag expressif.

        Args:
            tag: Tag audio (gasp, sigh, breath, etc.)
            intensity: Intensité du son

        Returns:
            Audio numpy array ou None si non supporté
        """
        # Mapping des tags vers les types
        tag_mapping = {
            "gasp": ("gasp", 0.3),
            "sigh": ("sigh", 0.6),
            "breath": ("soft", 0.4),
            "breath:soft": ("soft", 0.3),
            "breath:deep": ("deep", 0.5),
            "inhale": ("sharp", 0.25),
            "exhale": ("deep", 0.4),
        }

        tag = tag.lower().strip()

        if tag in tag_mapping:
            breath_type, duration = tag_mapping[tag]
            return self.generate_breath(breath_type, duration, intensity)

        # Déléguer les autres tags au générateur synthétique
        return self.synth_generator.generate_for_tag(tag, intensity)
