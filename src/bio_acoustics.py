"""
Bio-Acoustics Generator v2.3.

Génère des sons "biologiques" synthétiques (respirations, bruits de bouche)
et d'ambiance (room tone) pour remplacer le silence numérique absolu.

L'objectif est d'éviter l'effet "coupure" entre les phrases et d'ajouter
de la présence humaine.

v2.3: Améliorations "Style ElevenLabs"
- Bruit rose (1/f spectrum) au lieu de bruit blanc
- Filtrage formant (résonances du tract vocal)
- Jitter d'amplitude (micro-variations naturelles)
"""
import numpy as np
import random
from typing import Literal, Optional
from scipy import signal

class BioAudioGenerator:
    def __init__(self, sample_rate: int = 24000, use_advanced_breaths: bool = True):
        self.sample_rate = sample_rate
        self.use_advanced_breaths = use_advanced_breaths

        # Formants respiratoires approximatifs (Hz)
        # Basés sur les résonances du tract vocal pendant la respiration
        self.breath_formants = [500, 1500, 2500]
        self.formant_bandwidths = [100, 150, 200]

    def _generate_pink_noise(self, num_samples: int) -> np.ndarray:
        """
        Génère du bruit rose (1/f spectrum).

        Le bruit rose a plus d'énergie dans les basses fréquences,
        ce qui sonne plus naturel que le bruit blanc pour les respirations.

        Méthode: Voss-McCartney algorithm (approximation efficace).
        """
        if num_samples <= 0:
            return np.array([], dtype=np.float32)

        # Nombre de rangées pour l'algorithme Voss-McCartney
        num_rows = 16

        # Initialiser avec du bruit blanc
        white = np.random.randn(num_samples)

        # Accumulateur pour le bruit rose
        pink = np.zeros(num_samples)

        # Valeurs des rangées (sources de bruit à différentes fréquences)
        rows = np.zeros(num_rows)
        row_values = np.random.randn(num_rows)

        for i in range(num_samples):
            # Trouver le bit le moins significatif qui change
            # Cela détermine quelle rangée mettre à jour
            for j in range(num_rows):
                if (i >> j) & 1:
                    rows[j] = np.random.randn()
                    break

            # Somme des rangées + bruit blanc
            pink[i] = np.sum(rows) + white[i]

        # Normaliser
        pink = pink / (num_rows + 1)

        return pink.astype(np.float32)

    def _generate_pink_noise_fast(self, num_samples: int) -> np.ndarray:
        """
        Génère du bruit rose via filtrage (méthode rapide).

        Utilise un filtre pour créer un spectre 1/f à partir de bruit blanc.
        Le bruit rose a plus d'énergie dans les basses fréquences.
        """
        if num_samples <= 0:
            return np.array([], dtype=np.float32)

        # Générer bruit blanc
        white = np.random.randn(num_samples)

        # Méthode: intégration partielle (cumsum partiel)
        # Cela booste les basses fréquences
        # Utiliser un facteur de "leak" pour éviter la dérive
        pink = np.zeros(num_samples)
        state = 0.0
        leak = 0.99  # Facteur de fuite pour stabilité

        for i in range(num_samples):
            state = leak * state + white[i]
            pink[i] = state

        # Ajouter le bruit blanc original pour restaurer les hautes fréquences
        # mais à niveau réduit (ratio typique pour 1/f)
        pink = pink * 0.7 + white * 0.3

        # Normaliser pour avoir une variance unitaire
        if np.std(pink) > 0:
            pink = pink / np.std(pink)

        return pink.astype(np.float32)

    def _apply_formant_filter(
        self,
        audio: np.ndarray,
        formants: list[float] = None,
        bandwidths: list[float] = None,
        strength: float = 0.5
    ) -> np.ndarray:
        """
        Applique un filtrage formant pour simuler les résonances du tract vocal.

        Les formants donnent au bruit un caractère plus "humain" et moins
        synthétique, comme s'il passait à travers une gorge.

        Args:
            audio: Signal d'entrée
            formants: Fréquences centrales des formants (Hz)
            bandwidths: Largeurs de bande des formants (Hz)
            strength: Force du filtrage (0=aucun, 1=complet)
        """
        if len(audio) == 0:
            return audio

        formants = formants or self.breath_formants
        bandwidths = bandwidths or self.formant_bandwidths

        # Mélanger signal original et filtré selon strength
        if strength <= 0:
            return audio

        filtered = np.zeros_like(audio)

        for freq, bw in zip(formants, bandwidths):
            # Créer un filtre passe-bande pour chaque formant
            # Normaliser les fréquences par rapport à Nyquist
            nyquist = self.sample_rate / 2

            # S'assurer que les fréquences sont dans les limites
            if freq >= nyquist:
                continue

            low = max((freq - bw/2) / nyquist, 0.01)
            high = min((freq + bw/2) / nyquist, 0.99)

            if low >= high:
                continue

            # Filtre Butterworth passe-bande
            try:
                b, a = signal.butter(2, [low, high], btype='band')
                formant_contribution = signal.lfilter(b, a, audio)
                filtered += formant_contribution
            except ValueError:
                # En cas d'erreur de filtrage, ignorer ce formant
                continue

        # Normaliser le signal filtré
        if np.max(np.abs(filtered)) > 0:
            filtered = filtered / np.max(np.abs(filtered)) * np.max(np.abs(audio))

        # Mélanger original et filtré
        result = (1 - strength) * audio + strength * filtered

        return result.astype(np.float32)

    def _add_amplitude_jitter(
        self,
        audio: np.ndarray,
        amount: float = 0.05,
        frequency: float = 20.0
    ) -> np.ndarray:
        """
        Ajoute des micro-variations d'amplitude (jitter).

        Simule l'irrégularité naturelle de la respiration humaine,
        où le flux d'air n'est jamais parfaitement constant.

        Args:
            audio: Signal d'entrée
            amount: Amplitude des variations (0.05 = ±5%)
            frequency: Fréquence approximative des variations (Hz)
        """
        if len(audio) == 0 or amount <= 0:
            return audio

        num_samples = len(audio)

        # Générer une modulation basse fréquence
        # Utiliser du bruit filtré pour des variations naturelles
        t = np.linspace(0, num_samples / self.sample_rate, num_samples)

        # Plusieurs composantes de fréquence pour un effet naturel
        jitter = np.zeros(num_samples)
        for freq_mult in [0.5, 1.0, 2.0, 3.0]:
            phase = random.uniform(0, 2 * np.pi)
            jitter += np.sin(2 * np.pi * frequency * freq_mult * t + phase) / freq_mult

        # Ajouter une composante aléatoire
        jitter += np.random.randn(num_samples) * 0.3

        # Lisser le jitter
        window_size = max(1, int(self.sample_rate / frequency / 2))
        if window_size > 1 and len(jitter) > window_size:
            jitter = np.convolve(jitter, np.ones(window_size)/window_size, mode='same')

        # Normaliser entre -1 et 1, puis appliquer amount
        if np.max(np.abs(jitter)) > 0:
            jitter = jitter / np.max(np.abs(jitter))

        # Moduler l'amplitude: 1 ± amount*jitter
        modulation = 1.0 + amount * jitter

        return (audio * modulation).astype(np.float32)

    def generate_silence(self, duration: float, noise_floor: float = 0.0001) -> np.ndarray:
        """
        Génère un 'silence' naturel (Room Tone).
        Au lieu de 0.0 absolu, on génère un bruit de fond très faible.
        """
        num_samples = int(duration * self.sample_rate)
        if num_samples <= 0:
            return np.array([], dtype=np.float32)
        
        # Bruit blanc très faible
        noise = np.random.normal(0, noise_floor, num_samples).astype(np.float32)
        return noise

    def generate_breath(
        self,
        duration: float = 0.4,
        intensity: float = 0.5,
        type: Literal["soft", "sharp", "deep", "gasp", "sigh"] = "soft",
        use_advanced: bool = None
    ) -> np.ndarray:
        """
        Génère un son de respiration synthétique.

        En mode avancé (v2.3): bruit rose + formants + jitter pour un son réaliste.
        En mode standard: bruit blanc filtré (compatibilité).

        Types disponibles:
        - soft: Respiration calme et régulière
        - sharp: Halètement court et rapide
        - deep: Respiration profonde
        - gasp: Inspiration soudaine (surprise/choc)
        - sigh: Soupir long et expressif

        Args:
            duration: Durée approximative (secondes)
            intensity: Intensité du son (0.0 à 1.0)
            type: Type de respiration
            use_advanced: Override pour le mode avancé (None = utiliser self.use_advanced_breaths)
        """
        # Durée variable légèrement aléatoire
        actual_duration = duration * random.uniform(0.9, 1.1)
        num_samples = int(actual_duration * self.sample_rate)

        if num_samples <= 0:
            return np.array([], dtype=np.float32)

        # Déterminer si on utilise le mode avancé
        advanced = use_advanced if use_advanced is not None else self.use_advanced_breaths

        # === Génération du bruit de base ===
        if advanced:
            # Mode avancé v2.3: bruit rose (plus naturel)
            noise = self._generate_pink_noise_fast(num_samples)
        else:
            # Mode standard: bruit blanc
            noise = np.random.normal(0, 1, num_samples).astype(np.float32)

        # === Configuration selon le type de respiration ===
        t = np.linspace(0, 1, num_samples)

        # Paramètres par défaut
        formant_strength = 0.4
        jitter_amount = 0.05

        if type == "soft":
            # Respiration calme: montée douce, descente douce
            envelope = np.sin(np.pi * t) ** 2
            volume_scale = 0.02 * intensity
            window_size = 30
            formant_strength = 0.3
            jitter_amount = 0.03
        elif type == "sharp":
            # Halètement: montée rapide, coupure
            envelope = np.power(t, 0.5) * np.power(1 - t, 4) * 5
            volume_scale = 0.05 * intensity
            window_size = 10
            formant_strength = 0.5
            jitter_amount = 0.08
        elif type == "deep":
            # Respiration profonde: longue montée, longue descente
            envelope = np.power(t, 2) * np.power(1 - t, 0.5) * 2
            volume_scale = 0.04 * intensity
            window_size = 30
            formant_strength = 0.5
            jitter_amount = 0.04
        elif type == "gasp":
            # Gasp: montée très rapide (surprise), pic au début
            envelope = np.exp(-5 * t) * (1 - np.exp(-20 * t)) * 3
            volume_scale = 0.06 * intensity
            window_size = 8  # Plus aigu
            formant_strength = 0.6
            jitter_amount = 0.10  # Plus de variation (panique)
        elif type == "sigh":
            # Soupir: attaque moyenne, longue décroissance exponentielle
            envelope = np.power(t, 0.3) * np.exp(-2 * t) * 2.5
            volume_scale = 0.035 * intensity
            window_size = 40  # Plus doux
            formant_strength = 0.4
            jitter_amount = 0.03  # Moins de variation (relâchement)
        else:
            envelope = np.ones_like(t)
            volume_scale = 0.02
            window_size = 30

        # === Application des traitements avancés ===
        if advanced:
            # Filtrage formant (résonances vocales)
            noise = self._apply_formant_filter(noise, strength=formant_strength)

        # Appliquer l'enveloppe
        breath = noise * envelope * volume_scale

        # Filtre Low-pass simple (moyenne mobile) pour adoucir
        breath = np.convolve(breath, np.ones(window_size)/window_size, mode='same')

        if advanced:
            # Jitter d'amplitude (micro-variations naturelles)
            breath = self._add_amplitude_jitter(breath, amount=jitter_amount)

        return breath.astype(np.float32)

    def generate_for_tag(self, tag: str, intensity: float = 0.5) -> Optional[np.ndarray]:
        """
        Génère l'audio approprié pour un tag expressif.

        Args:
            tag: Le tag audio (gasp, sigh, breath, etc.)
            intensity: Intensité du son (0.0 à 1.0)

        Returns:
            Audio numpy array ou None si tag non supporté
        """
        tag = tag.lower().strip()

        # Mapping des tags vers les types de respiration
        tag_mapping = {
            "gasp": ("gasp", 0.3),      # Court et soudain
            "sigh": ("sigh", 0.6),       # Long et expressif
            "breath": ("soft", 0.4),     # Respiration normale
            "breath:soft": ("soft", 0.3),
            "breath:deep": ("deep", 0.5),
            "inhale": ("sharp", 0.25),   # Inspiration
            "exhale": ("deep", 0.4),     # Expiration
        }

        if tag in tag_mapping:
            breath_type, duration = tag_mapping[tag]
            return self.generate_breath(
                duration=duration,
                intensity=intensity,
                type=breath_type
            )

        # Tags de pause (silence avec room tone)
        if tag in ("pause", "beat"):
            return self.generate_silence(0.3)
        if tag == "long pause":
            return self.generate_silence(0.8)
        if tag == "silence":
            return self.generate_silence(1.0)

        return None

    def generate_mouth_noise(self) -> np.ndarray:
        """
        Génère un petit "click" ou bruit de bouche très court (smack).
        """
        duration = random.uniform(0.01, 0.03)
        num_samples = int(duration * self.sample_rate)
        noise = np.random.normal(0, 0.05, num_samples).astype(np.float32)
        
        # Enveloppe très courte
        t = np.linspace(0, 1, num_samples)
        envelope = np.sin(np.pi * t) ** 10
        
        return (noise * envelope).astype(np.float32)

    def apply_crossfade(self, audio1: np.ndarray, audio2: np.ndarray, duration: float = 0.05) -> np.ndarray:
        """
        Concatène deux segments avec un vrai crossfade (overlap).

        Les deux signaux sont mélangés dans la zone de transition,
        produisant une transition plus naturelle qu'une simple concaténation.
        """
        fade_samples = int(duration * self.sample_rate)

        # Si les segments sont trop courts, concaténer sans fade
        if len(audio1) < fade_samples or len(audio2) < fade_samples:
            return np.concatenate([audio1, audio2])

        # Créer les courbes de fade (forme cosinus pour transition douce)
        t = np.linspace(0, np.pi / 2, fade_samples)
        fade_out = np.cos(t) ** 2  # 1 -> 0 (courbe douce)
        fade_in = np.sin(t) ** 2   # 0 -> 1 (courbe douce)

        # Copier pour éviter de modifier les originaux
        a1 = audio1.copy()
        a2 = audio2.copy()

        # Zone de crossfade: mixer les deux signaux
        crossfade_zone = (a1[-fade_samples:] * fade_out +
                         a2[:fade_samples] * fade_in)

        # Construire le résultat: début de a1 + zone mixée + fin de a2
        result = np.concatenate([
            a1[:-fade_samples],
            crossfade_zone,
            a2[fade_samples:]
        ])

        return result.astype(np.float32)

    def concatenate_with_crossfade(
        self,
        segments: list[np.ndarray],
        fade_duration: float = 0.03
    ) -> np.ndarray:
        """
        Concatène une liste de segments audio avec crossfade entre chacun.

        Args:
            segments: Liste de tableaux audio numpy
            fade_duration: Durée du crossfade en secondes

        Returns:
            Audio concaténé avec transitions douces
        """
        if not segments:
            return np.array([], dtype=np.float32)

        if len(segments) == 1:
            return segments[0].astype(np.float32)

        result = segments[0]
        for seg in segments[1:]:
            if len(seg) > 0:
                result = self.apply_crossfade(result, seg, fade_duration)

        return result.astype(np.float32)
