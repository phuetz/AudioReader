"""
Moteur Kokoro TTS - Haute qualité pour audiobooks.

Kokoro-82M: 82 millions de paramètres, qualité proche d'ElevenLabs.
Performance: ~5x temps réel sur CPU.

Fonctionnalités avancées:
- Voice blending (mélange de voix)
- Pauses intelligentes (phrases, paragraphes)
- Cohérence cross-chapitre
- Multi-voix par personnage
- Prosodie émotionnelle
"""
from pathlib import Path
from typing import Optional, List, Tuple, TYPE_CHECKING
import time
import re
import numpy as np

if TYPE_CHECKING:
    from .advanced_preprocessor import EnrichedSegment


# Voix disponibles
KOKORO_VOICES = {
    # Français
    "ff_siwis": {"name": "Siwis", "gender": "F", "lang": "fr-fr", "desc": "Femme (France)"},

    # Anglais US - Femmes
    "af_heart": {"name": "Heart", "gender": "F", "lang": "en-us", "desc": "Female (US) - Warm"},
    "af_bella": {"name": "Bella", "gender": "F", "lang": "en-us", "desc": "Female (US) - Soft"},
    "af_nicole": {"name": "Nicole", "gender": "F", "lang": "en-us", "desc": "Female (US) - Clear"},
    "af_nova": {"name": "Nova", "gender": "F", "lang": "en-us", "desc": "Female (US) - Bright"},
    "af_sky": {"name": "Sky", "gender": "F", "lang": "en-us", "desc": "Female (US) - Young"},
    "af_sarah": {"name": "Sarah", "gender": "F", "lang": "en-us", "desc": "Female (US) - Mature"},

    # Anglais US - Hommes
    "am_adam": {"name": "Adam", "gender": "M", "lang": "en-us", "desc": "Male (US) - Deep"},
    "am_michael": {"name": "Michael", "gender": "M", "lang": "en-us", "desc": "Male (US) - Warm"},
    "am_eric": {"name": "Eric", "gender": "M", "lang": "en-us", "desc": "Male (US) - Clear"},

    # Anglais UK
    "bf_emma": {"name": "Emma", "gender": "F", "lang": "en-gb", "desc": "Female (UK)"},
    "bf_isabella": {"name": "Isabella", "gender": "F", "lang": "en-gb", "desc": "Female (UK)"},
    "bm_george": {"name": "George", "gender": "M", "lang": "en-gb", "desc": "Male (UK)"},
    "bm_lewis": {"name": "Lewis", "gender": "M", "lang": "en-gb", "desc": "Male (UK)"},

    # Autres langues
    "jf_alpha": {"name": "Alpha", "gender": "F", "lang": "ja", "desc": "Female (Japan)"},
    "zf_xiaoxiao": {"name": "Xiaoxiao", "gender": "F", "lang": "zh", "desc": "Female (China)"},
}


def parse_voice_blend(voice_spec: str) -> List[Tuple[str, float]]:
    """
    Parse une spécification de mélange de voix.

    Formats supportés:
    - "af_bella" -> [("af_bella", 1.0)]
    - "af_bella:60,am_adam:40" -> [("af_bella", 0.6), ("am_adam", 0.4)]
    - "af_bella,am_adam" -> [("af_bella", 0.5), ("am_adam", 0.5)]

    Les poids sont validés et normalisés pour garantir une somme de 1.0.
    """
    if ':' not in voice_spec and ',' not in voice_spec:
        return [(voice_spec, 1.0)]

    voices = []
    parts = voice_spec.split(',')

    for part in parts:
        part = part.strip()
        if ':' in part:
            voice_id, weight = part.split(':')
            # Limiter chaque poids à 100% max
            w = min(float(weight) / 100.0, 1.0)
            voices.append((voice_id.strip(), w))
        else:
            voices.append((part.strip(), None))

    # Si pas de poids spécifiés, répartir équitablement
    unweighted = [v for v in voices if v[1] is None]
    if unweighted:
        total_weighted = sum(w for _, w in voices if w is not None)
        remaining = max(0.0, 1.0 - total_weighted)  # Éviter négatif
        equal_weight = remaining / len(unweighted) if unweighted else 0.0
        voices = [(v, w if w is not None else equal_weight) for v, w in voices]

    # Normaliser pour que la somme soit exactement 1.0
    total_weight = sum(w for _, w in voices)
    if total_weight > 0 and abs(total_weight - 1.0) > 0.001:
        voices = [(v, w / total_weight) for v, w in voices]

    return voices


class KokoroEngine:
    """
    Moteur TTS utilisant Kokoro-82M.

    Installation:
        pip install kokoro-onnx soundfile

    Téléchargement modèle:
        curl -L -o kokoro-v1.0.onnx "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
        curl -L -o voices-v1.0.bin "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

    Fonctionnalités:
        - Voice blending: "af_bella:60,am_adam:40"
        - Pauses intelligentes entre phrases/paragraphes
        - Cohérence cross-chapitre
    """

    MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
    VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

    # Durées des pauses en secondes
    PAUSE_SENTENCE = 0.3    # Pause entre phrases
    PAUSE_PARAGRAPH = 0.8   # Pause entre paragraphes
    PAUSE_CHAPTER = 2.0     # Pause début/fin chapitre

    def __init__(
        self,
        model_path: str = "kokoro-v1.0.onnx",
        voices_path: str = "voices-v1.0.bin",
        voice: str = "ff_siwis",
        speed: float = 1.0,
        sentence_pause: float = 0.3,
        paragraph_pause: float = 0.8,
        enhance_audio: bool = True,
        enhance_style: str = "broadcast"
    ):
        """
        Initialise le moteur Kokoro.

        Args:
            model_path: Chemin vers le modèle ONNX
            voices_path: Chemin vers le fichier des voix
            voice: Voix par défaut (ff_siwis pour français)
            speed: Vitesse de lecture (1.0 = normal)
            sentence_pause: Pause entre phrases (secondes)
            paragraph_pause: Pause entre paragraphes (secondes)
            enhance_audio: Activer le post-processing audio (recommandé)
            enhance_style: Style d'enhancement ("broadcast", "natural", "bright")
        """
        self.model_path = Path(model_path)
        self.voices_path = Path(voices_path)
        self.voice = voice
        self.speed = speed
        self.sentence_pause = sentence_pause
        self.paragraph_pause = paragraph_pause
        self.enhance_audio = enhance_audio
        self.enhance_style = enhance_style
        self._kokoro = None
        self._available = None
        self._voice_cache = {}  # Cache pour les voix blendées
        self._audio_processor = None

    def is_available(self) -> bool:
        """Vérifie si Kokoro est disponible."""
        if self._available is not None:
            return self._available

        try:
            import kokoro_onnx
            import soundfile
            self._available = (
                self.model_path.exists() and
                self.voices_path.exists()
            )
        except ImportError:
            self._available = False

        return self._available

    def _load_model(self):
        """Charge le modèle Kokoro."""
        if self._kokoro is not None:
            return self._kokoro

        if not self.is_available():
            raise RuntimeError("Kokoro non disponible. Vérifiez l'installation.")

        from kokoro_onnx import Kokoro
        print("Chargement du modèle Kokoro-82M...")
        self._kokoro = Kokoro(str(self.model_path), str(self.voices_path))
        return self._kokoro

    def get_lang(self) -> str:
        """Retourne le code langue pour la voix sélectionnée."""
        if self.voice in KOKORO_VOICES:
            return KOKORO_VOICES[self.voice]["lang"]
        # Déduire de la première lettre
        prefix = self.voice[:2] if len(self.voice) >= 2 else "en"
        lang_map = {
            "af": "en-us", "am": "en-us",
            "bf": "en-gb", "bm": "en-gb",
            "ff": "fr-fr",
            "jf": "ja", "jm": "ja",
            "zf": "zh", "zm": "zh",
        }
        return lang_map.get(prefix, "en-us")

    def _create_silence(self, duration: float, sample_rate: int = 24000) -> np.ndarray:
        """Crée un segment de silence."""
        return np.zeros(int(duration * sample_rate), dtype=np.float32)

    def _split_into_segments(self, text: str, max_chars: int = 500) -> List[Tuple[str, str]]:
        """
        Découpe le texte en segments avec type de pause.

        Limite chaque segment à max_chars caractères pour éviter les overflow phonèmes.

        Returns:
            Liste de tuples (texte, type_pause) où type_pause est 'sentence' ou 'paragraph'
        """
        segments = []

        # Découper par paragraphes (double saut de ligne)
        paragraphs = re.split(r'\n\s*\n', text)

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Découper par phrases
            sentences = re.split(r'(?<=[.!?])\s+', para)

            for i, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if not sentence:
                    continue

                # Si la phrase est trop longue, la découper davantage
                if len(sentence) > max_chars:
                    # Découper aux virgules ou points-virgules
                    sub_parts = re.split(r'(?<=[,;:])\s+', sentence)
                    current_chunk = ""

                    for part in sub_parts:
                        # Si le part lui-même est trop long, le découper par mots
                        if len(part) > max_chars:
                            if current_chunk:
                                segments.append((current_chunk, 'sentence'))
                                current_chunk = ""
                            # Découper par mots
                            words = part.split()
                            word_chunk = ""
                            for word in words:
                                if len(word_chunk) + len(word) + 1 < max_chars:
                                    word_chunk = (word_chunk + " " + word).strip()
                                else:
                                    if word_chunk:
                                        segments.append((word_chunk, 'sentence'))
                                    word_chunk = word
                            if word_chunk:
                                current_chunk = word_chunk
                        elif len(current_chunk) + len(part) < max_chars:
                            current_chunk = (current_chunk + " " + part).strip()
                        else:
                            if current_chunk:
                                segments.append((current_chunk, 'sentence'))
                            current_chunk = part

                    if current_chunk:
                        # Dernière partie du paragraphe
                        if i == len(sentences) - 1:
                            segments.append((current_chunk, 'paragraph'))
                        else:
                            segments.append((current_chunk, 'sentence'))
                else:
                    # Dernière phrase du paragraphe -> pause paragraphe
                    if i == len(sentences) - 1:
                        segments.append((sentence, 'paragraph'))
                    else:
                        segments.append((sentence, 'sentence'))

        return segments

    def synthesize(
        self,
        text: str,
        output_path: Path,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        add_smart_pauses: bool = True
    ) -> bool:
        """
        Synthétise le texte en audio.

        Args:
            text: Texte à synthétiser
            output_path: Chemin du fichier de sortie (.wav)
            voice: Voix à utiliser (optionnel, utilise celle par défaut)
            speed: Vitesse de lecture (optionnel)
            add_smart_pauses: Ajouter des pauses intelligentes entre phrases/paragraphes

        Returns:
            True si succès
        """
        try:
            import soundfile as sf

            kokoro = self._load_model()
            voice = voice or self.voice
            speed = speed or self.speed
            lang = self.get_lang()

            # Parser le voice blend si nécessaire
            voice_blend = parse_voice_blend(voice)
            primary_voice = voice_blend[0][0]  # Utiliser la première voix

            if add_smart_pauses:
                # Synthétiser segment par segment avec pauses
                all_samples = []
                segments = self._split_into_segments(text)
                sample_rate = 24000  # Kokoro utilise 24kHz

                for segment_text, pause_type in segments:
                    if not segment_text.strip():
                        continue

                    # Générer l'audio pour ce segment avec gestion d'erreur
                    try:
                        samples, sample_rate = kokoro.create(
                            segment_text,
                            voice=primary_voice,
                            speed=speed,
                            lang=lang
                        )
                        all_samples.append(samples)
                    except (IndexError, RuntimeError) as e:
                        # Si le segment échoue, essayer de le découper davantage
                        print(f"\n    [Retry] Segment trop long ({len(segment_text)} chars), redécoupage...")
                        words = segment_text.split()
                        mid = len(words) // 2
                        for sub_text in [" ".join(words[:mid]), " ".join(words[mid:])]:
                            if sub_text.strip():
                                try:
                                    sub_samples, sample_rate = kokoro.create(
                                        sub_text,
                                        voice=primary_voice,
                                        speed=speed,
                                        lang=lang
                                    )
                                    all_samples.append(sub_samples)
                                except Exception:
                                    print(f"    [Skip] Sous-segment impossible à convertir")

                    # Ajouter la pause appropriée
                    if pause_type == 'paragraph':
                        all_samples.append(self._create_silence(self.paragraph_pause, sample_rate))
                    else:
                        all_samples.append(self._create_silence(self.sentence_pause, sample_rate))

                # Concaténer tous les segments
                if all_samples:
                    final_samples = np.concatenate(all_samples)
                else:
                    final_samples = np.array([], dtype=np.float32)

            else:
                # Mode simple: synthétiser tout le texte d'un coup
                final_samples, sample_rate = kokoro.create(
                    text,
                    voice=primary_voice,
                    speed=speed,
                    lang=lang
                )

            # Post-processing audio (enhancement)
            if self.enhance_audio and len(final_samples) > 0:
                final_samples = self._enhance_audio(final_samples, sample_rate)

            # Sauvegarder
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), final_samples, sample_rate)

            return True

        except Exception as e:
            print(f"Erreur Kokoro: {e}")
            return False

    def _enhance_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Applique le post-processing audio pour améliorer la qualité."""
        try:
            from .audio_processor import VoiceEnhancer

            if self._audio_processor is None:
                self._audio_processor = VoiceEnhancer()

            return self._audio_processor.enhance(audio, sample_rate, self.enhance_style)
        except ImportError:
            # Module non disponible, retourner audio original
            return audio
        except Exception as e:
            print(f"Warning: Enhancement échoué: {e}")
            return audio

    def synthesize_with_chapter_pauses(
        self,
        text: str,
        output_path: Path,
        voice: Optional[str] = None,
        speed: Optional[float] = None
    ) -> bool:
        """
        Synthétise avec pauses de chapitre (silence au début et à la fin).

        Ajoute un silence de 0.75s au début et 2s à la fin pour le confort d'écoute.
        """
        try:
            import soundfile as sf

            kokoro = self._load_model()
            voice = voice or self.voice
            speed = speed or self.speed
            lang = self.get_lang()

            voice_blend = parse_voice_blend(voice)
            primary_voice = voice_blend[0][0]

            # Générer l'audio principal
            samples, sample_rate = kokoro.create(
                text,
                voice=primary_voice,
                speed=speed,
                lang=lang
            )

            # Ajouter silences début/fin
            silence_start = self._create_silence(0.75, sample_rate)
            silence_end = self._create_silence(2.0, sample_rate)

            final_samples = np.concatenate([silence_start, samples, silence_end])

            # Sauvegarder
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), final_samples, sample_rate)

            return True

        except Exception as e:
            print(f"Erreur Kokoro: {e}")
            return False

    @staticmethod
    def list_voices() -> None:
        """Affiche les voix disponibles."""
        print("\nVoix Kokoro disponibles:")
        print("-" * 60)

        current_lang = None
        for voice_id, info in sorted(KOKORO_VOICES.items(), key=lambda x: x[1]["lang"]):
            if info["lang"] != current_lang:
                current_lang = info["lang"]
                print(f"\n  [{current_lang.upper()}]")
            print(f"    {voice_id:15} - {info['name']:12} {info['desc']}")

    @staticmethod
    def download_model(target_dir: Path = Path(".")) -> bool:
        """Télécharge les fichiers du modèle."""
        import subprocess

        model_path = target_dir / "kokoro-v1.0.onnx"
        voices_path = target_dir / "voices-v1.0.bin"

        if model_path.exists() and voices_path.exists():
            print("Modèle déjà téléchargé.")
            return True

        print("Téléchargement du modèle Kokoro (~340MB)...")

        try:
            if not model_path.exists():
                subprocess.run([
                    "curl", "-L", "-o", str(model_path),
                    KokoroEngine.MODEL_URL
                ], check=True)

            if not voices_path.exists():
                subprocess.run([
                    "curl", "-L", "-o", str(voices_path),
                    KokoroEngine.VOICES_URL
                ], check=True)

            print("Téléchargement terminé.")
            return True

        except Exception as e:
            print(f"Erreur téléchargement: {e}")
            return False

    def synthesize_enriched_segments(
        self,
        segments: List["EnrichedSegment"],
        output_path: Path,
        progress_callback: Optional[callable] = None
    ) -> bool:
        """
        Synthétise une liste de segments enrichis avec multi-voix.

        Cette méthode permet de générer un audio avec:
        - Voix différentes par personnage
        - Pauses contextuelles
        - Prosodie émotionnelle (via vitesse)

        Args:
            segments: Liste de segments enrichis (depuis AdvancedPreprocessor)
            output_path: Chemin du fichier de sortie (.wav)
            progress_callback: Fonction appelée avec (index, total, segment)

        Returns:
            True si succès
        """
        try:
            import soundfile as sf

            kokoro = self._load_model()
            all_samples = []
            sample_rate = 24000

            total = len(segments)
            for i, seg in enumerate(segments):
                if progress_callback:
                    progress_callback(i, total, seg)

                # Pause avant le segment
                if seg.pause_before > 0:
                    all_samples.append(self._create_silence(seg.pause_before, sample_rate))

                # Déterminer la voix et vitesse
                voice_id = seg.voice_id or self.voice

                # Ajuster la vitesse selon la prosodie émotionnelle
                speed = self.speed
                if seg.prosody and seg.prosody.speed != 1.0:
                    speed = self.speed * seg.prosody.speed

                # Limiter la vitesse
                speed = max(0.7, min(1.5, speed))

                # Déterminer la langue depuis la voix
                lang = self._get_lang_for_voice(voice_id)

                # Synthétiser le segment
                text = seg.text.strip()
                if not text:
                    continue

                try:
                    samples, sample_rate = kokoro.create(
                        text,
                        voice=voice_id,
                        speed=speed,
                        lang=lang
                    )
                    all_samples.append(samples)
                except (IndexError, RuntimeError) as e:
                    # Segment trop long, découper
                    print(f"\n    [Retry] Segment {i+1} trop long, redécoupage...")
                    sub_samples = self._synthesize_long_segment(
                        kokoro, text, voice_id, speed, lang, sample_rate
                    )
                    if sub_samples is not None:
                        all_samples.append(sub_samples)

                # Pause après le segment
                pause_after = seg.pause_after if seg.pause_after > 0 else self.sentence_pause
                all_samples.append(self._create_silence(pause_after, sample_rate))

            # Concaténer et sauvegarder
            if all_samples:
                final_samples = np.concatenate(all_samples)
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(output_path), final_samples, sample_rate)
                return True

            return False

        except Exception as e:
            print(f"Erreur synthèse multi-voix: {e}")
            return False

    def _get_lang_for_voice(self, voice_id: str) -> str:
        """Retourne le code langue pour une voix donnée."""
        if voice_id in KOKORO_VOICES:
            return KOKORO_VOICES[voice_id]["lang"]
        # Déduire de la première lettre
        prefix = voice_id[:2] if len(voice_id) >= 2 else "en"
        lang_map = {
            "af": "en-us", "am": "en-us",
            "bf": "en-gb", "bm": "en-gb",
            "ff": "fr-fr",
            "jf": "ja", "jm": "ja",
            "zf": "zh", "zm": "zh",
        }
        return lang_map.get(prefix, "en-us")

    def _synthesize_long_segment(
        self,
        kokoro,
        text: str,
        voice: str,
        speed: float,
        lang: str,
        sample_rate: int
    ) -> Optional[np.ndarray]:
        """Synthétise un segment trop long en le découpant."""
        words = text.split()
        if len(words) < 4:
            return None

        mid = len(words) // 2
        parts = [" ".join(words[:mid]), " ".join(words[mid:])]

        samples_list = []
        for part in parts:
            if not part.strip():
                continue
            try:
                samples, _ = kokoro.create(
                    part,
                    voice=voice,
                    speed=speed,
                    lang=lang
                )
                samples_list.append(samples)
                # Petite pause entre les parties
                samples_list.append(self._create_silence(0.1, sample_rate))
            except Exception:
                print(f"    [Skip] Sous-segment impossible")

        if samples_list:
            return np.concatenate(samples_list)
        return None


def get_installation_guide() -> str:
    """Guide d'installation Kokoro."""
    return """
╔══════════════════════════════════════════════════════════════════════════════╗
║                         INSTALLATION KOKORO TTS                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  1. Installer les dépendances Python:                                        ║
║     pip install kokoro-onnx soundfile                                        ║
║                                                                              ║
║  2. Télécharger le modèle (~340MB total):                                    ║
║     curl -L -o kokoro-v1.0.onnx \\                                           ║
║       "https://github.com/thewh1teagle/kokoro-onnx/releases/download/\\      ║
║        model-files-v1.0/kokoro-v1.0.onnx"                                    ║
║                                                                              ║
║     curl -L -o voices-v1.0.bin \\                                            ║
║       "https://github.com/thewh1teagle/kokoro-onnx/releases/download/\\      ║
║        model-files-v1.0/voices-v1.0.bin"                                     ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  VOIX DISPONIBLES                                                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Français:  ff_siwis (Femme)                                                 ║
║  English:   af_heart, af_bella, am_adam, am_michael, bf_emma, bm_george     ║
║  日本語:     jf_alpha                                                         ║
║  中文:       zf_xiaoxiao                                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


if __name__ == "__main__":
    print(get_installation_guide())
    KokoroEngine.list_voices()
