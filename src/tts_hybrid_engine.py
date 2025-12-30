"""
Moteur TTS Hybride : MMS (narrateur) + Kokoro (dialogues).

Combine le meilleur des deux mondes:
- MMS-TTS: Qualité native française pour la narration
- Kokoro: Voix distinctes pour les personnages

Usage:
    from src.tts_hybrid_engine import HybridTTSEngine

    engine = HybridTTSEngine()
    engine.synthesize_chapter(text, "output.wav")  # WAV
    engine.synthesize_chapter(text, "output.mp3", output_format="mp3")  # MP3
"""
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Set, Tuple, Callable, Literal
from dataclasses import dataclass, field
import re
import subprocess
import shutil

# Import du crossfade
try:
    from .audio_crossfade import apply_crossfade_to_chapter, AudioCrossfader
    HAS_CROSSFADE = True
except ImportError:
    HAS_CROSSFADE = False

# Import des corrections (optionnel)
try:
    from .corrections_loader import apply_default_corrections, create_correction_func
    HAS_CORRECTIONS = True
except ImportError:
    try:
        from .corrections_conquerants import apply_corrections as apply_default_corrections
        create_correction_func = None
        HAS_CORRECTIONS = True
    except ImportError:
        HAS_CORRECTIONS = False
        apply_default_corrections = None
        create_correction_func = None


@dataclass
class DialogueSegment:
    """Segment de dialogue ou narration."""
    type: str  # 'narration' ou 'dialogue'
    text: str
    speaker: str
    voice: Optional[str] = None


@dataclass
class Character:
    """Personnage détecté."""
    name: str
    gender: Optional[str] = None  # 'M', 'F', ou None
    voice: Optional[str] = None
    occurrences: int = 0


class HybridTTSEngine:
    """
    Moteur TTS hybride combinant MMS et Kokoro.

    - Narration → MMS-TTS (français natif)
    - Dialogues → Kokoro (voix par personnage)
    """

    # Mots à ignorer (faux positifs fréquents)
    IGNORED_WORDS = {
        # Mots courants
        "pas", "plus", "moins", "très", "tout", "tous", "bien", "mal",
        "oui", "non", "peut", "être", "fait", "faire", "dit", "dire",
        "voir", "rien", "quoi", "donc", "mais", "avec", "sans", "pour",
        "dans", "sur", "sous", "par", "entre", "vers", "chez",
        # Pronoms et déterminants
        "il", "elle", "ils", "elles", "lui", "leur", "son", "sa", "ses",
        "mon", "ma", "mes", "ton", "ta", "tes", "notre", "votre",
        "ce", "cette", "ces", "un", "une", "des", "le", "la", "les",
        # Adverbes
        "encore", "jamais", "toujours", "déjà", "aussi", "ainsi",
        "alors", "après", "avant", "maintenant", "soudain", "enfin",
        # Autres
        "autre", "même", "seul", "petit", "grand", "vieux", "jeune",
        "homme", "femme", "type", "gars", "mec", "fille",
    }

    # Verbes de parole français
    SPEECH_VERBS = [
        "dit", "disait", "répondit", "répondait", "demanda", "demandait",
        "murmura", "murmurait", "cria", "criait", "chuchota", "chuchotait",
        "s'exclama", "s'exclamait", "hurla", "hurlait", "soupira", "soupirait",
        "lança", "lançait", "ajouta", "ajoutait", "reprit", "reprenait",
        "continua", "continuait", "expliqua", "expliquait", "annonça",
        "avoua", "avouait", "supplia", "suppliait", "grogna", "grognait",
        "ricana", "ricanait", "souffla", "soufflait", "articula",
        "bégaya", "balbutia", "grommela", "marmonna", "protesta",
        "objecta", "confirma", "rétorqua", "répliqua", "interrogea",
        "questionna", "ordonna", "commanda", "proposa", "suggéra",
        "observa", "nota", "remarqua", "constata", "admit", "concéda",
        "fit", "faisait",  # "fit-il", "fit-elle"
    ]

    # Prénoms pour détection de genre
    FEMALE_NAMES = {
        "marie", "sophie", "claire", "julie", "emma", "lucie", "alice",
        "léa", "camille", "charlotte", "isabelle", "catherine", "anne",
        "jeanne", "marguerite", "hélène", "élise", "louise", "amélie",
        "sarah", "bella", "nicole", "nova", "céline", "nathalie",
        "sylvie", "valérie", "christine", "françoise", "monique",
        "patricia", "sandrine", "véronique", "caroline", "stéphanie",
        "audrey", "laura", "marine", "pauline", "manon", "chloé",
        "lisa", "nina", "eva", "léonie", "jade", "lina", "anna",
    }

    MALE_NAMES = {
        "pierre", "jean", "paul", "jacques", "michel", "philippe", "henri",
        "louis", "françois", "marc", "charles", "nicolas", "antoine",
        "thomas", "julien", "alexandre", "guillaume", "olivier", "éric",
        "adam", "george", "lewis", "michael", "eric", "victor", "hugo",
        "david", "patrick", "bernard", "alain", "christophe", "laurent",
        "stéphane", "frédéric", "sébastien", "jérôme", "thierry",
        "pascal", "bruno", "didier", "gilles", "yves", "denis",
        "kamel", "mohamed", "ahmed", "omar", "ali", "rachid",
        "crawford", "james", "john", "william", "robert", "richard",
        "momo", "paulo",
    }

    # Voix Kokoro par défaut
    VOICE_POOL_FEMALE = ["af_heart", "af_sarah", "af_bella", "af_nicole"]
    VOICE_POOL_MALE = ["am_adam", "bm_george", "am_michael", "am_eric"]

    def __init__(
        self,
        mms_language: str = "fra",
        voice_mapping: Optional[Dict[str, str]] = None,
        narrator_speed: float = 1.0,
        dialogue_speed: float = 1.0,
        apply_corrections: bool = True,
        corrections_func: Optional[Callable[[str], str]] = None,
        use_crossfade: bool = True,
        crossfade_ms: int = 50,
    ):
        self.mms_language = mms_language
        self.voice_mapping = {k.lower(): v for k, v in (voice_mapping or {}).items()}
        self.narrator_speed = narrator_speed
        self.dialogue_speed = dialogue_speed
        self.apply_corrections = apply_corrections
        self.use_crossfade = use_crossfade and HAS_CROSSFADE
        self.crossfade_ms = crossfade_ms

        # Fonction de corrections personnalisée ou par défaut
        if corrections_func:
            self._corrections_func = corrections_func
        elif HAS_CORRECTIONS and apply_corrections:
            self._corrections_func = apply_default_corrections
        else:
            self._corrections_func = None

        self._mms_engine = None
        self._kokoro = None

        # Tracking des personnages
        self._characters: Dict[str, Character] = {}
        self._dialogue_history: List[str] = []  # Pour alternance
        self._female_idx = 0
        self._male_idx = 0

        # Compiler le pattern des verbes de parole
        verbs = "|".join(re.escape(v) for v in self.SPEECH_VERBS)
        self._speech_verb_pattern = re.compile(
            rf'\b({verbs})\b',
            re.IGNORECASE
        )

    def _load_mms(self):
        """Charge le moteur MMS."""
        if self._mms_engine is None:
            from .tts_mms_engine import MMSTTSEngine
            self._mms_engine = MMSTTSEngine(
                language=self.mms_language,
                speed=self.narrator_speed
            )
        return self._mms_engine

    def _load_kokoro(self):
        """Charge le moteur Kokoro."""
        if self._kokoro is None:
            from kokoro_onnx import Kokoro
            self._kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
        return self._kokoro

    def _is_valid_name(self, name: str) -> bool:
        """Vérifie si le nom est un vrai nom de personnage."""
        if not name:
            return False

        name_lower = name.lower().strip()

        # Rejeter les mots ignorés
        if name_lower in self.IGNORED_WORDS:
            return False

        # Rejeter les noms trop courts
        if len(name_lower) < 2:
            return False

        # Rejeter les nombres
        if name_lower.isdigit():
            return False

        # Doit commencer par une majuscule dans le texte original
        if not name[0].isupper():
            return False

        return True

    def _guess_gender(self, name: str) -> Optional[str]:
        """Devine le genre à partir du prénom."""
        name_lower = name.lower().split()[0]

        if name_lower in self.FEMALE_NAMES:
            return "F"
        elif name_lower in self.MALE_NAMES:
            return "M"

        # Heuristiques françaises
        if name_lower.endswith(("ette", "elle", "ine", "ienne", "euse", "ice", "ade")):
            return "F"
        elif name_lower.endswith(("eur", "ien", "ois", "ard", "aud")):
            return "M"

        return None

    def _register_character(self, name: str) -> Character:
        """Enregistre ou met à jour un personnage."""
        name_lower = name.lower()

        if name_lower in self._characters:
            self._characters[name_lower].occurrences += 1
            return self._characters[name_lower]

        # Nouveau personnage
        gender = self._guess_gender(name)

        # Assigner une voix
        if name_lower in self.voice_mapping:
            voice = self.voice_mapping[name_lower]
        elif gender == "F":
            voice = self.VOICE_POOL_FEMALE[self._female_idx % len(self.VOICE_POOL_FEMALE)]
            self._female_idx += 1
        elif gender == "M":
            voice = self.VOICE_POOL_MALE[self._male_idx % len(self.VOICE_POOL_MALE)]
            self._male_idx += 1
        else:
            # Alterner
            if self._male_idx <= self._female_idx:
                voice = self.VOICE_POOL_MALE[self._male_idx % len(self.VOICE_POOL_MALE)]
                self._male_idx += 1
            else:
                voice = self.VOICE_POOL_FEMALE[self._female_idx % len(self.VOICE_POOL_FEMALE)]
                self._female_idx += 1

        char = Character(
            name=name,
            gender=gender,
            voice=voice,
            occurrences=1
        )
        self._characters[name_lower] = char
        return char

    def _extract_speaker_after(self, context: str) -> Optional[str]:
        """Extrait le locuteur du contexte APRÈS le dialogue."""
        if not context:
            return None

        # Patterns: "dit Victor", "répondit Marie", "fit-il"
        for verb in self.SPEECH_VERBS:
            # Pattern: verbe + Nom
            pattern = rf'\b{re.escape(verb)}\s+([A-ZÀÂÄÉÈÊËÏÎÔÙÛÜÇ][a-zàâäéèêëïîôùûüç]+)'
            match = re.search(pattern, context)
            if match:
                name = match.group(1)
                if self._is_valid_name(name):
                    return name

            # Pattern: verbe-il/elle (on ne peut pas identifier le personnage mais on sait qu'il y en a un)
            pattern_pronoun = rf'\b{re.escape(verb)}\s*-\s*(?:t-)?(?:il|elle)\b'
            if re.search(pattern_pronoun, context, re.IGNORECASE):
                # On retourne None mais on sait qu'il y a un locuteur
                pass

        return None

    def _extract_speaker_before(self, context: str) -> Optional[str]:
        """Extrait le locuteur du contexte AVANT le dialogue."""
        if not context:
            return None

        # Pattern: "Nom verbe:" ou "Nom :"
        # Ex: "Victor dit :", "Marie :", "Le commissaire demanda :"

        # Chercher un nom suivi d'un verbe de parole
        for verb in self.SPEECH_VERBS:
            pattern = rf'([A-ZÀÂÄÉÈÊËÏÎÔÙÛÜÇ][a-zàâäéèêëïîôùûüç]+)\s+{re.escape(verb)}\s*:'
            match = re.search(pattern, context)
            if match:
                name = match.group(1)
                if self._is_valid_name(name):
                    return name

        # Pattern simple: "Nom :" à la fin du contexte
        pattern = rf'([A-ZÀÂÄÉÈÊËÏÎÔÙÛÜÇ][a-zàâäéèêëïîôùûüç]+)\s*:\s*$'
        match = re.search(pattern, context.strip())
        if match:
            name = match.group(1)
            if self._is_valid_name(name):
                return name

        return None

    def _detect_segments(self, text: str) -> List[DialogueSegment]:
        """
        Détecte les segments de narration et dialogue avec identification des personnages.
        """
        segments = []

        # Pattern pour guillemets français « » et anglais " "
        # On capture: texte avant, guillemet, dialogue, guillemet, texte après
        pattern = re.compile(
            r'(.*?)'           # Texte avant (non-greedy)
            r'([«""])'         # Guillemet ouvrant
            r'([^»""]+)'       # Contenu du dialogue
            r'([»""])'         # Guillemet fermant
            r'([^«""]*?(?=[«""]|$))',  # Texte après jusqu'au prochain dialogue
            re.DOTALL
        )

        last_speaker = None
        last_pos = 0

        for match in pattern.finditer(text):
            context_before = match.group(1).strip()
            dialogue = match.group(3).strip()
            context_after = match.group(5).strip()

            # 1. Ajouter la narration avant le dialogue
            if context_before:
                segments.append(DialogueSegment(
                    type='narration',
                    text=context_before,
                    speaker='NARRATOR'
                ))

            # 2. Identifier le locuteur
            speaker = None

            # D'abord chercher APRÈS le dialogue ("dit Victor")
            speaker = self._extract_speaker_after(context_after)

            # Si pas trouvé, chercher AVANT ("Victor dit:")
            if not speaker:
                speaker = self._extract_speaker_before(context_before)

            # Si toujours pas trouvé, utiliser l'alternance
            if not speaker:
                if last_speaker and len(self._dialogue_history) >= 2:
                    # Alterner entre les 2 derniers locuteurs
                    if self._dialogue_history[-1] == last_speaker:
                        # Chercher l'avant-dernier locuteur différent
                        for prev in reversed(self._dialogue_history[:-1]):
                            if prev != last_speaker:
                                speaker = prev
                                break

                if not speaker:
                    speaker = "INCONNU"

            # Enregistrer le personnage
            if speaker and speaker != "INCONNU":
                char = self._register_character(speaker)
                self._dialogue_history.append(speaker)
                last_speaker = speaker

            # 3. Ajouter le dialogue
            if dialogue:
                segments.append(DialogueSegment(
                    type='dialogue',
                    text=dialogue,
                    speaker=speaker or "INCONNU"
                ))

            # 4. Traiter le contexte après (retirer la partie "dit X")
            if context_after:
                clean_context = self._clean_attribution(context_after)
                if clean_context and len(clean_context) > 10:
                    segments.append(DialogueSegment(
                        type='narration',
                        text=clean_context,
                        speaker='NARRATOR'
                    ))

            last_pos = match.end()

        # Texte restant après le dernier dialogue
        if last_pos < len(text):
            remaining = text[last_pos:].strip()
            if remaining:
                segments.append(DialogueSegment(
                    type='narration',
                    text=remaining,
                    speaker='NARRATOR'
                ))

        # Si aucun dialogue trouvé, tout est narration
        if not segments:
            segments.append(DialogueSegment(
                type='narration',
                text=text.strip(),
                speaker='NARRATOR'
            ))

        return segments

    def _clean_attribution(self, context: str) -> str:
        """Retire les attributions de parole du contexte."""
        result = context

        for verb in self.SPEECH_VERBS:
            # Retirer "verbe Nom"
            result = re.sub(
                rf'\b{re.escape(verb)}\s+[A-ZÀÂÄÉÈÊËÏÎÔÙÛÜÇ][a-zàâäéèêëïîôùûüç]+\s*[.,]?\s*',
                '',
                result,
                flags=re.IGNORECASE
            )
            # Retirer "verbe-t-il/elle"
            result = re.sub(
                rf'\b{re.escape(verb)}\s*-\s*(?:t-)?(?:il|elle)\s*[.,]?\s*',
                '',
                result,
                flags=re.IGNORECASE
            )

        return result.strip()

    def _get_voice_for_speaker(self, speaker: str) -> str:
        """Retourne la voix pour un locuteur."""
        speaker_lower = speaker.lower()

        # Mapping explicite
        if speaker_lower in self.voice_mapping:
            return self.voice_mapping[speaker_lower]

        # Personnage enregistré
        if speaker_lower in self._characters:
            return self._characters[speaker_lower].voice

        # Défaut
        return "af_heart"

    def _convert_to_mp3(
        self,
        wav_path: Path,
        mp3_path: Path,
        bitrate: str = "192k"
    ) -> bool:
        """
        Convertit un fichier WAV en MP3.

        Args:
            wav_path: Chemin du fichier WAV source
            mp3_path: Chemin du fichier MP3 destination
            bitrate: Bitrate du MP3 (64k, 128k, 192k, 256k, 320k)

        Returns:
            True si la conversion a réussi
        """
        if not shutil.which("ffmpeg"):
            print("⚠️ ffmpeg non trouvé, impossible de convertir en MP3")
            return False

        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(wav_path),
                "-codec:a", "libmp3lame",
                "-b:a", bitrate,
                "-q:a", "2",
                str(mp3_path)
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"⚠️ Erreur ffmpeg: {result.stderr}")
                return False
            return True
        except Exception as e:
            print(f"⚠️ Erreur de conversion MP3: {e}")
            return False

    def synthesize_chapter(
        self,
        text: str,
        output_path: Path,
        voice_mapping: Optional[Dict[str, str]] = None,
        output_format: Literal["wav", "mp3"] = "wav",
        mp3_bitrate: str = "192k"
    ) -> bool:
        """
        Synthétise un chapitre complet en mode hybride.

        Args:
            text: Texte du chapitre à synthétiser
            output_path: Chemin du fichier de sortie
            voice_mapping: Mapping personnage -> voix Kokoro
            output_format: Format de sortie ("wav" ou "mp3")
            mp3_bitrate: Bitrate pour MP3 (64k, 128k, 192k, 256k, 320k)

        Returns:
            True si la synthèse a réussi
        """
        import soundfile as sf

        if voice_mapping:
            self.voice_mapping.update({k.lower(): v for k, v in voice_mapping.items()})

        # Réinitialiser le tracking
        self._characters.clear()
        self._dialogue_history.clear()
        self._female_idx = 0
        self._male_idx = 0

        # Appliquer les corrections phonétiques
        if self._corrections_func:
            print("Application des corrections phonétiques...")
            original_len = len(text)
            text = self._corrections_func(text)
            print(f"  {original_len} → {len(text)} caractères")

        # Détecter les segments
        print("Analyse du texte...")
        segments = self._detect_segments(text)

        narr_count = sum(1 for s in segments if s.type == 'narration')
        dial_count = len(segments) - narr_count
        print(f"  {len(segments)} segments ({narr_count} narration, {dial_count} dialogues)")

        # Afficher les personnages détectés
        if self._characters:
            print(f"\n  Personnages détectés:")
            for name, char in sorted(self._characters.items(), key=lambda x: -x[1].occurrences):
                gender_str = f"({char.gender})" if char.gender else "(?)"
                print(f"    {char.name:15} {gender_str:4} → {char.voice} ({char.occurrences}x)")

        # Charger les moteurs
        print("\nChargement des moteurs...")
        mms = self._load_mms()
        kokoro = self._load_kokoro()

        # Synthétiser
        all_audio = []
        sample_rate = 24000
        mms_rate = 16000

        print("\nSynthèse...")
        for i, seg in enumerate(segments):
            text_seg = seg.text.strip()
            if not text_seg:
                continue

            icon = "📖" if seg.type == "narration" else "💬"
            speaker_display = seg.speaker[:12] if seg.speaker else "?"
            print(f"  [{i+1:3}/{len(segments)}] {icon} {speaker_display:12}: {text_seg[:40]}...")

            try:
                if seg.type == 'narration':
                    audio = self._synthesize_mms(text_seg)
                    if audio is not None:
                        audio = self._resample(audio, mms_rate, sample_rate)
                        all_audio.append(audio)
                        all_audio.append(np.zeros(int(0.4 * sample_rate), dtype=np.float32))
                else:
                    voice = self._get_voice_for_speaker(seg.speaker)
                    audio, sr = kokoro.create(text_seg, voice=voice, speed=1.0, lang='fr-fr')
                    all_audio.append(audio)
                    all_audio.append(np.zeros(int(0.3 * sample_rate), dtype=np.float32))

            except Exception as e:
                print(f"    ⚠️ Erreur: {e}")
                continue

        if not all_audio:
            print("Aucun audio généré")
            return False

        # Assembler l'audio avec ou sans crossfade
        if self.use_crossfade and len(all_audio) > 1:
            print("\nApplication du crossfade entre segments...")
            final_audio = apply_crossfade_to_chapter(all_audio, sample_rate, self.crossfade_ms)
        else:
            final_audio = np.concatenate(all_audio)

        max_val = np.max(np.abs(final_audio))
        if max_val > 0:
            final_audio = (final_audio / max_val * 0.9).astype(np.float32)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Déterminer le format à partir de l'extension ou du paramètre
        if output_format == "mp3" or str(output_path).lower().endswith(".mp3"):
            # Générer d'abord en WAV temporaire puis convertir
            wav_path = output_path.with_suffix(".wav")
            sf.write(str(wav_path), final_audio, sample_rate)

            mp3_path = output_path.with_suffix(".mp3")
            print(f"\nConversion en MP3 ({mp3_bitrate})...")
            if self._convert_to_mp3(wav_path, mp3_path, mp3_bitrate):
                # Supprimer le WAV temporaire
                wav_path.unlink()
                final_path = mp3_path
                # Afficher la taille
                size_mb = mp3_path.stat().st_size / (1024 * 1024)
                print(f"   Taille: {size_mb:.1f} MB")
            else:
                print("⚠️ Échec conversion MP3, fichier WAV conservé")
                final_path = wav_path
        else:
            # Sortie WAV directe
            sf.write(str(output_path), final_audio, sample_rate)
            final_path = output_path

        duration = len(final_audio) / sample_rate
        print(f"\n✅ Audio hybride: {final_path}")
        print(f"   Durée: {duration:.1f}s ({duration/60:.1f} min)")

        return True

    def _synthesize_mms(self, text: str) -> Optional[np.ndarray]:
        """Synthétise avec MMS."""
        import tempfile
        import soundfile as sf

        mms = self._load_mms()
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = Path(f.name)

        try:
            success = mms.synthesize(text, temp_path)
            if success and temp_path.exists():
                audio, sr = sf.read(str(temp_path))
                return audio.astype(np.float32)
        finally:
            if temp_path.exists():
                temp_path.unlink()

        return None

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample l'audio."""
        if orig_sr == target_sr:
            return audio

        try:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            ratio = target_sr / orig_sr
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length)
            return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def create_hybrid_engine(
    language: str = "fr",
    voice_mapping: Optional[Dict[str, str]] = None,
    apply_corrections: bool = True,
    corrections_func: Optional[Callable[[str], str]] = None,
    use_crossfade: bool = True,
    crossfade_ms: int = 50,
) -> HybridTTSEngine:
    """
    Crée un moteur hybride configuré.

    Args:
        language: Code langue (fr, en, de, es)
        voice_mapping: Dict personnage -> voix Kokoro
        apply_corrections: Appliquer les corrections phonétiques
        corrections_func: Fonction de corrections personnalisée
        use_crossfade: Utiliser le crossfade entre segments
        crossfade_ms: Durée du crossfade en millisecondes

    Returns:
        Instance HybridTTSEngine configurée
    """
    lang_map = {"fr": "fra", "en": "eng", "de": "deu", "es": "spa"}
    mms_lang = lang_map.get(language, language)

    return HybridTTSEngine(
        mms_language=mms_lang,
        voice_mapping=voice_mapping or {},
        apply_corrections=apply_corrections,
        corrections_func=corrections_func,
        use_crossfade=use_crossfade,
        crossfade_ms=crossfade_ms,
    )


if __name__ == "__main__":
    test_text = """
    Victor entra dans la pièce.

    « Où est Marie ? » demanda Victor.

    « Elle est partie » répondit Kamel. « Tu l'as manquée de peu. »

    Victor soupira. Crawford s'approcha.

    « On a un problème » dit Crawford.
    """

    engine = create_hybrid_engine("fr")
    engine.synthesize_chapter(test_text, Path("output/test_hybrid_v2.wav"))
