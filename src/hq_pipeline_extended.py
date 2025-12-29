"""
Pipeline haute qualite etendu avec toutes les fonctionnalites avancees.

Integration complete de:
- Pipeline HQ de base
- Audio Tags (style ElevenLabs v3)
- Voice Morphing (pitch, formant, time stretch)
- Clonage de voix (XTTS-v2)
- Cache intelligent et parallelisation
- Controle d'emotion et phonemes
- Generation de conversations multi-speakers
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any
import numpy as np
import json
import time

# Pipeline de base
from .hq_pipeline import (
    HQPipeline,
    HQPipelineConfig,
    HQSegment,
    create_hq_pipeline
)

# Nouveaux modules
from .audio_tags import (
    AudioTagProcessor,
    AudioTag,
    process_text_with_audio_tags,
    ProcessedSegment
)
from .voice_morphing import (
    VoiceMorpher,
    VoiceMorphSettings,
    VoicePresets
)
from .voice_cloning import (
    VoiceCloner,
    VoiceCloningManager,
    ClonedVoice
)
from .synthesis_cache import (
    SynthesisCache,
    ParallelSynthesizer,
    BatchProcessor,
    CacheStats
)
from .emotion_control import (
    EmotionController,
    EmotionSettings,
    PhonemeProcessor,
    PronunciationManager,
    create_pronunciation_config
)
from .conversation_generator import (
    ConversationGenerator,
    DialogueParser,
    VoicePool,
    Conversation,
    Speaker
)


@dataclass
class ExtendedPipelineConfig(HQPipelineConfig):
    """Configuration etendue avec toutes les nouvelles fonctionnalites."""

    # Audio Tags
    enable_audio_tags: bool = True
    preserve_tts_native_tags: bool = True

    # Voice Morphing
    enable_voice_morphing: bool = False
    default_morph_preset: Optional[str] = None
    character_morph_presets: Dict[str, str] = field(default_factory=dict)

    # Voice Cloning
    enable_voice_cloning: bool = False
    cloned_voices_dir: Path = field(default_factory=lambda: Path(".voice_cache"))

    # Cache
    enable_cache: bool = True
    cache_dir: Path = field(default_factory=lambda: Path(".synthesis_cache"))
    max_cache_size_mb: float = 1000.0

    # Parallelisation
    enable_parallel: bool = True
    num_workers: int = 4

    # Emotion Control
    emotion_intensity: float = 0.5
    emotion_stability: float = 0.7
    style_exaggeration: float = 0.5

    # Phonemes
    enable_phoneme_processing: bool = True
    custom_phonemes: Dict[str, str] = field(default_factory=dict)

    def save(self, path: Path):
        """Sauvegarde la configuration etendue."""
        data = {
            # Config de base
            "lang": self.lang,
            "narrator_voice": self.narrator_voice,
            "voice_mapping": self.voice_mapping,
            "auto_assign_voices": self.auto_assign_voices,
            "enable_emotion_analysis": self.enable_emotion_analysis,
            "enable_narrative_context": self.enable_narrative_context,
            "enable_audio_enhancement": self.enable_audio_enhancement,
            # Config etendue
            "enable_audio_tags": self.enable_audio_tags,
            "enable_voice_morphing": self.enable_voice_morphing,
            "default_morph_preset": self.default_morph_preset,
            "character_morph_presets": self.character_morph_presets,
            "enable_voice_cloning": self.enable_voice_cloning,
            "enable_cache": self.enable_cache,
            "enable_parallel": self.enable_parallel,
            "num_workers": self.num_workers,
            "emotion_intensity": self.emotion_intensity,
            "emotion_stability": self.emotion_stability,
            "style_exaggeration": self.style_exaggeration,
            "enable_phoneme_processing": self.enable_phoneme_processing,
            "custom_phonemes": self.custom_phonemes,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


@dataclass
class ExtendedHQSegment(HQSegment):
    """Segment HQ etendu avec nouvelles metadonnees."""

    # Audio Tags
    audio_tags: List[AudioTag] = field(default_factory=list)
    audio_tag_prosody: Dict[str, float] = field(default_factory=dict)

    # Voice Morphing
    morph_settings: Optional[VoiceMorphSettings] = None

    # Clonage
    is_cloned_voice: bool = False
    cloned_voice_name: Optional[str] = None

    # Cache
    cache_key: Optional[str] = None
    from_cache: bool = False

    # Emotion control
    emotion_exaggeration: float = 0.5


class ExtendedHQPipeline:
    """
    Pipeline haute qualite etendu.

    Combine le pipeline HQ de base avec:
    - Traitement des audio tags
    - Voice morphing
    - Clonage de voix
    - Cache et parallelisation
    - Controle d'emotion avance
    """

    def __init__(self, config: Optional[ExtendedPipelineConfig] = None):
        self.config = config or ExtendedPipelineConfig()

        # Pipeline de base
        self.base_pipeline = HQPipeline(self.config)

        # Initialiser les composants etendus
        self._init_extended_components()

        # Stats
        self._stats = {
            "segments_processed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_time_saved": 0.0,
            "morphed_segments": 0,
            "cloned_voice_segments": 0,
        }

    def _init_extended_components(self):
        """Initialise les composants etendus."""
        # Audio Tags
        self.tag_processor = AudioTagProcessor()

        # Voice Morphing
        self.voice_morpher = VoiceMorpher()

        # Voice Cloning
        if self.config.enable_voice_cloning:
            self.cloning_manager = VoiceCloningManager(
                self.config.cloned_voices_dir
            )
        else:
            self.cloning_manager = None

        # Cache
        if self.config.enable_cache:
            self.cache = SynthesisCache(
                cache_dir=self.config.cache_dir,
                max_size_mb=self.config.max_cache_size_mb
            )
        else:
            self.cache = None

        # Parallelisation
        if self.config.enable_parallel and self.cache:
            self.parallel_synth = ParallelSynthesizer(
                num_workers=self.config.num_workers,
                cache=self.cache
            )
        else:
            self.parallel_synth = None

        # Emotion Control
        self.emotion_controller = EmotionController(EmotionSettings(
            intensity=self.config.emotion_intensity,
            stability=self.config.emotion_stability,
            style_exaggeration=self.config.style_exaggeration
        ))

        # Phonemes
        if self.config.enable_phoneme_processing:
            self.pronunciation_manager = create_pronunciation_config(
                self.config.lang
            )
            # Ajouter les phonemes personnalises
            for word, ipa in self.config.custom_phonemes.items():
                self.pronunciation_manager.add_phoneme(word, ipa)
        else:
            self.pronunciation_manager = None

    def _process_audio_tags(self, text: str) -> ProcessedSegment:
        """Traite les audio tags dans le texte."""
        return process_text_with_audio_tags(text)

    def _get_morph_settings(self, speaker: str) -> Optional[VoiceMorphSettings]:
        """Recupere les parametres de morphing pour un speaker."""
        if not self.config.enable_voice_morphing:
            return None

        # Preset specifique au personnage
        preset_name = self.config.character_morph_presets.get(speaker)

        # Ou preset par defaut
        if not preset_name:
            preset_name = self.config.default_morph_preset

        if preset_name:
            return VoicePresets.get(preset_name)

        return None

    def _apply_pronunciation(self, text: str) -> str:
        """Applique les corrections de prononciation."""
        if self.pronunciation_manager:
            return self.pronunciation_manager.process(text)
        return text

    def _calculate_emotion_prosody(
        self,
        base_prosody: dict,
        emotion_type: Optional[str] = None
    ) -> dict:
        """Calcule la prosodie avec controle d'emotion."""
        return self.emotion_controller.calculate_prosody(
            base_speed=base_prosody.get("speed", 1.0),
            base_pitch=base_prosody.get("pitch", 0.0),
            base_volume=base_prosody.get("volume", 1.0),
            emotion_type=emotion_type
        )

    def process_chapter(
        self,
        text: str,
        chapter_index: int = 0,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[ExtendedHQSegment]:
        """
        Traite un chapitre avec toutes les fonctionnalites etendues.

        Args:
            text: Texte du chapitre
            chapter_index: Index du chapitre
            progress_callback: Callback de progression

        Returns:
            Liste de segments HQ etendus
        """
        total_steps = 8
        step = 0

        # Etape 1: Traitement des audio tags
        if progress_callback:
            progress_callback(step, total_steps, "Traitement des audio tags")
        step += 1

        if self.config.enable_audio_tags:
            tag_result = self._process_audio_tags(text)
            text_without_tags = tag_result.text
            global_tags = tag_result.tags
        else:
            text_without_tags = text
            global_tags = []

        # Etape 2: Corrections de prononciation
        if progress_callback:
            progress_callback(step, total_steps, "Corrections de prononciation")
        step += 1

        if self.config.enable_phoneme_processing:
            text_corrected = self._apply_pronunciation(text_without_tags)
        else:
            text_corrected = text_without_tags

        # Etape 3-7: Pipeline de base
        if progress_callback:
            progress_callback(step, total_steps, "Analyse du texte")

        base_segments = self.base_pipeline.process_chapter(
            text_corrected,
            chapter_index,
            progress_callback=lambda s, t, m: progress_callback(
                step + s, total_steps + t - 1, m
            ) if progress_callback else None
        )

        step = 6

        # Etape 8: Extension des segments
        if progress_callback:
            progress_callback(step, total_steps, "Extension des segments")
        step += 1

        extended_segments = []

        for base_seg in base_segments:
            # Traiter les audio tags du segment
            if self.config.enable_audio_tags:
                seg_tag_result = self._process_audio_tags(base_seg.text)
                clean_text = seg_tag_result.text
                seg_tags = seg_tag_result.tags
                tag_prosody = seg_tag_result.prosody
            else:
                clean_text = base_seg.text
                seg_tags = []
                tag_prosody = {}

            # Combiner avec les tags globaux
            all_tags = global_tags + seg_tags

            # Calculer la prosodie emotionnelle
            emotion_prosody = self._calculate_emotion_prosody(
                {
                    "speed": base_seg.final_speed,
                    "pitch": base_seg.prosody.pitch if base_seg.prosody else 0.0,
                    "volume": 1.0
                },
                base_seg.emotion.value if base_seg.emotion else None
            )

            # Recuperer les parametres de morphing
            morph_settings = self._get_morph_settings(base_seg.speaker)

            # Verifier si c'est une voix clonee
            is_cloned = False
            cloned_name = None
            if self.cloning_manager and base_seg.voice_id.startswith("cloned_"):
                is_cloned = True
                cloned_name = base_seg.voice_id

            # Creer le segment etendu
            extended_seg = ExtendedHQSegment(
                # Attributs de base
                text=clean_text,
                index=base_seg.index,
                speaker=base_seg.speaker,
                speaker_type=base_seg.speaker_type,
                voice_id=base_seg.voice_id,
                emotion=base_seg.emotion,
                intensity=base_seg.intensity,
                prosody=base_seg.prosody,
                narrative_type=base_seg.narrative_type,
                narrative_confidence=base_seg.narrative_confidence,
                tts_tags=base_seg.tts_tags,
                pause_before=base_seg.pause_before,
                pause_after=base_seg.pause_after,
                is_dialogue=base_seg.is_dialogue,
                is_internal_thought=base_seg.is_internal_thought,
                chapter_index=base_seg.chapter_index,
                final_speed=emotion_prosody["speed"],

                # Attributs etendus
                audio_tags=all_tags,
                audio_tag_prosody=tag_prosody,
                morph_settings=morph_settings,
                is_cloned_voice=is_cloned,
                cloned_voice_name=cloned_name,
                emotion_exaggeration=self.config.style_exaggeration,
            )

            extended_segments.append(extended_seg)
            self._stats["segments_processed"] += 1

            if morph_settings:
                self._stats["morphed_segments"] += 1
            if is_cloned:
                self._stats["cloned_voice_segments"] += 1

        # Finalisation
        if progress_callback:
            progress_callback(total_steps, total_steps, "Termine")

        return extended_segments

    def synthesize_segments(
        self,
        segments: List[ExtendedHQSegment],
        synthesize_fn: Callable[[str, str, float], np.ndarray],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[np.ndarray]:
        """
        Synthetise une liste de segments.

        Utilise le cache et la parallelisation si actives.

        Args:
            segments: Segments a synthetiser
            synthesize_fn: Fonction de synthese (text, voice, speed) -> audio
            progress_callback: Callback (done, total)

        Returns:
            Liste des audios generes
        """
        if self.parallel_synth and self.config.enable_parallel:
            # Preparer les segments pour la parallelisation
            batch = [
                {
                    "text": seg.text,
                    "voice_id": seg.voice_id,
                    "speed": seg.final_speed,
                }
                for seg in segments
            ]

            # Synthese parallele avec cache
            audios = self.parallel_synth.synthesize_batch(
                batch,
                synthesize_fn=lambda s: synthesize_fn(
                    s["text"], s["voice_id"], s["speed"]
                ),
                progress_callback=progress_callback
            )

        else:
            # Synthese sequentielle
            audios = []
            for i, seg in enumerate(segments):
                # Verifier le cache
                cached = None
                if self.cache:
                    cached = self.cache.get(seg.text, seg.voice_id, seg.final_speed)

                if cached:
                    import soundfile as sf
                    audio, _ = sf.read(str(cached))
                    self._stats["cache_hits"] += 1
                else:
                    audio = synthesize_fn(seg.text, seg.voice_id, seg.final_speed)
                    self._stats["cache_misses"] += 1

                    # Mettre en cache
                    if self.cache:
                        self.cache.put(
                            seg.text, seg.voice_id, seg.final_speed, audio
                        )

                audios.append(audio)

                if progress_callback:
                    progress_callback(i + 1, len(segments))

        # Appliquer le morphing si active
        if self.config.enable_voice_morphing:
            audios = self._apply_morphing(segments, audios)

        return audios

    def _apply_morphing(
        self,
        segments: List[ExtendedHQSegment],
        audios: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Applique le voice morphing aux audios."""
        morphed = []

        for seg, audio in zip(segments, audios):
            if seg.morph_settings:
                audio = self.voice_morpher.morph(audio, seg.morph_settings)
                self._stats["morphed_segments"] += 1

            morphed.append(audio)

        return morphed

    def get_stats(self) -> dict:
        """Retourne les statistiques du pipeline."""
        stats = self._stats.copy()

        if self.cache:
            cache_stats = self.cache.get_stats()
            stats["cache_entries"] = cache_stats.total_entries
            stats["cache_size_mb"] = cache_stats.total_size_mb
            stats["cache_time_saved"] = cache_stats.time_saved_seconds

        return stats

    def get_characters(self) -> list:
        """Retourne les personnages detectes."""
        return self.base_pipeline.get_characters()

    def get_voice_assignments(self) -> dict:
        """Retourne les assignations de voix."""
        return self.base_pipeline.get_voice_assignments()

    def reset(self):
        """Reinitialise le pipeline."""
        self.base_pipeline.reset()
        self._stats = {
            "segments_processed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_time_saved": 0.0,
            "morphed_segments": 0,
            "cloned_voice_segments": 0,
        }


def create_extended_pipeline(
    lang: str = "fr",
    narrator_voice: str = "ff_siwis",
    enable_all_features: bool = False,
    **kwargs
) -> ExtendedHQPipeline:
    """
    Cree un pipeline etendu.

    Args:
        lang: Langue
        narrator_voice: Voix du narrateur
        enable_all_features: Activer toutes les fonctionnalites
        **kwargs: Options supplementaires

    Returns:
        Pipeline etendu configure
    """
    if enable_all_features:
        kwargs.setdefault("enable_audio_tags", True)
        kwargs.setdefault("enable_voice_morphing", True)
        kwargs.setdefault("enable_cache", True)
        kwargs.setdefault("enable_parallel", True)
        kwargs.setdefault("enable_phoneme_processing", True)

    config = ExtendedPipelineConfig(
        lang=lang,
        narrator_voice=narrator_voice,
        **kwargs
    )

    return ExtendedHQPipeline(config)


class AudiobookGenerator:
    """
    Generateur d'audiobooks complet utilisant le pipeline etendu.

    Produit des audiobooks de qualite professionnelle.
    """

    def __init__(
        self,
        config: Optional[ExtendedPipelineConfig] = None,
        tts_engine=None
    ):
        self.pipeline = ExtendedHQPipeline(config)
        self.tts_engine = tts_engine

    def generate_audiobook(
        self,
        chapters: List[str],
        output_dir: Path,
        title: str = "audiobook",
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> dict:
        """
        Genere un audiobook complet.

        Args:
            chapters: Liste des chapitres (texte)
            output_dir: Dossier de sortie
            title: Titre de l'audiobook
            progress_callback: Callback (phase, current, total)

        Returns:
            Dictionnaire avec les informations de l'audiobook
        """
        import soundfile as sf

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "title": title,
            "chapters": [],
            "total_duration": 0.0,
            "total_segments": 0,
        }

        for i, chapter_text in enumerate(chapters):
            chapter_name = f"chapter_{i+1:03d}"

            if progress_callback:
                progress_callback("chapter", i + 1, len(chapters))

            # Traiter le chapitre
            segments = self.pipeline.process_chapter(chapter_text, i)

            # Synthetiser
            if self.tts_engine:
                def synth_fn(text, voice, speed):
                    return self.tts_engine.synthesize(text, voice, speed)

                audios = self.pipeline.synthesize_segments(
                    segments,
                    synth_fn,
                    progress_callback=lambda d, t: progress_callback(
                        "synthesis", d, t
                    ) if progress_callback else None
                )

                # Concatener
                full_audio = self._concatenate_with_pauses(segments, audios)

                # Sauvegarder
                chapter_path = output_dir / f"{chapter_name}.wav"
                sf.write(str(chapter_path), full_audio, 24000)

                duration = len(full_audio) / 24000
            else:
                chapter_path = None
                duration = 0

            results["chapters"].append({
                "index": i,
                "name": chapter_name,
                "segments": len(segments),
                "duration": duration,
                "path": str(chapter_path) if chapter_path else None,
            })

            results["total_duration"] += duration
            results["total_segments"] += len(segments)

        # Sauvegarder les metadonnees
        meta_path = output_dir / "metadata.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Ajouter les stats
        results["pipeline_stats"] = self.pipeline.get_stats()
        results["voice_assignments"] = self.pipeline.get_voice_assignments()

        return results

    def _concatenate_with_pauses(
        self,
        segments: List[ExtendedHQSegment],
        audios: List[np.ndarray],
        sample_rate: int = 24000
    ) -> np.ndarray:
        """Concatene les audios avec les pauses appropriees."""
        result_parts = []

        for seg, audio in zip(segments, audios):
            # Pause avant
            if seg.pause_before > 0:
                pause_samples = int(seg.pause_before * sample_rate)
                result_parts.append(np.zeros(pause_samples, dtype=np.float32))

            # Audio
            result_parts.append(audio)

            # Pause apres
            if seg.pause_after > 0:
                pause_samples = int(seg.pause_after * sample_rate)
                result_parts.append(np.zeros(pause_samples, dtype=np.float32))

        return np.concatenate(result_parts)


if __name__ == "__main__":
    print("=== Test Pipeline Etendu ===\n")

    # Creer un pipeline avec toutes les fonctionnalites
    pipeline = create_extended_pipeline(
        lang="fr",
        narrator_voice="ff_siwis",
        enable_audio_tags=True,
        enable_cache=True,
        enable_phoneme_processing=True,
        custom_phonemes={
            "API": "a pe i",
            "Python": "pitonne",
        }
    )

    # Texte de test avec audio tags
    test_text = """
    Chapitre 1 : La Revelation

    [dramatic] Le silence etait pesant. Marie attendait, le coeur battant.

    « [whispers] J'ai quelque chose a te dire... » murmura Pierre.

    [excited] Soudain, il s'exclama : « J'ai reussi ! L'API Python fonctionne ! »

    [pause] Marie le regarda, incrédule.

    « [surprised] Vraiment ? C'est incroyable ! »

    [laugh] Ils eclaterent de rire tous les deux.
    """

    def progress(step, total, message):
        print(f"  [{step}/{total}] {message}")

    segments = pipeline.process_chapter(test_text, progress_callback=progress)

    print(f"\n=== Resultats ===")
    print(f"Segments: {len(segments)}")

    print(f"\n=== Segments avec Audio Tags ===")
    for seg in segments[:5]:
        tags_str = ", ".join([t.name for t in seg.audio_tags]) if seg.audio_tags else "-"
        print(f"[{seg.speaker:10}] tags=[{tags_str:20}] | {seg.text[:40]}...")

    print(f"\n=== Stats ===")
    for key, value in pipeline.get_stats().items():
        print(f"  {key}: {value}")
