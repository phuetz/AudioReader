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
- v2.3: Respirations avancees, contours d'intonation, timing humanise
- v2.4: Styles de narration, controle mot-par-mot, attribution dialogue,
        conformite ACX/Audible, detection emotion LLM
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

from .bio_acoustics import BioAudioGenerator
from .dynamic_voice import DynamicVoiceManager

# v2.3 modules
from .breath_samples import HybridBreathGenerator
from .intonation_contour import (
    IntonationProcessor,
    IntonationContour,
    IntonationContourDetector
)
from .timing_humanizer import (
    TimingHumanizer,
    TimingConfig,
    PauseCalculator,
    TextTimingProcessor,
    ClauseType
)

# v2.4 modules
from .narration_styles import (
    NarrationStyle,
    NarrationStyleManager,
    StyleProsodyProfile,
    STYLE_PROFILES
)
from .word_level_control import (
    WordLevelController,
    WordProsody,
    ProcessedText as WordProcessedText,
    process_text_with_word_control
)
from .dialogue_attribution import (
    DialogueAttributor,
    DialogueAttribution,
    AttributedDialogue,
    AttributionMethod
)
from .acx_compliance import (
    ACXAnalyzer,
    ACXCorrector,
    ACXStandards,
    ComplianceReport,
    ComplianceLevel
)
from .llm_emotion_detector import (
    LLMEmotionDetector,
    LLMConfig,
    EmotionResult,
    EmotionCategory
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

    # v2.3: Advanced breath generation
    enable_advanced_breaths: bool = True
    breath_samples_dir: Optional[Path] = None  # None = synthese uniquement
    prefer_breath_samples: bool = True

    # v2.3: Intonation contours
    enable_intonation_contours: bool = True
    intonation_strength: float = 0.7  # 0.0-1.0

    # v2.3: Timing humanization
    enable_timing_humanization: bool = True
    pause_variation_sigma: float = 0.05  # 5% standard deviation
    enable_emphasis_pauses: bool = True
    emphasis_pause_duration: float = 0.05  # 50ms

    # v2.4: Narration styles
    enable_narration_styles: bool = True
    default_narration_style: str = "storytelling"  # formal, conversational, dramatic, storytelling

    # v2.4: Word-level control
    enable_word_level_control: bool = True
    auto_emphasis: bool = True
    emphasis_strength: float = 1.0

    # v2.4: Dialogue attribution
    enable_dialogue_attribution: bool = True
    known_characters: Dict[str, str] = field(default_factory=dict)  # {name: gender}

    # v2.4: ACX/Audible compliance
    enable_acx_compliance: bool = True
    acx_target_lufs: float = -20.0
    acx_peak_limit: float = -3.0

    # v2.4: LLM emotion detection
    enable_llm_emotion: bool = False  # Requires Ollama or OpenAI
    llm_provider: str = "ollama"
    llm_model: str = "llama3.2"

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
            # v2.3
            "enable_advanced_breaths": self.enable_advanced_breaths,
            "breath_samples_dir": str(self.breath_samples_dir) if self.breath_samples_dir else None,
            "enable_intonation_contours": self.enable_intonation_contours,
            "intonation_strength": self.intonation_strength,
            "enable_timing_humanization": self.enable_timing_humanization,
            "pause_variation_sigma": self.pause_variation_sigma,
            "enable_emphasis_pauses": self.enable_emphasis_pauses,
            "emphasis_pause_duration": self.emphasis_pause_duration,
            # v2.4
            "enable_narration_styles": self.enable_narration_styles,
            "default_narration_style": self.default_narration_style,
            "enable_word_level_control": self.enable_word_level_control,
            "auto_emphasis": self.auto_emphasis,
            "emphasis_strength": self.emphasis_strength,
            "enable_dialogue_attribution": self.enable_dialogue_attribution,
            "known_characters": self.known_characters,
            "enable_acx_compliance": self.enable_acx_compliance,
            "acx_target_lufs": self.acx_target_lufs,
            "acx_peak_limit": self.acx_peak_limit,
            "enable_llm_emotion": self.enable_llm_emotion,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
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

    # v2.3: Intonation and timing
    intonation_contour: Optional[IntonationContour] = None
    humanized_pause_before: float = 0.0
    humanized_pause_after: float = 0.0
    clause_type: Optional[ClauseType] = None
    has_emphasis_pause: bool = False

    # v2.4: Narration style
    narration_style: Optional[NarrationStyle] = None
    style_prosody: Dict[str, float] = field(default_factory=dict)

    # v2.4: Word-level control
    word_prosodies: List[WordProsody] = field(default_factory=list)
    has_word_emphasis: bool = False

    # v2.4: Dialogue attribution
    attributed_speaker: Optional[str] = None
    attribution_method: Optional[AttributionMethod] = None
    attribution_confidence: float = 0.0

    # v2.4: LLM emotion
    llm_emotion: Optional[EmotionCategory] = None
    llm_emotion_confidence: float = 0.0
    detected_subtext: Optional[str] = None


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

        # Dynamic Voice (Blending)
        self.dynamic_voice_manager = DynamicVoiceManager()

        # v2.3: Advanced breath generation
        if self.config.enable_advanced_breaths:
            self.breath_generator = HybridBreathGenerator(
                sample_rate=24000,
                samples_dir=self.config.breath_samples_dir,
                prefer_samples=self.config.prefer_breath_samples
            )
        else:
            self.breath_generator = BioAudioGenerator(
                sample_rate=24000,
                use_advanced_breaths=False
            )

        # v2.3: Intonation contours
        if self.config.enable_intonation_contours:
            self.intonation_processor = IntonationProcessor(
                sample_rate=24000,
                language=self.config.lang,
                strength=self.config.intonation_strength,
                enabled=True
            )
            self.intonation_detector = IntonationContourDetector(
                language=self.config.lang
            )
        else:
            self.intonation_processor = None
            self.intonation_detector = None

        # v2.3: Timing humanization
        if self.config.enable_timing_humanization:
            timing_config = TimingConfig(
                pause_variation_sigma=self.config.pause_variation_sigma,
                enable_emphasis_pauses=self.config.enable_emphasis_pauses,
                emphasis_pause_duration=self.config.emphasis_pause_duration
            )
            self.timing_humanizer = TimingHumanizer(
                config=timing_config,
                language=self.config.lang
            )
            self.pause_calculator = PauseCalculator(
                humanizer=self.timing_humanizer
            )
        else:
            self.timing_humanizer = None
            self.pause_calculator = None

        # v2.4: Narration styles
        if self.config.enable_narration_styles:
            try:
                default_style = NarrationStyle(self.config.default_narration_style)
            except ValueError:
                default_style = NarrationStyle.STORYTELLING
            self.narration_style_manager = NarrationStyleManager(default_style)
        else:
            self.narration_style_manager = None

        # v2.4: Word-level control
        if self.config.enable_word_level_control:
            self.word_controller = WordLevelController(
                auto_emphasis=self.config.auto_emphasis,
                emphasis_strength=self.config.emphasis_strength,
                lang=self.config.lang
            )
        else:
            self.word_controller = None

        # v2.4: Dialogue attribution
        if self.config.enable_dialogue_attribution:
            self.dialogue_attributor = DialogueAttributor(lang=self.config.lang)
            # Register known characters
            for name, gender in self.config.known_characters.items():
                self.dialogue_attributor.register_character(name, gender)
        else:
            self.dialogue_attributor = None

        # v2.4: ACX compliance
        if self.config.enable_acx_compliance:
            standards = ACXStandards(
                lufs_target=self.config.acx_target_lufs,
                peak_max_db=self.config.acx_peak_limit
            )
            self.acx_analyzer = ACXAnalyzer(standards)
            self.acx_corrector = ACXCorrector(standards)
        else:
            self.acx_analyzer = None
            self.acx_corrector = None

        # v2.4: LLM emotion detection
        if self.config.enable_llm_emotion:
            llm_config = LLMConfig(
                provider=self.config.llm_provider,
                model=self.config.llm_model
            )
            self.llm_emotion_detector = LLMEmotionDetector(llm_config)
        else:
            self.llm_emotion_detector = None

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

            # --- DYNAMIC VOICE BLENDING ---
            # Uniquement si on utilise Kokoro (détecté par le format de la voix ou config)
            # On assume que si la voix ne contient pas "Neural", c'est du Kokoro
            final_voice_id = base_seg.voice_id
            if "Neural" not in final_voice_id: # Simple heuristic
                final_voice_id = self.dynamic_voice_manager.get_voice_config(
                    base_voice=base_seg.voice_id,
                    emotion=base_seg.emotion,
                    intensity=base_seg.intensity
                )
            
            # ---

            # Recuperer les parametres de morphing
            morph_settings = self._get_morph_settings(base_seg.speaker)

            # Verifier si c'est une voix clonee
            is_cloned = False
            cloned_name = None
            if self.cloning_manager and final_voice_id.startswith("cloned_"):
                is_cloned = True
                cloned_name = final_voice_id

            # --- v2.3: Intonation contour detection ---
            detected_contour = None
            if self.intonation_detector:
                detected_contour = self.intonation_detector.detect(clean_text)

            # --- v2.3: Timing humanization ---
            humanized_pause_before = base_seg.pause_before
            humanized_pause_after = base_seg.pause_after
            detected_clause_type = None
            has_emphasis = False

            if self.timing_humanizer:
                # Humanize pauses
                humanized_pause_before = self.timing_humanizer.humanize_pause(
                    base_seg.pause_before
                )
                humanized_pause_after = self.timing_humanizer.humanize_pause(
                    base_seg.pause_after
                )

                # Detect clause type for speed adjustment
                detected_clause_type = self.timing_humanizer.get_clause_type(clean_text)

                # Check for emphasis pauses
                if self.config.enable_emphasis_pauses:
                    processed_text = self.timing_humanizer.add_emphasis_pauses(clean_text)
                    if processed_text != clean_text:
                        has_emphasis = True
                        clean_text = processed_text

            # Creer le segment etendu
            extended_seg = ExtendedHQSegment(
                # Attributs de base
                text=clean_text,
                index=base_seg.index,
                speaker=base_seg.speaker,
                speaker_type=base_seg.speaker_type,
                voice_id=final_voice_id, # Voix modifiée ici
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

                # v2.3 attributes
                intonation_contour=detected_contour,
                humanized_pause_before=humanized_pause_before,
                humanized_pause_after=humanized_pause_after,
                clause_type=detected_clause_type,
                has_emphasis_pause=has_emphasis,
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

            # Fonction de synthese qui extrait l'audio si c'est un tuple
            def wrapped_synth_fn(s):
                res = synthesize_fn(s["text"], s["voice_id"], s["speed"])
                return res[0] if isinstance(res, tuple) else res

            # Synthese parallele avec cache
            audios = self.parallel_synth.synthesize_batch(
                batch,
                synthesize_fn=wrapped_synth_fn,
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
                    audio_full = synthesize_fn(seg.text, seg.voice_id, seg.final_speed)
                    audio = audio_full[0] if isinstance(audio_full, tuple) else audio_full
                    
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

        # v2.3: Appliquer les contours d'intonation
        if self.config.enable_intonation_contours and self.intonation_processor:
            audios = self._apply_intonation_contours(segments, audios)

        return audios

    def _apply_intonation_contours(
        self,
        segments: List[ExtendedHQSegment],
        audios: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Applique les contours d'intonation aux audios."""
        processed = []

        for seg, audio in zip(segments, audios):
            if seg.intonation_contour and seg.intonation_contour != IntonationContour.NEUTRAL:
                try:
                    audio = self.intonation_processor.applicator.apply_contour(
                        audio,
                        seg.intonation_contour,
                        strength=self.config.intonation_strength
                    )
                except Exception as e:
                    # En cas d'erreur, garder l'audio original
                    import logging
                    logging.warning(f"Erreur application contour: {e}")

            processed.append(audio)

        return processed

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
        # v2.3 features
        kwargs.setdefault("enable_advanced_breaths", True)
        kwargs.setdefault("enable_intonation_contours", True)
        kwargs.setdefault("enable_timing_humanization", True)
        kwargs.setdefault("enable_emphasis_pauses", True)
        # v2.4 features
        kwargs.setdefault("enable_narration_styles", True)
        kwargs.setdefault("enable_word_level_control", True)
        kwargs.setdefault("enable_dialogue_attribution", True)
        kwargs.setdefault("enable_acx_compliance", True)
        # Note: LLM emotion requires explicit opt-in due to external dependency
        # kwargs.setdefault("enable_llm_emotion", True)

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
        # v2.3: Use the pipeline's breath generator (hybrid or basic)
        self.breath_generator = self.pipeline.breath_generator

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
                from .tts_unified import TTSEngine
                
                # Mapper le moteur
                engine_map = {
                    "auto": TTSEngine.AUTO,
                    "kokoro": TTSEngine.KOKORO,
                    "edge-tts": TTSEngine.EDGE_TTS
                }
                selected_engine = engine_map.get(self.pipeline.config.tts_engine, TTSEngine.AUTO)

                def synth_fn(text, voice, speed):
                    return self.tts_engine.synthesize(
                        text, 
                        voice=voice, 
                        speed=speed, 
                        lang=self.pipeline.config.lang,
                        engine=selected_engine
                    )

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
        """
        Concatene les audios avec pauses bio-acoustiques.

        v2.3: Utilise les pauses humanisees et le generateur hybride.
        """
        result_parts = []

        # Helper pour generer silence/room tone
        def generate_silence(duration: float) -> np.ndarray:
            if hasattr(self.breath_generator, 'synth_generator'):
                # HybridBreathGenerator
                return self.breath_generator.synth_generator.generate_silence(duration)
            else:
                # BioAudioGenerator direct
                return self.breath_generator.generate_silence(duration)

        # Helper pour generer respiration
        def generate_breath(breath_type: str = "soft", duration: float = 0.3) -> np.ndarray:
            if hasattr(self.breath_generator, 'generate_breath'):
                return self.breath_generator.generate_breath(breath_type, duration)
            else:
                return self.breath_generator.generate_breath(type=breath_type, duration=duration)

        # Helper pour generer bruit de bouche
        def generate_mouth_noise() -> np.ndarray:
            if hasattr(self.breath_generator, 'synth_generator'):
                return self.breath_generator.synth_generator.generate_mouth_noise()
            else:
                return self.breath_generator.generate_mouth_noise()

        # Ajouter room tone initial
        result_parts.append(generate_silence(0.5))

        for seg, audio in zip(segments, audios):
            # v2.3: Utiliser les pauses humanisees si disponibles
            pause_before = seg.humanized_pause_before if seg.humanized_pause_before > 0 else seg.pause_before
            pause_after = seg.humanized_pause_after if seg.humanized_pause_after > 0 else seg.pause_after

            # 1. Gestion des respirations avant (definies par emotion_analyzer)
            if hasattr(seg.prosody, 'breath_before') and seg.prosody and seg.prosody.breath_before:
                breath = generate_breath("soft", 0.3)
                result_parts.append(breath)
                # Petit silence apres respiration
                result_parts.append(generate_silence(0.1))

            # 2. Pause avant explicite (Room Tone au lieu de 0)
            if pause_before > 0:
                # Si pause longue, inserer un bruit de bouche aleatoire parfois
                if pause_before > 1.0 and np.random.random() < 0.2:
                    result_parts.append(generate_silence(pause_before * 0.8))
                    result_parts.append(generate_mouth_noise())
                    result_parts.append(generate_silence(pause_before * 0.2))
                else:
                    result_parts.append(generate_silence(pause_before))

            # 3. Audio du segment
            result_parts.append(audio)

            # 4. Pause apres (Room Tone)
            if pause_after > 0:
                result_parts.append(generate_silence(pause_after))

        # Room tone final
        result_parts.append(generate_silence(1.0))

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
