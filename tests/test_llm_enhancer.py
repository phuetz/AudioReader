"""
Tests pour le module llm_enhancer.
"""
import pytest
from src.llm_enhancer import (
    LLMEnhancer,
    LLMConfig,
    LLMProvider,
    EmotionType,
    NarrativeContext,
    EmotionResult,
    ProsodyHints,
    CharacterValidation,
    DialogueAttribution,
    LLMCache,
    create_enhancer,
    create_gemini_enhancer,
    create_ollama_enhancer,
)


class TestLLMConfig:
    """Tests pour LLMConfig."""

    def test_default_config(self):
        """Test configuration par défaut."""
        config = LLMConfig()
        assert config.provider == LLMProvider.OLLAMA
        assert config.model == "llama3.2"
        assert config.temperature == 0.3
        assert config.cache_enabled is True

    def test_auto_model_selection(self):
        """Test sélection automatique du modèle."""
        configs = [
            (LLMProvider.OLLAMA, "llama3.2"),
            (LLMProvider.OPENAI, "gpt-4o-mini"),
            (LLMProvider.ANTHROPIC, "claude-3-haiku-20240307"),
            (LLMProvider.GEMINI, "gemini-2.5-flash-preview-05-20"),
        ]
        for provider, expected_model in configs:
            config = LLMConfig(provider=provider)
            assert config.model == expected_model, f"Failed for {provider}"

    def test_custom_model(self):
        """Test modèle personnalisé."""
        config = LLMConfig(provider=LLMProvider.GEMINI, model="gemini-pro")
        assert config.model == "gemini-pro"


class TestLLMCache:
    """Tests pour LLMCache."""

    def test_cache_set_get(self):
        """Test set/get basique."""
        cache = LLMCache(ttl=3600)
        cache.set("prompt1", "response1")
        assert cache.get("prompt1") == "response1"

    def test_cache_miss(self):
        """Test cache miss."""
        cache = LLMCache(ttl=3600)
        assert cache.get("nonexistent") is None

    def test_cache_with_kwargs(self):
        """Test cache avec kwargs."""
        cache = LLMCache(ttl=3600)
        cache.set("prompt", "response1", system="sys1")
        cache.set("prompt", "response2", system="sys2")
        assert cache.get("prompt", system="sys1") == "response1"
        assert cache.get("prompt", system="sys2") == "response2"

    def test_cache_clear(self):
        """Test vidage du cache."""
        cache = LLMCache(ttl=3600)
        cache.set("prompt", "response")
        cache.clear()
        assert cache.get("prompt") is None


class TestLLMEnhancerHeuristics:
    """Tests pour LLMEnhancer avec heuristiques (sans LLM)."""

    @pytest.fixture
    def enhancer(self):
        """Crée un enhancer avec fallback heuristique."""
        return LLMEnhancer(LLMConfig(fallback_to_heuristics=True))

    def test_validate_character_french_name(self, enhancer):
        """Test validation d'un prénom français."""
        result = enhancer.validate_character_name("Marie", "« Bonjour ! » dit Marie.")
        assert result.is_character is True
        assert result.confidence >= 0.5

    def test_validate_character_stop_word(self, enhancer):
        """Test rejet d'un stop word."""
        result = enhancer._validate_character_heuristic("coupé", "Il a coupé la parole.")
        assert result.is_character is False
        assert "stop word" in result.reasoning.lower()

    def test_analyze_emotion_joy(self, enhancer):
        """Test détection de joie."""
        result = enhancer._analyze_emotion_heuristic("Elle sourit de bonheur.")
        assert result.primary_emotion == EmotionType.JOY

    def test_analyze_emotion_sadness(self, enhancer):
        """Test détection de tristesse."""
        result = enhancer._analyze_emotion_heuristic("Il pleurait doucement.")
        assert result.primary_emotion == EmotionType.SADNESS

    def test_analyze_emotion_fear(self, enhancer):
        """Test détection de peur."""
        result = enhancer._analyze_emotion_heuristic("Elle tremblait de peur.")
        assert result.primary_emotion == EmotionType.FEAR

    def test_analyze_emotion_suspense(self, enhancer):
        """Test détection de suspense."""
        result = enhancer._analyze_emotion_heuristic("Soudain, une ombre apparut.")
        assert result.primary_emotion == EmotionType.SUSPENSE

    def test_analyze_emotion_exclamation(self, enhancer):
        """Test détection avec ponctuation."""
        result = enhancer._analyze_emotion_heuristic("C'est incroyable !")
        assert result.primary_emotion == EmotionType.EXCITEMENT

    def test_analyze_emotion_question(self, enhancer):
        """Test détection avec question."""
        result = enhancer._analyze_emotion_heuristic("Que se passe-t-il ?")
        assert result.primary_emotion == EmotionType.CURIOSITY

    def test_analyze_emotion_neutral(self, enhancer):
        """Test texte neutre."""
        result = enhancer._analyze_emotion_heuristic("Il marcha dans la rue.")
        assert result.primary_emotion == EmotionType.NEUTRAL

    def test_detect_narrative_dialogue(self, enhancer):
        """Test détection de dialogue."""
        context, conf = enhancer._detect_narrative_heuristic("« Tu viens ? » demanda-t-il.")
        assert context == NarrativeContext.DIALOGUE

    def test_detect_narrative_action(self, enhancer):
        """Test détection d'action."""
        context, conf = enhancer._detect_narrative_heuristic("Il courut vers la porte.")
        assert context == NarrativeContext.ACTION

    def test_detect_narrative_introspection(self, enhancer):
        """Test détection d'introspection."""
        context, conf = enhancer._detect_narrative_heuristic("Elle pensait à son passé.")
        assert context == NarrativeContext.INTROSPECTION

    def test_detect_narrative_flashback(self, enhancer):
        """Test détection de flashback."""
        context, conf = enhancer._detect_narrative_heuristic("Il se souvint de son enfance.")
        assert context == NarrativeContext.FLASHBACK

    def test_detect_narrative_suspense(self, enhancer):
        """Test détection de suspense."""
        context, conf = enhancer._detect_narrative_heuristic("Soudain, le silence se fit.")
        assert context == NarrativeContext.SUSPENSE

    def test_suggest_prosody_joy(self, enhancer):
        """Test prosodie pour joie."""
        prosody = enhancer._suggest_prosody_heuristic(EmotionType.JOY, None)
        assert prosody.speed > 1.0
        assert prosody.pitch > 0

    def test_suggest_prosody_sadness(self, enhancer):
        """Test prosodie pour tristesse."""
        prosody = enhancer._suggest_prosody_heuristic(EmotionType.SADNESS, None)
        assert prosody.speed < 1.0
        assert prosody.pitch < 0

    def test_suggest_prosody_with_context(self, enhancer):
        """Test prosodie avec contexte."""
        prosody = enhancer._suggest_prosody_heuristic(None, NarrativeContext.ACTION)
        assert prosody.speed > 1.0

    def test_suggest_prosody_suspense(self, enhancer):
        """Test prosodie pour suspense."""
        prosody = enhancer._suggest_prosody_heuristic(EmotionType.SUSPENSE, None)
        assert prosody.speed < 1.0
        assert prosody.pause_after > 0


class TestLLMEnhancerDataclasses:
    """Tests pour les dataclasses."""

    def test_emotion_result(self):
        """Test EmotionResult."""
        result = EmotionResult(
            primary_emotion=EmotionType.JOY,
            intensity=0.8,
            confidence=0.9,
        )
        assert result.primary_emotion == EmotionType.JOY
        assert result.intensity == 0.8
        assert result.secondary_emotion is None

    def test_prosody_hints_defaults(self):
        """Test ProsodyHints valeurs par défaut."""
        hints = ProsodyHints()
        assert hints.speed == 1.0
        assert hints.pitch == 0.0
        assert hints.volume == 1.0
        assert hints.emphasis_words == []

    def test_character_validation(self):
        """Test CharacterValidation."""
        validation = CharacterValidation(
            is_character=True,
            confidence=0.9,
            reasoning="Prénom reconnu",
            suggested_gender="F",
        )
        assert validation.is_character is True
        assert validation.suggested_gender == "F"

    def test_dialogue_attribution(self):
        """Test DialogueAttribution."""
        attr = DialogueAttribution(
            speaker="Marie",
            confidence=0.85,
            method="explicit",
            emotion=EmotionType.JOY,
        )
        assert attr.speaker == "Marie"
        assert attr.method == "explicit"


class TestFactoryFunctions:
    """Tests pour les fonctions factory."""

    def test_create_enhancer_ollama(self):
        """Test création avec Ollama."""
        enhancer = create_enhancer(provider="ollama", model="llama3.2")
        assert enhancer.config.provider == LLMProvider.OLLAMA
        assert enhancer.config.model == "llama3.2"

    def test_create_enhancer_gemini(self):
        """Test création avec Gemini."""
        enhancer = create_enhancer(provider="gemini")
        assert enhancer.config.provider == LLMProvider.GEMINI
        assert "gemini" in enhancer.config.model

    def test_create_gemini_enhancer(self):
        """Test factory Gemini."""
        enhancer = create_gemini_enhancer(model="gemini-pro")
        assert enhancer.config.provider == LLMProvider.GEMINI
        assert enhancer.config.model == "gemini-pro"

    def test_create_ollama_enhancer(self):
        """Test factory Ollama."""
        enhancer = create_ollama_enhancer(model="mistral")
        assert enhancer.config.provider == LLMProvider.OLLAMA
        assert enhancer.config.model == "mistral"

    def test_create_enhancer_with_kwargs(self):
        """Test création avec paramètres supplémentaires."""
        enhancer = create_enhancer(
            provider="openai",
            temperature=0.7,
            max_tokens=2048,
        )
        assert enhancer.config.temperature == 0.7
        assert enhancer.config.max_tokens == 2048


class TestEnhanceTextPipeline:
    """Tests pour le pipeline complet."""

    @pytest.fixture
    def enhancer(self):
        """Crée un enhancer avec fallback."""
        return LLMEnhancer(LLMConfig(fallback_to_heuristics=True))

    def test_enhance_text_basic(self, enhancer):
        """Test pipeline basique."""
        result = enhancer.enhance_text_for_tts(
            "Soudain, un cri retentit !",
            insert_tags=False,  # Pas de LLM disponible
        )
        assert "original_text" in result
        assert "enhanced_text" in result
        assert "emotion" in result
        assert "narrative_context" in result

    def test_enhance_text_emotion_detected(self, enhancer):
        """Test détection d'émotion dans pipeline."""
        result = enhancer.enhance_text_for_tts(
            "Elle sourit de bonheur.",
            insert_tags=False,
        )
        assert result["emotion"]["primary"] == "joy"

    def test_enhance_text_context_detected(self, enhancer):
        """Test détection de contexte dans pipeline."""
        result = enhancer.enhance_text_for_tts(
            "« Tu viens ? » demanda Marie.",
            insert_tags=False,
        )
        assert result["narrative_context"]["type"] == "dialogue"

    def test_enhance_text_prosody_included(self, enhancer):
        """Test inclusion de prosodie."""
        result = enhancer.enhance_text_for_tts(
            "Il courut aussi vite qu'il put.",
            insert_tags=False,
            suggest_prosody=True,
        )
        assert result["prosody"] is not None
        assert "speed" in result["prosody"]


class TestEmotionTypes:
    """Tests pour les types d'émotions."""

    def test_all_emotion_types_exist(self):
        """Vérifie que tous les types d'émotions sont définis."""
        expected = [
            "neutral", "joy", "sadness", "anger", "fear", "surprise",
            "disgust", "suspense", "irony", "tenderness", "excitement",
            "nostalgia", "hope", "despair", "curiosity", "determination",
            "relief", "anxiety"
        ]
        actual = [e.value for e in EmotionType]
        for emotion in expected:
            assert emotion in actual, f"Missing emotion: {emotion}"


class TestNarrativeContextTypes:
    """Tests pour les types de contexte narratif."""

    def test_all_context_types_exist(self):
        """Vérifie que tous les types de contexte sont définis."""
        expected = [
            "action", "description", "dialogue", "introspection",
            "flashback", "suspense", "transition", "climax", "resolution"
        ]
        actual = [c.value for c in NarrativeContext]
        for context in expected:
            assert context in actual, f"Missing context: {context}"


class TestLLMProviders:
    """Tests pour les providers LLM."""

    def test_all_providers_exist(self):
        """Vérifie que tous les providers sont définis."""
        expected = ["ollama", "openai", "anthropic", "gemini"]
        actual = [p.value for p in LLMProvider]
        for provider in expected:
            assert provider in actual, f"Missing provider: {provider}"


class TestCharacterDetectorIntegration:
    """Tests pour l'intégration avec CharacterDetector."""

    def test_character_detector_with_llm_enhancer(self):
        """Test du CharacterDetector avec LLMEnhancer."""
        from src.character_detector import CharacterDetector

        # Créer un enhancer avec fallback heuristique
        enhancer = LLMEnhancer(LLMConfig(fallback_to_heuristics=True))
        detector = CharacterDetector(lang="fr", llm_enhancer=enhancer)

        text = """
        « Bonjour », dit Marie.
        « Comment vas-tu ? » répondit Pierre.
        Il a coupé court à la discussion.
        """

        segments = detector.detect_dialogue_segments(text)
        characters = detector.get_characters()

        # Marie et Pierre doivent être détectés
        char_names = [c.name for c in characters]
        assert "Marie" in char_names or "Pierre" in char_names

        # "coupé" ne doit PAS être un personnage
        assert "coupé" not in char_names
        assert "Coupé" not in char_names

    def test_validate_with_llm_fallback(self):
        """Test du fallback heuristique quand LLM non disponible."""
        from src.character_detector import CharacterDetector

        # Sans LLM enhancer
        detector = CharacterDetector(lang="fr", llm_enhancer=None)

        # _validate_with_llm doit retourner None sans enhancer
        result = detector._validate_with_llm("Marie", "dit Marie")
        assert result is None

    def test_confidence_scoring_with_names(self):
        """Test du scoring de confiance pour les prénoms."""
        from src.character_detector import CharacterDetector

        detector = CharacterDetector(lang="fr")

        # Prénom français connu = haute confiance
        conf_marie = detector._calculate_name_confidence("Marie", "dit Marie")
        assert conf_marie >= 0.5

        # Stop word = confiance nulle
        conf_pas = detector._calculate_name_confidence("pas", "dit pas")
        assert conf_pas == 0.0

        # Mot inconnu = confiance basse
        conf_xyz = detector._calculate_name_confidence("xyz", "dit xyz")
        assert conf_xyz < 0.5
