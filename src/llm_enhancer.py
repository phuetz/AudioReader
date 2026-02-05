"""
Pipeline LLM unifié pour AudioReader v5.0.

Centralise toutes les améliorations basées sur LLM:
- Validation de noms de personnages
- Insertion automatique de tags audio
- Analyse émotionnelle contextuelle
- Attribution de dialogue intelligente
- Détection de contexte narratif
- Suggestions de prosodie

Providers supportés: Ollama, OpenAI, Anthropic, Gemini
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any
from functools import lru_cache
import json
import time
import hashlib
import os


class LLMProvider(Enum):
    """Providers LLM supportés."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class EmotionType(Enum):
    """Types d'émotions détectables."""
    NEUTRAL = "neutral"
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    SUSPENSE = "suspense"
    IRONY = "irony"
    TENDERNESS = "tenderness"
    EXCITEMENT = "excitement"
    NOSTALGIA = "nostalgia"
    HOPE = "hope"
    DESPAIR = "despair"
    CURIOSITY = "curiosity"
    DETERMINATION = "determination"
    RELIEF = "relief"
    ANXIETY = "anxiety"


class NarrativeContext(Enum):
    """Types de contexte narratif."""
    ACTION = "action"
    DESCRIPTION = "description"
    DIALOGUE = "dialogue"
    INTROSPECTION = "introspection"
    FLASHBACK = "flashback"
    SUSPENSE = "suspense"
    TRANSITION = "transition"
    CLIMAX = "climax"
    RESOLUTION = "resolution"


@dataclass
class LLMConfig:
    """Configuration du pipeline LLM."""
    provider: LLMProvider = LLMProvider.OLLAMA
    model: str = ""  # Auto-select based on provider
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 1024
    timeout: float = 30.0
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1 heure
    fallback_to_heuristics: bool = True

    def __post_init__(self):
        """Set default model based on provider."""
        if not self.model:
            defaults = {
                LLMProvider.OLLAMA: "llama3.2",
                LLMProvider.OPENAI: "gpt-4o-mini",
                LLMProvider.ANTHROPIC: "claude-3-haiku-20240307",
                LLMProvider.GEMINI: "gemini-2.5-flash-preview-05-20",
            }
            self.model = defaults.get(self.provider, "llama3.2")

        # Auto-detect API keys from environment
        if not self.api_key:
            env_keys = {
                LLMProvider.OPENAI: "OPENAI_API_KEY",
                LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
                LLMProvider.GEMINI: "GEMINI_API_KEY",
            }
            env_var = env_keys.get(self.provider)
            if env_var:
                self.api_key = os.environ.get(env_var)


@dataclass
class EmotionResult:
    """Résultat d'analyse émotionnelle."""
    primary_emotion: EmotionType
    intensity: float  # 0.0 - 1.0
    secondary_emotion: Optional[EmotionType] = None
    confidence: float = 0.5
    subtext: Optional[str] = None  # Sous-texte détecté
    prosody_hints: dict = field(default_factory=dict)


@dataclass
class DialogueAttribution:
    """Résultat d'attribution de dialogue."""
    speaker: str
    confidence: float
    method: str  # "llm", "explicit", "context", "alternation"
    emotion: Optional[EmotionType] = None
    reasoning: Optional[str] = None


@dataclass
class ProsodyHints:
    """Suggestions de prosodie."""
    speed: float = 1.0  # 0.5 - 2.0
    pitch: float = 0.0  # -1.0 - 1.0
    volume: float = 1.0  # 0.5 - 1.5
    pause_before: float = 0.0  # secondes
    pause_after: float = 0.0
    emphasis_words: list = field(default_factory=list)
    breathing: bool = False  # Insérer une respiration


@dataclass
class CharacterValidation:
    """Résultat de validation d'un nom de personnage."""
    is_character: bool
    confidence: float
    reasoning: Optional[str] = None
    suggested_gender: Optional[str] = None


class LLMCache:
    """Cache simple pour les réponses LLM."""

    def __init__(self, ttl: int = 3600):
        self.ttl = ttl
        self._cache: dict[str, tuple[Any, float]] = {}

    def _make_key(self, prompt: str, **kwargs) -> str:
        """Génère une clé de cache."""
        data = f"{prompt}:{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(data.encode()).hexdigest()

    def get(self, prompt: str, **kwargs) -> Optional[Any]:
        """Récupère une valeur du cache."""
        key = self._make_key(prompt, **kwargs)
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            del self._cache[key]
        return None

    def set(self, prompt: str, value: Any, **kwargs):
        """Stocke une valeur dans le cache."""
        key = self._make_key(prompt, **kwargs)
        self._cache[key] = (value, time.time())

    def clear(self):
        """Vide le cache."""
        self._cache.clear()


class LLMEnhancer:
    """
    Pipeline LLM unifié pour AudioReader.

    Centralise toutes les améliorations basées sur LLM avec:
    - Support multi-providers (Ollama, OpenAI, Anthropic, Gemini)
    - Cache partagé
    - Fallback gracieux vers heuristiques
    - Rate limiting
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialise le pipeline LLM.

        Args:
            config: Configuration LLM (défaut: Ollama local)
        """
        self.config = config or LLMConfig()
        self.cache = LLMCache(ttl=self.config.cache_ttl) if self.config.cache_enabled else None
        self._client = None
        self._last_request_time = 0.0
        self._min_request_interval = 0.1  # 100ms entre requêtes

    def _init_client(self):
        """Initialise le client LLM selon le provider."""
        if self._client is not None:
            return

        if self.config.provider == LLMProvider.OLLAMA:
            self._init_ollama()
        elif self.config.provider == LLMProvider.OPENAI:
            self._init_openai()
        elif self.config.provider == LLMProvider.ANTHROPIC:
            self._init_anthropic()
        elif self.config.provider == LLMProvider.GEMINI:
            self._init_gemini()

    def _init_ollama(self):
        """Initialise le client Ollama."""
        try:
            import ollama
            self._client = ollama.Client(
                host=self.config.base_url or "http://localhost:11434"
            )
        except ImportError:
            self._client = None

    def _init_openai(self):
        """Initialise le client OpenAI."""
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
            )
        except ImportError:
            self._client = None

    def _init_anthropic(self):
        """Initialise le client Anthropic."""
        try:
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=self.config.api_key,
                timeout=self.config.timeout,
            )
        except ImportError:
            self._client = None

    def _init_gemini(self):
        """Initialise le client Gemini."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.config.api_key)
            self._client = genai.GenerativeModel(
                model_name=self.config.model,
                generation_config={
                    "temperature": self.config.temperature,
                    "max_output_tokens": self.config.max_tokens,
                }
            )
        except ImportError:
            self._client = None

    def is_available(self) -> bool:
        """Vérifie si le LLM est disponible."""
        self._init_client()
        if self._client is None:
            return False

        # Test de connexion rapide
        try:
            if self.config.provider == LLMProvider.OLLAMA:
                self._client.list()
            elif self.config.provider == LLMProvider.GEMINI:
                # Gemini ne nécessite pas de test préalable
                return self.config.api_key is not None
            return True
        except Exception:
            return False

    def _rate_limit(self):
        """Applique le rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _call_llm(self, prompt: str, system: str = "") -> Optional[str]:
        """
        Appelle le LLM avec le prompt donné.

        Args:
            prompt: Le prompt utilisateur
            system: Le prompt système

        Returns:
            La réponse du LLM ou None si erreur
        """
        # Vérifier le cache
        if self.cache:
            cached = self.cache.get(prompt, system=system)
            if cached is not None:
                return cached

        self._init_client()
        if self._client is None:
            return None

        self._rate_limit()

        try:
            response = None

            if self.config.provider == LLMProvider.OLLAMA:
                result = self._client.chat(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": system} if system else None,
                        {"role": "user", "content": prompt},
                    ],
                    options={"temperature": self.config.temperature},
                )
                response = result["message"]["content"]

            elif self.config.provider == LLMProvider.OPENAI:
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})

                result = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                response = result.choices[0].message.content

            elif self.config.provider == LLMProvider.ANTHROPIC:
                result = self._client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    system=system if system else "",
                    messages=[{"role": "user", "content": prompt}],
                )
                response = result.content[0].text

            elif self.config.provider == LLMProvider.GEMINI:
                full_prompt = f"{system}\n\n{prompt}" if system else prompt
                result = self._client.generate_content(full_prompt)
                response = result.text

            # Mettre en cache
            if self.cache and response:
                self.cache.set(prompt, response, system=system)

            return response

        except Exception as e:
            print(f"Erreur LLM ({self.config.provider.value}): {e}")
            return None

    # =========================================================================
    # MÉTHODES PRINCIPALES
    # =========================================================================

    def validate_character_name(
        self,
        name: str,
        context: str,
        existing_characters: Optional[list[str]] = None
    ) -> CharacterValidation:
        """
        Valide si un nom est un personnage ou un faux positif.

        Args:
            name: Le nom à valider
            context: Le contexte où le nom apparaît
            existing_characters: Liste des personnages déjà détectés

        Returns:
            CharacterValidation avec is_character, confidence, reasoning
        """
        system = """Tu es un expert en analyse littéraire française.
Tu dois déterminer si un mot est un nom de personnage dans un texte narratif.
Réponds UNIQUEMENT en JSON avec ce format:
{"is_character": true/false, "confidence": 0.0-1.0, "reasoning": "explication", "gender": "M"/"F"/null}"""

        chars_info = ""
        if existing_characters:
            chars_info = f"\nPersonnages déjà identifiés: {', '.join(existing_characters)}"

        prompt = f"""Analyse ce mot dans son contexte:

Mot à analyser: "{name}"
Contexte: "{context}"{chars_info}

Le mot "{name}" est-il un nom de personnage (prénom ou nom propre d'une personne)?
Attention aux faux positifs: participes passés (coupé, pointé), adverbes, noms communs."""

        response = self._call_llm(prompt, system)

        if response:
            try:
                # Extraire le JSON de la réponse
                json_str = response
                if "```json" in response:
                    json_str = response.split("```json")[1].split("```")[0]
                elif "```" in response:
                    json_str = response.split("```")[1].split("```")[0]

                data = json.loads(json_str.strip())
                return CharacterValidation(
                    is_character=data.get("is_character", False),
                    confidence=data.get("confidence", 0.5),
                    reasoning=data.get("reasoning"),
                    suggested_gender=data.get("gender"),
                )
            except (json.JSONDecodeError, KeyError):
                pass

        # Fallback heuristique
        if self.config.fallback_to_heuristics:
            return self._validate_character_heuristic(name, context)

        return CharacterValidation(is_character=False, confidence=0.0)

    def _validate_character_heuristic(self, name: str, context: str) -> CharacterValidation:
        """Validation heuristique de nom de personnage."""
        # Importer le dictionnaire de prénoms
        try:
            from src.french_names import is_french_name, get_gender_from_name

            if is_french_name(name):
                return CharacterValidation(
                    is_character=True,
                    confidence=0.8,
                    reasoning="Prénom français reconnu",
                    suggested_gender=get_gender_from_name(name),
                )
        except ImportError:
            pass

        # Vérifier si c'est un mot de stop
        stop_words = {
            "pas", "plus", "jamais", "rien", "coupé", "pointé", "tourné",
            "passé", "fini", "parti", "rire", "sourire", "soupir",
        }
        if name.lower() in stop_words:
            return CharacterValidation(
                is_character=False,
                confidence=0.9,
                reasoning="Stop word détecté",
            )

        # Par défaut, incertain
        return CharacterValidation(
            is_character=True,  # Conservateur
            confidence=0.3,
            reasoning="Analyse heuristique incertaine",
        )

    def auto_insert_audio_tags(
        self,
        text: str,
        available_tags: Optional[list[str]] = None
    ) -> str:
        """
        Insère automatiquement des tags audio dans le texte.

        Args:
            text: Le texte à enrichir
            available_tags: Liste des tags disponibles

        Returns:
            Le texte avec tags insérés
        """
        if available_tags is None:
            available_tags = [
                "whispers", "excited", "sad", "angry", "fearful",
                "tender", "dramatic", "sarcastic", "cheerful",
                "laugh", "sigh", "gasp", "pause", "long pause"
            ]

        system = f"""Tu es un expert en narration audio. Tu dois insérer des tags expressifs dans le texte pour guider la synthèse vocale.

Tags disponibles: {', '.join(f'[{t}]' for t in available_tags)}

Règles:
1. Insère les tags AVANT le texte qu'ils affectent
2. N'en abuse pas - seulement aux moments clés
3. [pause] pour les moments de silence dramatique
4. Retourne UNIQUEMENT le texte modifié, sans explication"""

        prompt = f"""Insère des tags audio appropriés dans ce texte:

{text}

Texte avec tags:"""

        response = self._call_llm(prompt, system)

        if response:
            # Nettoyer la réponse
            result = response.strip()
            # Supprimer les balises de code si présentes
            if result.startswith("```"):
                lines = result.split("\n")
                result = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            return result

        # Fallback: retourner le texte original
        return text

    def analyze_emotion_contextual(
        self,
        text: str,
        previous_emotions: Optional[list[EmotionType]] = None,
        character: Optional[str] = None,
        narrative_context: Optional[NarrativeContext] = None,
    ) -> EmotionResult:
        """
        Analyse l'émotion d'un texte avec contexte.

        Args:
            text: Le texte à analyser
            previous_emotions: Historique des émotions récentes
            character: Nom du personnage qui parle
            narrative_context: Type de contexte narratif

        Returns:
            EmotionResult avec émotion primaire, intensité, etc.
        """
        emotions_list = [e.value for e in EmotionType]

        context_info = ""
        if previous_emotions:
            context_info += f"\nÉmotions précédentes: {', '.join(e.value for e in previous_emotions[-3:])}"
        if character:
            context_info += f"\nPersonnage: {character}"
        if narrative_context:
            context_info += f"\nContexte narratif: {narrative_context.value}"

        system = f"""Tu es un expert en analyse émotionnelle de textes littéraires.
Analyse l'émotion exprimée, en tenant compte du contexte et du sous-texte.

Émotions possibles: {', '.join(emotions_list)}

Réponds UNIQUEMENT en JSON:
{{"primary": "emotion", "intensity": 0.0-1.0, "secondary": "emotion"/null, "confidence": 0.0-1.0, "subtext": "sous-texte détecté"/null, "prosody": {{"speed": 0.8-1.2, "pitch": -0.5-0.5, "volume": 0.8-1.2}}}}"""

        prompt = f"""Analyse l'émotion de ce texte:{context_info}

Texte: "{text}"

Analyse émotionnelle:"""

        response = self._call_llm(prompt, system)

        if response:
            try:
                json_str = response
                if "```json" in response:
                    json_str = response.split("```json")[1].split("```")[0]
                elif "```" in response:
                    json_str = response.split("```")[1].split("```")[0]

                data = json.loads(json_str.strip())

                primary = EmotionType.NEUTRAL
                for e in EmotionType:
                    if e.value == data.get("primary", "neutral"):
                        primary = e
                        break

                secondary = None
                if data.get("secondary"):
                    for e in EmotionType:
                        if e.value == data.get("secondary"):
                            secondary = e
                            break

                prosody = data.get("prosody", {})

                return EmotionResult(
                    primary_emotion=primary,
                    intensity=data.get("intensity", 0.5),
                    secondary_emotion=secondary,
                    confidence=data.get("confidence", 0.5),
                    subtext=data.get("subtext"),
                    prosody_hints={
                        "speed": prosody.get("speed", 1.0),
                        "pitch": prosody.get("pitch", 0.0),
                        "volume": prosody.get("volume", 1.0),
                    }
                )
            except (json.JSONDecodeError, KeyError):
                pass

        # Fallback heuristique
        if self.config.fallback_to_heuristics:
            return self._analyze_emotion_heuristic(text)

        return EmotionResult(primary_emotion=EmotionType.NEUTRAL, intensity=0.5)

    def _analyze_emotion_heuristic(self, text: str) -> EmotionResult:
        """Analyse émotionnelle heuristique."""
        text_lower = text.lower()

        # Mots-clés simples (avec variantes conjuguées)
        keywords = {
            EmotionType.JOY: ["joie", "heureux", "rire", "sourire", "bonheur", "content", "ravie"],
            EmotionType.SADNESS: ["triste", "pleurer", "pleurait", "larme", "chagrin", "mélancolie", "sanglot"],
            EmotionType.ANGER: ["colère", "furieux", "rage", "énervé", "irrité", "fureur"],
            EmotionType.FEAR: ["peur", "terrifié", "angoisse", "effroi", "trembler", "terreur"],
            EmotionType.SURPRISE: ["surpris", "étonné", "stupéfait"],
            EmotionType.SUSPENSE: ["soudain", "mystère", "ombre", "silence", "attendre"],
            EmotionType.EXCITEMENT: ["incroyable", "fantastique", "génial", "extraordinaire"],
        }

        for emotion, words in keywords.items():
            if any(w in text_lower for w in words):
                return EmotionResult(
                    primary_emotion=emotion,
                    intensity=0.6,
                    confidence=0.4,
                )

        # Ponctuation
        if "!" in text:
            return EmotionResult(
                primary_emotion=EmotionType.EXCITEMENT,
                intensity=0.5,
                confidence=0.3,
            )
        if "?" in text:
            return EmotionResult(
                primary_emotion=EmotionType.CURIOSITY,
                intensity=0.4,
                confidence=0.3,
            )

        return EmotionResult(primary_emotion=EmotionType.NEUTRAL, intensity=0.3, confidence=0.5)

    def attribute_dialogue(
        self,
        dialogue: str,
        context: str,
        known_characters: Optional[list[str]] = None,
        last_speaker: Optional[str] = None,
    ) -> DialogueAttribution:
        """
        Attribue un dialogue à un personnage.

        Args:
            dialogue: Le dialogue à attribuer
            context: Le contexte entourant le dialogue
            known_characters: Liste des personnages connus
            last_speaker: Dernier personnage ayant parlé

        Returns:
            DialogueAttribution avec speaker, confidence, method
        """
        chars_info = ""
        if known_characters:
            chars_info = f"\nPersonnages connus: {', '.join(known_characters)}"
        if last_speaker:
            chars_info += f"\nDernier locuteur: {last_speaker}"

        system = """Tu es un expert en analyse littéraire. Tu dois identifier QUI prononce un dialogue.

Réponds UNIQUEMENT en JSON:
{"speaker": "nom", "confidence": 0.0-1.0, "method": "explicit/context/alternation/inference", "emotion": "emotion"/null, "reasoning": "explication"}"""

        prompt = f"""Qui prononce ce dialogue?{chars_info}

Contexte: "{context}"

Dialogue: "{dialogue}"

Attribution:"""

        response = self._call_llm(prompt, system)

        if response:
            try:
                json_str = response
                if "```json" in response:
                    json_str = response.split("```json")[1].split("```")[0]
                elif "```" in response:
                    json_str = response.split("```")[1].split("```")[0]

                data = json.loads(json_str.strip())

                emotion = None
                if data.get("emotion"):
                    for e in EmotionType:
                        if e.value == data.get("emotion"):
                            emotion = e
                            break

                return DialogueAttribution(
                    speaker=data.get("speaker", "NARRATEUR"),
                    confidence=data.get("confidence", 0.5),
                    method=data.get("method", "llm"),
                    emotion=emotion,
                    reasoning=data.get("reasoning"),
                )
            except (json.JSONDecodeError, KeyError):
                pass

        # Fallback
        return DialogueAttribution(
            speaker=last_speaker or "NARRATEUR",
            confidence=0.3,
            method="fallback",
        )

    def detect_narrative_context(self, text: str) -> tuple[NarrativeContext, float]:
        """
        Détecte le type de contexte narratif.

        Args:
            text: Le texte à analyser

        Returns:
            Tuple (NarrativeContext, confidence)
        """
        contexts_list = [c.value for c in NarrativeContext]

        system = f"""Tu es un expert en analyse narrative. Identifie le type de contexte narratif du texte.

Types possibles: {', '.join(contexts_list)}

Réponds UNIQUEMENT en JSON:
{{"context": "type", "confidence": 0.0-1.0}}"""

        prompt = f"""Quel est le type de contexte narratif?

Texte: "{text}"

Analyse:"""

        response = self._call_llm(prompt, system)

        if response:
            try:
                json_str = response
                if "```json" in response:
                    json_str = response.split("```json")[1].split("```")[0]
                elif "```" in response:
                    json_str = response.split("```")[1].split("```")[0]

                data = json.loads(json_str.strip())

                context = NarrativeContext.DESCRIPTION
                for c in NarrativeContext:
                    if c.value == data.get("context"):
                        context = c
                        break

                return context, data.get("confidence", 0.5)
            except (json.JSONDecodeError, KeyError):
                pass

        # Fallback heuristique
        return self._detect_narrative_heuristic(text)

    def _detect_narrative_heuristic(self, text: str) -> tuple[NarrativeContext, float]:
        """Détection heuristique du contexte narratif."""
        text_lower = text.lower()

        # Dialogue
        if any(c in text for c in ['«', '»', '"', '—', '–']):
            return NarrativeContext.DIALOGUE, 0.7

        # Action
        action_verbs = ["courut", "sauta", "frappa", "saisit", "bondit", "s'élança"]
        if any(v in text_lower for v in action_verbs):
            return NarrativeContext.ACTION, 0.6

        # Introspection
        intro_words = ["pensait", "songeait", "se demandait", "réfléchissait"]
        if any(w in text_lower for w in intro_words):
            return NarrativeContext.INTROSPECTION, 0.6

        # Flashback
        if any(w in text_lower for w in ["se souvint", "autrefois", "jadis", "à l'époque"]):
            return NarrativeContext.FLASHBACK, 0.6

        # Suspense
        if any(w in text_lower for w in ["soudain", "tout à coup", "silence", "ombre"]):
            return NarrativeContext.SUSPENSE, 0.5

        return NarrativeContext.DESCRIPTION, 0.4

    def suggest_prosody(
        self,
        text: str,
        emotion: Optional[EmotionType] = None,
        context: Optional[NarrativeContext] = None,
        character: Optional[str] = None,
    ) -> ProsodyHints:
        """
        Suggère des paramètres de prosodie pour un segment.

        Args:
            text: Le texte à synthétiser
            emotion: Émotion détectée
            context: Contexte narratif
            character: Personnage qui parle

        Returns:
            ProsodyHints avec speed, pitch, volume, pauses, etc.
        """
        info = ""
        if emotion:
            info += f"\nÉmotion: {emotion.value}"
        if context:
            info += f"\nContexte: {context.value}"
        if character:
            info += f"\nPersonnage: {character}"

        system = """Tu es un expert en prosodie et synthèse vocale.
Suggère les paramètres de prosodie optimaux pour la lecture de ce texte.

Réponds UNIQUEMENT en JSON:
{"speed": 0.8-1.2, "pitch": -0.5-0.5, "volume": 0.8-1.2, "pause_before": 0.0-1.0, "pause_after": 0.0-1.0, "emphasis_words": ["mot1", "mot2"], "breathing": true/false}"""

        prompt = f"""Suggère la prosodie pour ce texte:{info}

Texte: "{text}"

Prosodie:"""

        response = self._call_llm(prompt, system)

        if response:
            try:
                json_str = response
                if "```json" in response:
                    json_str = response.split("```json")[1].split("```")[0]
                elif "```" in response:
                    json_str = response.split("```")[1].split("```")[0]

                data = json.loads(json_str.strip())

                return ProsodyHints(
                    speed=data.get("speed", 1.0),
                    pitch=data.get("pitch", 0.0),
                    volume=data.get("volume", 1.0),
                    pause_before=data.get("pause_before", 0.0),
                    pause_after=data.get("pause_after", 0.0),
                    emphasis_words=data.get("emphasis_words", []),
                    breathing=data.get("breathing", False),
                )
            except (json.JSONDecodeError, KeyError):
                pass

        # Fallback basé sur l'émotion
        return self._suggest_prosody_heuristic(emotion, context)

    def _suggest_prosody_heuristic(
        self,
        emotion: Optional[EmotionType],
        context: Optional[NarrativeContext]
    ) -> ProsodyHints:
        """Suggestion de prosodie heuristique."""
        hints = ProsodyHints()

        if emotion:
            presets = {
                EmotionType.JOY: ProsodyHints(speed=1.1, pitch=0.2, volume=1.1),
                EmotionType.SADNESS: ProsodyHints(speed=0.9, pitch=-0.2, volume=0.9),
                EmotionType.ANGER: ProsodyHints(speed=1.15, pitch=0.1, volume=1.2),
                EmotionType.FEAR: ProsodyHints(speed=1.1, pitch=0.3, volume=0.95),
                EmotionType.SUSPENSE: ProsodyHints(speed=0.85, pitch=-0.1, pause_after=0.3),
                EmotionType.EXCITEMENT: ProsodyHints(speed=1.15, pitch=0.25, volume=1.1),
            }
            hints = presets.get(emotion, hints)

        if context:
            context_mods = {
                NarrativeContext.ACTION: {"speed": 1.1},
                NarrativeContext.DESCRIPTION: {"speed": 0.95},
                NarrativeContext.INTROSPECTION: {"speed": 0.9, "volume": 0.95},
                NarrativeContext.SUSPENSE: {"speed": 0.85, "pause_after": 0.4},
            }
            mods = context_mods.get(context, {})
            for key, value in mods.items():
                setattr(hints, key, value)

        return hints

    def enhance_text_for_tts(
        self,
        text: str,
        insert_tags: bool = True,
        detect_emotions: bool = True,
        suggest_prosody: bool = True,
    ) -> dict:
        """
        Pipeline complet d'amélioration de texte pour TTS.

        Args:
            text: Le texte à améliorer
            insert_tags: Insérer automatiquement des tags audio
            detect_emotions: Détecter les émotions
            suggest_prosody: Suggérer la prosodie

        Returns:
            Dict avec text, emotion, prosody, etc.
        """
        result = {
            "original_text": text,
            "enhanced_text": text,
            "emotion": None,
            "narrative_context": None,
            "prosody": None,
        }

        # 1. Détecter le contexte narratif
        context, ctx_confidence = self.detect_narrative_context(text)
        result["narrative_context"] = {
            "type": context.value,
            "confidence": ctx_confidence,
        }

        # 2. Analyser les émotions
        if detect_emotions:
            emotion = self.analyze_emotion_contextual(text, narrative_context=context)
            result["emotion"] = {
                "primary": emotion.primary_emotion.value,
                "intensity": emotion.intensity,
                "secondary": emotion.secondary_emotion.value if emotion.secondary_emotion else None,
                "confidence": emotion.confidence,
                "subtext": emotion.subtext,
            }

        # 3. Insérer les tags audio
        if insert_tags:
            result["enhanced_text"] = self.auto_insert_audio_tags(text)

        # 4. Suggérer la prosodie
        if suggest_prosody:
            emotion_type = EmotionType(result["emotion"]["primary"]) if result.get("emotion") else None
            prosody = self.suggest_prosody(text, emotion=emotion_type, context=context)
            result["prosody"] = {
                "speed": prosody.speed,
                "pitch": prosody.pitch,
                "volume": prosody.volume,
                "pause_before": prosody.pause_before,
                "pause_after": prosody.pause_after,
                "emphasis_words": prosody.emphasis_words,
                "breathing": prosody.breathing,
            }

        return result


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_enhancer(
    provider: str = "ollama",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> LLMEnhancer:
    """
    Crée un LLMEnhancer avec la configuration spécifiée.

    Args:
        provider: "ollama", "openai", "anthropic", ou "gemini"
        model: Nom du modèle (auto-détecté si non spécifié)
        api_key: Clé API (auto-détectée depuis l'environnement)
        **kwargs: Arguments supplémentaires pour LLMConfig

    Returns:
        LLMEnhancer configuré
    """
    provider_enum = LLMProvider(provider.lower())

    config = LLMConfig(
        provider=provider_enum,
        model=model or "",
        api_key=api_key,
        **kwargs
    )

    return LLMEnhancer(config)


def create_gemini_enhancer(
    api_key: Optional[str] = None,
    model: str = "gemini-2.5-flash-preview-05-20",
    **kwargs
) -> LLMEnhancer:
    """
    Crée un LLMEnhancer configuré pour Gemini 2.5 Flash.

    Args:
        api_key: Clé API Gemini (ou GEMINI_API_KEY env var)
        model: Modèle Gemini (défaut: gemini-2.5-flash-preview-05-20)
        **kwargs: Arguments supplémentaires

    Returns:
        LLMEnhancer configuré pour Gemini
    """
    return create_enhancer(
        provider="gemini",
        model=model,
        api_key=api_key,
        **kwargs
    )


def create_ollama_enhancer(
    model: str = "llama3.2",
    base_url: str = "http://localhost:11434",
    **kwargs
) -> LLMEnhancer:
    """
    Crée un LLMEnhancer configuré pour Ollama local.

    Args:
        model: Modèle Ollama (défaut: llama3.2)
        base_url: URL du serveur Ollama
        **kwargs: Arguments supplémentaires

    Returns:
        LLMEnhancer configuré pour Ollama
    """
    return create_enhancer(
        provider="ollama",
        model=model,
        base_url=base_url,
        **kwargs
    )


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=== Test LLM Enhancer ===\n")

    # Test avec fallback heuristique (pas besoin de LLM)
    enhancer = LLMEnhancer(LLMConfig(fallback_to_heuristics=True))

    # Test validation de personnage
    print("1. Validation de personnage:")
    for name in ["Marie", "coupé", "Victor", "pointé"]:
        result = enhancer.validate_character_name(name, f"« Bonjour ! » dit {name}.")
        print(f"   {name}: is_character={result.is_character}, confidence={result.confidence:.2f}")

    # Test analyse émotionnelle
    print("\n2. Analyse émotionnelle:")
    texts = [
        "Il sourit, le coeur léger et plein de joie.",
        "Une larme coula sur sa joue.",
        "Soudain, un bruit dans l'ombre...",
    ]
    for text in texts:
        result = enhancer.analyze_emotion_contextual(text)
        print(f"   \"{text[:40]}...\" -> {result.primary_emotion.value} ({result.intensity:.2f})")

    # Test contexte narratif
    print("\n3. Contexte narratif:")
    texts = [
        "« Tu viens ? » demanda Marie.",
        "Il courut aussi vite qu'il put.",
        "Elle pensait à son enfance.",
    ]
    for text in texts:
        context, conf = enhancer.detect_narrative_context(text)
        print(f"   \"{text[:40]}...\" -> {context.value} ({conf:.2f})")

    # Test prosodie
    print("\n4. Suggestions de prosodie:")
    prosody = enhancer.suggest_prosody(
        "Il murmura doucement son nom.",
        emotion=EmotionType.TENDERNESS,
    )
    print(f"   speed={prosody.speed}, pitch={prosody.pitch}, volume={prosody.volume}")

    # Test pipeline complet
    print("\n5. Pipeline complet:")
    result = enhancer.enhance_text_for_tts(
        "Soudain, un cri déchira le silence de la nuit !",
        insert_tags=False,  # Désactivé sans LLM
    )
    print(f"   Contexte: {result['narrative_context']['type']}")
    print(f"   Émotion: {result['emotion']['primary']} ({result['emotion']['intensity']:.2f})")

    print("\n=== Tests terminés ===")
