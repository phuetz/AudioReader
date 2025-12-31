"""
Detection d'Emotion par LLM v2.4.

Utilise un LLM pour analyser le contexte emotionnel du texte:
- Comprend le sous-texte et l'ironie
- Analyse le contexte (paragraphes precedents)
- Detecte les emotions complexes

Strategies:
1. Analyse locale: emotion du segment actuel
2. Analyse contextuelle: emotion dans le contexte narratif
3. Analyse des personnages: etat emotionnel de chaque personnage

Supporte:
- Ollama (local): llama3, mistral, etc.
- OpenAI API (optionnel)
- Fallback vers detection par regles
"""
import json
import re
import subprocess
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class EmotionCategory(Enum):
    """Categories d'emotions detectables."""
    NEUTRAL = "neutral"
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TENDERNESS = "tenderness"
    EXCITEMENT = "excitement"
    SUSPENSE = "suspense"
    IRONY = "irony"
    MELANCHOLY = "melancholy"
    HOPE = "hope"
    DESPAIR = "despair"
    ANXIETY = "anxiety"
    RELIEF = "relief"
    NOSTALGIA = "nostalgia"
    DETERMINATION = "determination"


@dataclass
class EmotionResult:
    """Resultat de la detection d'emotion."""
    primary_emotion: EmotionCategory
    confidence: float  # 0.0 - 1.0
    secondary_emotion: Optional[EmotionCategory] = None
    intensity: float = 0.5  # 0.0 - 1.0
    valence: float = 0.0  # -1.0 (negatif) a +1.0 (positif)
    arousal: float = 0.5  # 0.0 (calme) a 1.0 (excite)
    explanation: str = ""
    detected_subtext: Optional[str] = None


@dataclass
class CharacterEmotionState:
    """Etat emotionnel d'un personnage."""
    character_name: str
    current_emotion: EmotionCategory
    emotion_history: List[EmotionCategory] = field(default_factory=list)
    mood_trend: str = "stable"  # "improving", "declining", "stable"


@dataclass
class LLMConfig:
    """Configuration pour le LLM."""
    provider: str = "ollama"  # "ollama", "openai", "anthropic"
    model: str = "llama3.2"  # ou "gpt-4", "claude-3", etc.
    base_url: str = "http://localhost:11434"
    api_key: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 500
    timeout: int = 30


class LLMEmotionDetector:
    """
    Detecteur d'emotions utilisant un LLM.

    Fournit une analyse emotionnelle contextuelle plus sophistiquee
    que la detection par regles.
    """

    # Prompt systeme pour l'analyse emotionnelle
    SYSTEM_PROMPT = """Tu es un expert en analyse emotionnelle de textes litteraires.
Ton role est d'identifier l'emotion principale et secondaire dans un passage de texte.

Pour chaque texte, tu dois retourner un JSON avec:
- primary_emotion: l'emotion principale (une parmi: neutral, joy, sadness, anger, fear, surprise, disgust, tenderness, excitement, suspense, irony, melancholy, hope, despair, anxiety, relief, nostalgia, determination)
- confidence: ta confiance de 0.0 a 1.0
- secondary_emotion: emotion secondaire (peut etre null)
- intensity: intensite de l'emotion de 0.0 a 1.0
- valence: -1.0 (negatif) a +1.0 (positif)
- arousal: 0.0 (calme) a 1.0 (excite)
- explanation: courte explication en francais
- subtext: si l'emotion de surface differe de l'emotion reelle (ex: ironie)

Sois attentif au sous-texte, a l'ironie, et au contexte narratif.
Reponds UNIQUEMENT avec le JSON, sans autre texte."""

    # Prompt pour analyse avec contexte
    CONTEXT_PROMPT_TEMPLATE = """Contexte precedent:
{context}

---
Texte a analyser:
{text}

---
Analyse l'emotion de ce texte en tenant compte du contexte. Retourne le JSON."""

    # Prompt simple (sans contexte)
    SIMPLE_PROMPT_TEMPLATE = """Texte a analyser:
{text}

Analyse l'emotion de ce texte. Retourne le JSON."""

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialise le detecteur.

        Args:
            config: Configuration du LLM
        """
        self.config = config or LLMConfig()
        self._llm_available = None
        self._character_states: Dict[str, CharacterEmotionState] = {}

    def is_available(self) -> bool:
        """Verifie si le LLM est disponible."""
        if self._llm_available is not None:
            return self._llm_available

        if self.config.provider == "ollama":
            self._llm_available = self._check_ollama()
        elif self.config.provider == "openai":
            self._llm_available = self.config.api_key is not None
        else:
            self._llm_available = False

        return self._llm_available

    def _check_ollama(self) -> bool:
        """Verifie si Ollama est disponible."""
        try:
            import requests
            response = requests.get(
                f"{self.config.base_url}/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False

    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Appelle Ollama pour generer une reponse."""
        try:
            import requests

            payload = {
                "model": self.config.model,
                "prompt": prompt,
                "system": self.SYSTEM_PROMPT,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                }
            }

            response = requests.post(
                f"{self.config.base_url}/api/generate",
                json=payload,
                timeout=self.config.timeout
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("response", "")

        except Exception as e:
            logger.warning(f"Erreur Ollama: {e}")

        return None

    def _call_openai(self, prompt: str) -> Optional[str]:
        """Appelle l'API OpenAI."""
        try:
            import requests

            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            }

            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.config.timeout
            )

            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]

        except Exception as e:
            logger.warning(f"Erreur OpenAI: {e}")

        return None

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Appelle le LLM configure."""
        if self.config.provider == "ollama":
            return self._call_ollama(prompt)
        elif self.config.provider == "openai":
            return self._call_openai(prompt)
        return None

    def _parse_llm_response(self, response: str) -> Optional[Dict]:
        """Parse la reponse JSON du LLM."""
        try:
            # Nettoyer la reponse (enlever markdown si present)
            clean = response.strip()
            if clean.startswith("```json"):
                clean = clean[7:]
            if clean.startswith("```"):
                clean = clean[3:]
            if clean.endswith("```"):
                clean = clean[:-3]

            # Trouver le JSON dans la reponse
            json_match = re.search(r'\{[^{}]*\}', clean, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())

            return json.loads(clean)

        except json.JSONDecodeError as e:
            logger.warning(f"Erreur parsing JSON: {e}")
            return None

    def detect_emotion(
        self,
        text: str,
        context: Optional[str] = None,
        character: Optional[str] = None
    ) -> EmotionResult:
        """
        Detecte l'emotion dans un texte.

        Args:
            text: Texte a analyser
            context: Contexte precedent (optionnel)
            character: Nom du personnage (pour suivi d'etat)

        Returns:
            EmotionResult avec l'analyse
        """
        # Verifier la disponibilite du LLM
        if not self.is_available():
            return self._fallback_detection(text)

        # Construire le prompt
        if context:
            prompt = self.CONTEXT_PROMPT_TEMPLATE.format(
                context=context[-500:],  # Limiter le contexte
                text=text
            )
        else:
            prompt = self.SIMPLE_PROMPT_TEMPLATE.format(text=text)

        # Appeler le LLM
        response = self._call_llm(prompt)

        if not response:
            return self._fallback_detection(text)

        # Parser la reponse
        data = self._parse_llm_response(response)

        if not data:
            return self._fallback_detection(text)

        # Construire le resultat
        try:
            primary = EmotionCategory(data.get("primary_emotion", "neutral"))
        except ValueError:
            primary = EmotionCategory.NEUTRAL

        secondary = None
        if data.get("secondary_emotion"):
            try:
                secondary = EmotionCategory(data["secondary_emotion"])
            except ValueError:
                pass

        result = EmotionResult(
            primary_emotion=primary,
            confidence=float(data.get("confidence", 0.7)),
            secondary_emotion=secondary,
            intensity=float(data.get("intensity", 0.5)),
            valence=float(data.get("valence", 0.0)),
            arousal=float(data.get("arousal", 0.5)),
            explanation=data.get("explanation", ""),
            detected_subtext=data.get("subtext")
        )

        # Mettre a jour l'etat du personnage si specifie
        if character:
            self._update_character_state(character, result)

        return result

    def _fallback_detection(self, text: str) -> EmotionResult:
        """
        Detection par regles en cas d'echec du LLM.

        Utilise des mots-cles simples.
        """
        text_lower = text.lower()

        # Mots-cles par emotion
        emotion_keywords = {
            EmotionCategory.JOY: ["heureux", "heureuse", "joie", "rire", "sourire", "content", "ravis"],
            EmotionCategory.SADNESS: ["triste", "pleurer", "larmes", "chagrin", "melancolie", "desole"],
            EmotionCategory.ANGER: ["colere", "furieux", "rage", "enerve", "crier", "hurler"],
            EmotionCategory.FEAR: ["peur", "terreur", "effraye", "anxieux", "trembler", "craindre"],
            EmotionCategory.SURPRISE: ["surpris", "etonne", "stupefait", "incroyable", "soudain"],
            EmotionCategory.SUSPENSE: ["suspense", "attendre", "silence", "immobile", "souffle"],
            EmotionCategory.TENDERNESS: ["tendresse", "doux", "caresse", "amour", "affection"],
            EmotionCategory.EXCITEMENT: ["excite", "enthousiaste", "impatient", "hate", "vibrant"],
            EmotionCategory.IRONY: ["ironique", "sarcastique", "ricaner", "evidemment", "bien sur"],
        }

        # Compter les correspondances
        scores = {}
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[emotion] = score

        if scores:
            primary = max(scores, key=scores.get)
            confidence = min(scores[primary] / 3, 0.8)
        else:
            primary = EmotionCategory.NEUTRAL
            confidence = 0.5

        return EmotionResult(
            primary_emotion=primary,
            confidence=confidence,
            intensity=0.5,
            valence=0.0,
            arousal=0.5,
            explanation="Detection par regles (fallback)"
        )

    def _update_character_state(self, character: str, emotion: EmotionResult):
        """Met a jour l'etat emotionnel d'un personnage."""
        if character not in self._character_states:
            self._character_states[character] = CharacterEmotionState(
                character_name=character,
                current_emotion=emotion.primary_emotion
            )

        state = self._character_states[character]
        state.emotion_history.append(emotion.primary_emotion)
        state.current_emotion = emotion.primary_emotion

        # Calculer la tendance (sur les 5 dernieres emotions)
        if len(state.emotion_history) >= 5:
            recent = state.emotion_history[-5:]
            positive = [EmotionCategory.JOY, EmotionCategory.TENDERNESS,
                       EmotionCategory.EXCITEMENT, EmotionCategory.HOPE]
            negative = [EmotionCategory.SADNESS, EmotionCategory.ANGER,
                       EmotionCategory.FEAR, EmotionCategory.DESPAIR]

            pos_count = sum(1 for e in recent if e in positive)
            neg_count = sum(1 for e in recent if e in negative)

            if pos_count > neg_count + 1:
                state.mood_trend = "improving"
            elif neg_count > pos_count + 1:
                state.mood_trend = "declining"
            else:
                state.mood_trend = "stable"

    def get_character_state(self, character: str) -> Optional[CharacterEmotionState]:
        """Retourne l'etat emotionnel d'un personnage."""
        return self._character_states.get(character)

    def analyze_paragraph(
        self,
        paragraph: str,
        previous_paragraphs: List[str] = None
    ) -> Dict:
        """
        Analyse un paragraphe complet avec son contexte.

        Args:
            paragraph: Paragraphe a analyser
            previous_paragraphs: Paragraphes precedents

        Returns:
            Dict avec analyse detaillee
        """
        context = ""
        if previous_paragraphs:
            context = "\n\n".join(previous_paragraphs[-3:])  # 3 derniers paragraphes

        emotion = self.detect_emotion(paragraph, context)

        # Analyser la structure du paragraphe
        sentences = re.split(r'[.!?]+', paragraph)
        sentence_emotions = []

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:
                sent_emotion = self.detect_emotion(sentence)
                sentence_emotions.append({
                    "text": sentence[:50] + "..." if len(sentence) > 50 else sentence,
                    "emotion": sent_emotion.primary_emotion.value,
                    "intensity": sent_emotion.intensity
                })

        return {
            "paragraph_emotion": emotion,
            "sentence_analysis": sentence_emotions,
            "emotional_arc": self._analyze_arc(sentence_emotions),
            "dominant_tone": emotion.primary_emotion.value
        }

    def _analyze_arc(self, sentence_emotions: List[Dict]) -> str:
        """Analyse l'arc emotionnel d'un paragraphe."""
        if len(sentence_emotions) < 2:
            return "stable"

        first_intensity = sentence_emotions[0].get("intensity", 0.5)
        last_intensity = sentence_emotions[-1].get("intensity", 0.5)

        if last_intensity > first_intensity + 0.2:
            return "rising"
        elif last_intensity < first_intensity - 0.2:
            return "falling"
        else:
            return "stable"

    def reset(self):
        """Reinitialise les etats des personnages."""
        self._character_states.clear()


def detect_emotion_with_llm(
    text: str,
    context: Optional[str] = None,
    provider: str = "ollama",
    model: str = "llama3.2"
) -> EmotionResult:
    """
    Fonction utilitaire pour detecter l'emotion avec un LLM.

    Args:
        text: Texte a analyser
        context: Contexte optionnel
        provider: Fournisseur LLM ("ollama" ou "openai")
        model: Modele a utiliser

    Returns:
        EmotionResult
    """
    config = LLMConfig(provider=provider, model=model)
    detector = LLMEmotionDetector(config)
    return detector.detect_emotion(text, context)


if __name__ == "__main__":
    print("=== Test Detection d'Emotion par LLM ===\n")

    # Configuration (Ollama local)
    config = LLMConfig(
        provider="ollama",
        model="llama3.2",
        temperature=0.3
    )

    detector = LLMEmotionDetector(config)

    # Verifier la disponibilite
    print(f"LLM disponible: {detector.is_available()}")

    # Textes de test
    test_texts = [
        # Simple
        ("Il etait heureux de la revoir apres tant d'annees.", None),

        # Ironie
        ("« Quelle surprise! » dit-il avec un sourire amer, sachant "
         "pertinemment qu'elle allait venir.", None),

        # Contexte important
        ("Elle sourit.", "Apres des mois de souffrance et de deuil, "
         "elle avait enfin trouve la paix."),

        # Emotion cachee
        ("« Tout va bien » repondit-il, les mains tremblantes.",
         "Il venait d'apprendre la terrible nouvelle."),
    ]

    for text, context in test_texts:
        print(f"\nTexte: \"{text[:60]}...\"")
        if context:
            print(f"Contexte: \"{context[:40]}...\"")

        result = detector.detect_emotion(text, context)

        print(f"  Emotion: {result.primary_emotion.value} "
              f"(confiance: {result.confidence:.2f})")
        print(f"  Intensite: {result.intensity:.2f}, "
              f"Valence: {result.valence:.2f}")
        if result.detected_subtext:
            print(f"  Sous-texte: {result.detected_subtext}")
        if result.explanation:
            print(f"  Explication: {result.explanation}")
