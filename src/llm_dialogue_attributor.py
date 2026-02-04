"""
Attribution de dialogues par LLM pour AudioReader.

Utilise un LLM (Ollama ou OpenAI) pour resoudre les ambiguites
d'attribution de dialogues aux personnages.
"""
import json
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class LLMAttribution:
    """Resultat d'attribution par LLM."""
    speaker: str
    text: str
    emotion: str = "neutral"
    confidence: float = 0.0
    method: str = "llm"


@dataclass
class LLMAttributorConfig:
    """Configuration de l'attributeur LLM."""
    provider: str = "ollama"  # "ollama" ou "openai"
    model: str = "llama3.2"
    temperature: float = 0.3
    base_url: str = "http://localhost:11434"
    api_key: Optional[str] = None


class LLMDialogueAttributor:
    """
    Attributeur de dialogues utilisant un LLM pour resoudre
    les ambiguites que le regex ne peut pas gerer.
    """

    PROMPT_TEMPLATE = """Analyse le texte suivant et identifie qui parle dans chaque dialogue.
Retourne un JSON avec la liste des dialogues attribues.

Personnages connus : {characters}

Texte :
{text}

Retourne UNIQUEMENT un JSON valide au format :
[{{"speaker": "Nom", "text": "dialogue", "emotion": "neutral|joy|anger|sadness|fear|surprise"}}]"""

    def __init__(self, config: Optional[LLMAttributorConfig] = None):
        self.config = config or LLMAttributorConfig()
        self._available = None
        self.known_characters: Dict[str, str] = {}  # name -> gender

    def register_character(self, name: str, gender: str = "?"):
        """Enregistre un personnage connu."""
        self.known_characters[name] = gender

    def is_available(self) -> bool:
        """Verifie si le LLM est disponible."""
        if self._available is not None:
            return self._available

        if self.config.provider == "ollama":
            try:
                import urllib.request
                req = urllib.request.Request(
                    f"{self.config.base_url}/api/tags",
                    method="GET",
                )
                urllib.request.urlopen(req, timeout=2)
                self._available = True
            except Exception:
                self._available = False
        elif self.config.provider == "openai":
            self._available = self.config.api_key is not None
        else:
            self._available = False

        return self._available

    def _call_ollama(self, prompt: str) -> str:
        """Appelle l'API Ollama."""
        import urllib.request

        payload = json.dumps({
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.config.temperature},
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.config.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "")

    def _call_openai(self, prompt: str) -> str:
        """Appelle l'API OpenAI."""
        import urllib.request

        payload = json.dumps({
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]

    def _parse_response(self, response: str) -> List[LLMAttribution]:
        """Parse la reponse JSON du LLM."""
        # Extraire le JSON de la reponse
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if not match:
            return []

        try:
            items = json.loads(match.group())
            return [
                LLMAttribution(
                    speaker=item.get("speaker", "Inconnu"),
                    text=item.get("text", ""),
                    emotion=item.get("emotion", "neutral"),
                    confidence=0.8,
                    method="llm",
                )
                for item in items
            ]
        except (json.JSONDecodeError, KeyError):
            return []

    def attribute_dialogues(self, text: str) -> List[LLMAttribution]:
        """
        Attribue les dialogues du texte aux personnages via LLM.

        Args:
            text: Texte contenant des dialogues

        Returns:
            Liste des dialogues attribues
        """
        if not self.is_available():
            return self._fallback_attribution(text)

        characters = ", ".join(
            f"{name} ({gender})" for name, gender in self.known_characters.items()
        ) or "aucun personnage connu"

        prompt = self.PROMPT_TEMPLATE.format(
            characters=characters,
            text=text[:3000],  # Limiter la taille
        )

        try:
            if self.config.provider == "ollama":
                response = self._call_ollama(prompt)
            elif self.config.provider == "openai":
                response = self._call_openai(prompt)
            else:
                return self._fallback_attribution(text)

            return self._parse_response(response)

        except Exception:
            return self._fallback_attribution(text)

    def _fallback_attribution(self, text: str) -> List[LLMAttribution]:
        """Fallback regex quand le LLM n'est pas disponible."""
        try:
            from .dialogue_attribution import DialogueAttributor
            attributor = DialogueAttributor()
            for name, gender in self.known_characters.items():
                attributor.register_character(name, gender)

            results = []
            # Trouver les dialogues dans le texte
            dialogue_pattern = re.compile(r'["\u00ab\u201c](.+?)["\u00bb\u201d]')
            for match in dialogue_pattern.finditer(text):
                dialogue_text = match.group(1)
                context = text[max(0, match.start() - 100):match.end() + 100]
                attr = attributor.attribute_dialogue(dialogue_text, context)
                results.append(LLMAttribution(
                    speaker=attr.speaker if attr else "Inconnu",
                    text=dialogue_text,
                    confidence=attr.confidence if attr else 0.0,
                    method="regex_fallback",
                ))
            return results
        except ImportError:
            return []
