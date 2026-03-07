"""
Voice Designer — Creation de voix personnalisees pour Kokoro.

Permet de creer de nouvelles voix en interpolant les tenseurs de style
existants, sans aucun entrainement ni sample audio.

Techniques:
- Blending pondere de voix existantes
- Walk aleatoire dans l'espace d'embedding
- Presets de personnages (age, genre, timbre)

Necessite: kokoro-onnx avec le modele et voices charge.
"""
import numpy as np
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class VoiceDesign:
    """Description d'une voix personnalisee."""
    name: str
    description: str
    tensor: np.ndarray  # Shape (510, 1, 256) pour Kokoro v1.0
    source_voices: dict[str, float]  # {voice_id: weight}


# Presets de personnages par archetype
CHARACTER_PRESETS = {
    # Femmes
    "jeune_femme": {
        "base": {"af_heart": 0.4, "af_sky": 0.3, "af_nicole": 0.3},
        "description": "Voix feminine jeune, claire et dynamique",
    },
    "femme_mature": {
        "base": {"af_bella": 0.5, "af_sarah": 0.3, "ff_siwis": 0.2},
        "description": "Voix feminine posee et chaleureuse",
    },
    "vieille_femme": {
        "base": {"af_bella": 0.4, "ff_siwis": 0.4, "af_sarah": 0.2},
        "description": "Voix feminine agee, douce et un peu rauque",
        "post_morph": {"breathiness": 0.15, "pitch_shift": -1.5},
    },
    # Hommes
    "jeune_homme": {
        "base": {"am_michael": 0.4, "am_adam": 0.3, "bf_emma": 0.3},
        "description": "Voix masculine jeune, energique",
    },
    "homme_mature": {
        "base": {"am_adam": 0.5, "am_michael": 0.3, "bm_george": 0.2},
        "description": "Voix masculine grave et autoritaire",
    },
    "vieil_homme": {
        "base": {"am_adam": 0.4, "bm_george": 0.3, "am_michael": 0.3},
        "description": "Voix masculine agee, posee",
        "post_morph": {"breathiness": 0.2, "roughness": 0.1, "pitch_shift": -2.0},
    },
    # Enfants
    "enfant": {
        "base": {"af_sky": 0.5, "af_heart": 0.3, "af_nicole": 0.2},
        "description": "Voix d'enfant, aigue et vive",
        "post_morph": {"pitch_shift": 4.0, "formant_shift": 1.15},
    },
    # Narrateurs
    "narrateur_calme": {
        "base": {"ff_siwis": 0.5, "af_bella": 0.3, "am_michael": 0.2},
        "description": "Voix de narrateur calme et posee",
    },
    "narrateur_dramatique": {
        "base": {"am_adam": 0.4, "am_michael": 0.3, "bm_george": 0.3},
        "description": "Voix de narrateur dramatique et profonde",
    },
}


class VoiceDesigner:
    """
    Cree des voix personnalisees pour Kokoro par interpolation de tenseurs.

    Usage:
        designer = VoiceDesigner(kokoro_engine)
        voice = designer.create_from_preset("jeune_femme")
        # ou
        voice = designer.create_blend({"af_bella": 0.6, "am_adam": 0.4})
        # ou
        voice = designer.random_walk("af_bella", variation=0.3)
    """

    def __init__(self, kokoro_model=None):
        """
        Args:
            kokoro_model: Instance de kokoro_onnx.Kokoro (optionnel, lazy load)
        """
        self._kokoro = kokoro_model
        self._style_cache: dict[str, np.ndarray] = {}

    def _ensure_kokoro(self):
        """Charge Kokoro si necessaire."""
        if self._kokoro is not None:
            return

        try:
            from kokoro_onnx import Kokoro
            model_path = Path("kokoro-v1.0.onnx")
            voices_path = Path("voices-v1.0.bin")
            if model_path.exists() and voices_path.exists():
                self._kokoro = Kokoro(str(model_path), str(voices_path))
            else:
                raise FileNotFoundError("Fichiers Kokoro non trouves")
        except (ImportError, FileNotFoundError) as e:
            raise RuntimeError(f"Kokoro non disponible: {e}")

    def _get_style(self, voice_id: str) -> np.ndarray:
        """Charge et cache un tenseur de style."""
        if voice_id in self._style_cache:
            return self._style_cache[voice_id]

        self._ensure_kokoro()
        style = self._kokoro.get_voice_style(voice_id)
        self._style_cache[voice_id] = style
        return style

    def create_blend(self, voices: dict[str, float], name: str = "") -> VoiceDesign:
        """
        Cree une voix par melange pondere.

        Args:
            voices: Dict {voice_id: weight} (poids normalises a 1.0)
            name: Nom de la voix (optionnel)

        Returns:
            VoiceDesign avec le tenseur blende
        """
        # Normaliser les poids
        total = sum(voices.values())
        if total <= 0:
            raise ValueError("Les poids doivent etre positifs")
        normalized = {k: v / total for k, v in voices.items()}

        # Blender les tenseurs
        blended = None
        for voice_id, weight in normalized.items():
            style = self._get_style(voice_id)
            if blended is None:
                blended = style * weight
            else:
                blended = blended + style * weight

        if not name:
            name = "+".join(f"{v}:{w:.0%}" for v, w in normalized.items())

        return VoiceDesign(
            name=name,
            description=f"Blend: {normalized}",
            tensor=blended,
            source_voices=normalized,
        )

    def create_from_preset(self, preset_name: str) -> VoiceDesign:
        """
        Cree une voix a partir d'un preset de personnage.

        Args:
            preset_name: Nom du preset (voir CHARACTER_PRESETS)

        Returns:
            VoiceDesign
        """
        if preset_name not in CHARACTER_PRESETS:
            available = ", ".join(CHARACTER_PRESETS.keys())
            raise ValueError(f"Preset inconnu: {preset_name}. Disponibles: {available}")

        preset = CHARACTER_PRESETS[preset_name]
        design = self.create_blend(preset["base"], name=preset_name)
        design.description = preset["description"]

        return design

    def random_walk(
        self,
        base_voice: str,
        variation: float = 0.2,
        seed: Optional[int] = None,
        name: str = "",
    ) -> VoiceDesign:
        """
        Cree une voix par marche aleatoire dans l'espace d'embedding.

        Ajoute du bruit gaussien au tenseur de base pour creer
        une variation subtile de la voix originale.

        Args:
            base_voice: Voix de base
            variation: Amplitude de la variation (0.0-1.0)
            seed: Graine aleatoire (pour reproductibilite)
            name: Nom de la voix

        Returns:
            VoiceDesign avec voix modifiee
        """
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()

        style = self._get_style(base_voice).copy()

        # Bruit gaussien proportionnel a l'ecart-type du tenseur
        noise_scale = np.std(style) * variation
        noise = rng.randn(*style.shape).astype(np.float32) * noise_scale

        modified = style + noise

        if not name:
            name = f"{base_voice}_v{seed or rng.randint(10000)}"

        return VoiceDesign(
            name=name,
            description=f"Random walk from {base_voice} (variation={variation})",
            tensor=modified,
            source_voices={base_voice: 1.0},
        )

    def interpolate(
        self,
        voice_a: str,
        voice_b: str,
        steps: int = 5,
    ) -> list[VoiceDesign]:
        """
        Cree une serie de voix interpolees entre deux voix.

        Utile pour trouver un "sweet spot" entre deux voix.

        Args:
            voice_a: Premiere voix
            voice_b: Deuxieme voix
            steps: Nombre d'etapes d'interpolation

        Returns:
            Liste de VoiceDesign
        """
        style_a = self._get_style(voice_a)
        style_b = self._get_style(voice_b)

        designs = []
        for i in range(steps):
            t = i / max(1, steps - 1)
            tensor = style_a * (1 - t) + style_b * t
            designs.append(VoiceDesign(
                name=f"{voice_a}_to_{voice_b}_{i}",
                description=f"Interpolation {voice_a}→{voice_b} t={t:.2f}",
                tensor=tensor,
                source_voices={voice_a: 1 - t, voice_b: t},
            ))

        return designs

    def save_voice(self, design: VoiceDesign, output_dir: Path) -> Path:
        """
        Sauvegarde une voix personnalisee.

        Args:
            design: VoiceDesign a sauvegarder
            output_dir: Repertoire de sortie

        Returns:
            Chemin du fichier sauvegarde
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Sauvegarder le tenseur
        tensor_path = output_dir / f"{design.name}.npy"
        np.save(str(tensor_path), design.tensor)

        # Sauvegarder les metadonnees
        meta_path = output_dir / f"{design.name}.json"
        meta = {
            "name": design.name,
            "description": design.description,
            "source_voices": design.source_voices,
            "tensor_shape": list(design.tensor.shape),
            "tensor_dtype": str(design.tensor.dtype),
        }
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

        logger.info(f"Voix '{design.name}' sauvegardee dans {output_dir}")
        return tensor_path

    def load_voice(self, path: Path) -> VoiceDesign:
        """
        Charge une voix personnalisee depuis un fichier.

        Args:
            path: Chemin du fichier .npy

        Returns:
            VoiceDesign
        """
        path = Path(path)
        tensor = np.load(str(path))

        # Charger les metadonnees si disponibles
        meta_path = path.with_suffix(".json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            return VoiceDesign(
                name=meta.get("name", path.stem),
                description=meta.get("description", ""),
                tensor=tensor,
                source_voices=meta.get("source_voices", {}),
            )

        return VoiceDesign(
            name=path.stem,
            description="Loaded from file",
            tensor=tensor,
            source_voices={},
        )

    @staticmethod
    def list_presets():
        """Affiche les presets disponibles."""
        print("\n=== Presets de voix personnalisees ===\n")
        for name, preset in CHARACTER_PRESETS.items():
            voices = ", ".join(f"{v}:{w:.0%}" for v, w in preset["base"].items())
            morph = ""
            if "post_morph" in preset:
                morph = f" [morph: {preset['post_morph']}]"
            print(f"  {name:22}: {preset['description']}")
            print(f"  {'':22}  Blend: {voices}{morph}")
            print()
