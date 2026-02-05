"""
Profils de configuration prédéfinis pour AudioReader.

Permet de sauvegarder et charger des configurations
complètes pour différents cas d'usage.
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Any
import json

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


@dataclass
class AudioProfile:
    """Profil de configuration audio."""
    name: str
    description: str = ""

    # Moteur et voix
    engine: str = "kokoro"
    voice: str = "ff_siwis"
    speed: float = 1.0
    language: str = "fr"

    # Pipeline HQ
    hq: bool = True
    multivoice: bool = False
    master: bool = True

    # Style de narration
    style: str = "storytelling"

    # Effets
    enable_sound_effects: bool = False
    sound_effects_intensity: float = 0.3
    chapter_jingle: Optional[str] = None
    ambiance: Optional[str] = None

    # Cache et performance
    use_cache: bool = True
    parallel_workers: int = 4

    # Post-processing
    acx_compliance: bool = True
    target_lufs: float = -20.0

    # Métadonnées personnalisées
    custom_settings: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convertit le profil en dictionnaire."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AudioProfile":
        """Crée un profil depuis un dictionnaire."""
        # Filtrer les clés inconnues
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    def save(self, path: Path) -> None:
        """Sauvegarde le profil dans un fichier."""
        if path.suffix == ".json":
            path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
        elif path.suffix == ".toml":
            # Conversion manuelle en TOML (format simple)
            lines = [f'name = "{self.name}"']
            if self.description:
                lines.append(f'description = "{self.description}"')
            lines.append("")

            for key, value in self.to_dict().items():
                if key in ("name", "description", "custom_settings"):
                    continue
                if isinstance(value, bool):
                    lines.append(f"{key} = {str(value).lower()}")
                elif isinstance(value, str):
                    lines.append(f'{key} = "{value}"')
                elif isinstance(value, (int, float)):
                    lines.append(f"{key} = {value}")
                elif value is None:
                    continue  # Skip None values

            if self.custom_settings:
                lines.append("\n[custom_settings]")
                for k, v in self.custom_settings.items():
                    if isinstance(v, str):
                        lines.append(f'{k} = "{v}"')
                    else:
                        lines.append(f"{k} = {v}")

            path.write_text("\n".join(lines))
        else:
            raise ValueError(f"Format non supporté: {path.suffix}")

    @classmethod
    def load(cls, path: Path) -> "AudioProfile":
        """Charge un profil depuis un fichier."""
        content = path.read_text()

        if path.suffix == ".json":
            data = json.loads(content)
        elif path.suffix == ".toml":
            if tomllib is None:
                raise ImportError("tomllib ou tomli requis pour les fichiers TOML")
            data = tomllib.loads(content)
        else:
            raise ValueError(f"Format non supporté: {path.suffix}")

        return cls.from_dict(data)


# Profils prédéfinis
BUILTIN_PROFILES = {
    "podcast": AudioProfile(
        name="podcast",
        description="Configuration optimisée pour les podcasts",
        hq=True,
        master=True,
        style="conversational",
        speed=1.05,
        acx_compliance=False,
        target_lufs=-16.0,  # Standard podcast
    ),
    "audiobook": AudioProfile(
        name="audiobook",
        description="Configuration complète pour audiobooks professionnels",
        hq=True,
        multivoice=True,
        master=True,
        style="storytelling",
        chapter_jingle="chapter_start",
        acx_compliance=True,
    ),
    "dramatic": AudioProfile(
        name="dramatic",
        description="Style dramatique avec effets sonores",
        hq=True,
        multivoice=True,
        master=True,
        style="dramatic",
        enable_sound_effects=True,
        sound_effects_intensity=0.4,
    ),
    "fast": AudioProfile(
        name="fast",
        description="Conversion rapide sans post-processing",
        hq=False,
        master=False,
        style="conversational",
        speed=1.2,
        use_cache=True,
        acx_compliance=False,
    ),
    "documentary": AudioProfile(
        name="documentary",
        description="Style documentaire, neutre et informatif",
        hq=True,
        master=True,
        style="documentary",
        speed=0.95,
    ),
    "intimate": AudioProfile(
        name="intimate",
        description="Style intime pour lectures personnelles",
        hq=True,
        master=True,
        style="intimate",
        speed=0.9,
        ambiance="fireplace",
    ),
    "energetic": AudioProfile(
        name="energetic",
        description="Style énergique et dynamique",
        hq=True,
        master=True,
        style="energetic",
        speed=1.1,
    ),
}


class ProfileManager:
    """Gestionnaire de profils de configuration."""

    def __init__(self, profiles_dir: Optional[Path] = None):
        """
        Initialise le gestionnaire.

        Args:
            profiles_dir: Dossier contenant les profils personnalisés
        """
        self.profiles_dir = profiles_dir or Path.home() / ".audioreader" / "profiles"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self._custom_profiles: dict[str, AudioProfile] = {}
        self._load_custom_profiles()

    def _load_custom_profiles(self) -> None:
        """Charge les profils personnalisés."""
        for path in self.profiles_dir.glob("*.json"):
            try:
                profile = AudioProfile.load(path)
                self._custom_profiles[profile.name] = profile
            except Exception:
                pass

        for path in self.profiles_dir.glob("*.toml"):
            try:
                profile = AudioProfile.load(path)
                self._custom_profiles[profile.name] = profile
            except Exception:
                pass

    def get_profile(self, name: str) -> Optional[AudioProfile]:
        """
        Récupère un profil par son nom.

        Cherche d'abord dans les profils personnalisés, puis dans les builtin.
        """
        if name in self._custom_profiles:
            return self._custom_profiles[name]
        if name in BUILTIN_PROFILES:
            return BUILTIN_PROFILES[name]
        return None

    def list_profiles(self) -> list[dict]:
        """Liste tous les profils disponibles."""
        profiles = []

        # Builtin profiles
        for name, profile in BUILTIN_PROFILES.items():
            profiles.append({
                "name": name,
                "description": profile.description,
                "builtin": True,
            })

        # Custom profiles
        for name, profile in self._custom_profiles.items():
            if name not in BUILTIN_PROFILES:  # Éviter les doublons
                profiles.append({
                    "name": name,
                    "description": profile.description,
                    "builtin": False,
                })

        return profiles

    def save_profile(self, profile: AudioProfile, format: str = "toml") -> Path:
        """
        Sauvegarde un profil personnalisé.

        Args:
            profile: Le profil à sauvegarder
            format: "json" ou "toml"

        Returns:
            Chemin du fichier créé
        """
        path = self.profiles_dir / f"{profile.name}.{format}"
        profile.save(path)
        self._custom_profiles[profile.name] = profile
        return path

    def delete_profile(self, name: str) -> bool:
        """
        Supprime un profil personnalisé.

        Returns:
            True si supprimé
        """
        if name in BUILTIN_PROFILES:
            return False  # Impossible de supprimer un builtin

        if name in self._custom_profiles:
            # Supprimer les fichiers
            for ext in (".json", ".toml"):
                path = self.profiles_dir / f"{name}{ext}"
                if path.exists():
                    path.unlink()
            del self._custom_profiles[name]
            return True

        return False

    def create_from_args(self, name: str, **kwargs) -> AudioProfile:
        """
        Crée un nouveau profil depuis des arguments.

        Args:
            name: Nom du profil
            **kwargs: Arguments du profil

        Returns:
            Le profil créé
        """
        profile = AudioProfile(name=name, **kwargs)
        return profile


def load_profile(name: str) -> Optional[dict]:
    """
    Charge un profil et retourne sa configuration en dict.

    Fonction utilitaire pour intégration avec audio_reader.py.
    """
    manager = ProfileManager()
    profile = manager.get_profile(name)
    if profile:
        return profile.to_dict()
    return None


def apply_profile_to_config(config: dict, profile_name: str) -> dict:
    """
    Applique un profil à une configuration existante.

    Les valeurs du profil écrasent celles de la config.
    """
    profile = load_profile(profile_name)
    if not profile:
        return config

    # Fusionner (profil écrase config)
    merged = config.copy()
    for key, value in profile.items():
        if key not in ("name", "description") and value is not None:
            merged[key] = value

    return merged


if __name__ == "__main__":
    # Test des profils
    print("=== Test Configuration Profiles ===\n")

    print("Profils prédéfinis:")
    for name, profile in BUILTIN_PROFILES.items():
        print(f"  {name}: {profile.description}")

    print("\n--- Test ProfileManager ---")
    manager = ProfileManager()
    profiles = manager.list_profiles()
    print(f"Total profils: {len(profiles)}")

    # Test création d'un profil personnalisé
    custom = manager.create_from_args(
        name="test_custom",
        description="Profil de test",
        hq=True,
        style="dramatic",
        enable_sound_effects=True,
    )
    print(f"\nProfil créé: {custom.name}")
    print(f"  Style: {custom.style}")
    print(f"  Sound effects: {custom.enable_sound_effects}")

    # Test chargement
    loaded = load_profile("audiobook")
    print(f"\nProfil 'audiobook' chargé:")
    print(f"  HQ: {loaded['hq']}")
    print(f"  Multivoice: {loaded['multivoice']}")
    print(f"  Style: {loaded['style']}")
