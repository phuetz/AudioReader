"""
Reprise apres interruption pour AudioReader.

Sauvegarde l'etat de progression dans un fichier JSON
pour permettre de reprendre la conversion d'un livre
au dernier chapitre traite.
"""
import hashlib
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


CHECKPOINT_FILE = ".audioreader_progress.json"


@dataclass
class Checkpoint:
    """Etat de progression d'une conversion."""
    book_hash: str
    book_path: str
    output_dir: str
    last_completed_chapter: int  # index 0-based du dernier chapitre termine
    total_chapters: int
    engine: str = "kokoro"
    hq: bool = False
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = time.time()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Checkpoint":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ProgressCheckpoint:
    """Gestionnaire de checkpoints pour la reprise apres interruption."""

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        self.checkpoint_dir = checkpoint_dir or Path(".")
        self.checkpoint_path = self.checkpoint_dir / CHECKPOINT_FILE

    @staticmethod
    def compute_book_hash(file_path: Path) -> str:
        """Calcule le hash MD5 du fichier source."""
        h = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def load(self) -> Optional[Checkpoint]:
        """Charge le checkpoint existant."""
        if not self.checkpoint_path.exists():
            return None
        try:
            data = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
            return Checkpoint.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    def save(self, checkpoint: Checkpoint):
        """Sauvegarde le checkpoint."""
        self.checkpoint_path.write_text(
            json.dumps(checkpoint.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def update_chapter(self, chapter_index: int):
        """Met a jour le dernier chapitre complete."""
        cp = self.load()
        if cp:
            cp.last_completed_chapter = chapter_index
            self.save(cp)

    def clear(self):
        """Supprime le checkpoint (conversion terminee)."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()

    def can_resume(self, book_hash: str) -> bool:
        """Verifie si on peut reprendre pour ce livre."""
        cp = self.load()
        if cp is None:
            return False
        return (
            cp.book_hash == book_hash
            and cp.last_completed_chapter < cp.total_chapters - 1
        )

    def get_resume_chapter(self, book_hash: str) -> int:
        """Retourne l'index du prochain chapitre a traiter (0-based)."""
        cp = self.load()
        if cp is None or cp.book_hash != book_hash:
            return 0
        return cp.last_completed_chapter + 1
