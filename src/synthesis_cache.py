"""
Cache intelligent et parallelisation pour la synthese TTS.

Fonctionnalites:
- Cache des segments generes (evite re-generation)
- Parallelisation sur multi-coeurs
- Gestion de la memoire cache
- Statistiques de performance
"""
import hashlib
import json
import time
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Callable, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
import numpy as np


@dataclass
class CacheEntry:
    """Entree de cache."""
    text_hash: str
    voice_id: str
    speed: float
    audio_path: Path
    created_at: float
    text_preview: str  # Premiers 50 caracteres
    duration_seconds: float = 0.0
    file_size_bytes: int = 0


@dataclass
class CacheStats:
    """Statistiques du cache."""
    total_entries: int = 0
    total_size_mb: float = 0.0
    hits: int = 0
    misses: int = 0
    time_saved_seconds: float = 0.0


class SynthesisCache:
    """
    Cache intelligent pour les segments audio generes.

    Evite de regenerer des segments identiques.
    """

    DEFAULT_MAX_SIZE_MB = 1000  # 1 GB par defaut
    DEFAULT_MAX_ENTRIES = 10000

    def __init__(
        self,
        cache_dir: Path = Path(".synthesis_cache"),
        max_size_mb: float = DEFAULT_MAX_SIZE_MB,
        max_entries: int = DEFAULT_MAX_ENTRIES
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_entries = max_entries

        self._index_path = self.cache_dir / "index.json"
        self._index: dict[str, CacheEntry] = {}
        self._stats = CacheStats()
        self._lock = threading.Lock()

        self._load_index()

    def _compute_key(
        self,
        text: str,
        voice_id: str,
        speed: float
    ) -> str:
        """Calcule une cle unique pour un segment."""
        content = f"{text}|{voice_id}|{speed:.2f}"
        return hashlib.md5(content.encode()).hexdigest()

    def _load_index(self):
        """Charge l'index du cache."""
        if self._index_path.exists():
            try:
                with open(self._index_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for key, entry_data in data.get("entries", {}).items():
                    self._index[key] = CacheEntry(
                        text_hash=entry_data["text_hash"],
                        voice_id=entry_data["voice_id"],
                        speed=entry_data["speed"],
                        audio_path=Path(entry_data["audio_path"]),
                        created_at=entry_data["created_at"],
                        text_preview=entry_data.get("text_preview", ""),
                        duration_seconds=entry_data.get("duration_seconds", 0.0),
                        file_size_bytes=entry_data.get("file_size_bytes", 0)
                    )

                self._stats.total_entries = len(self._index)
                self._stats.total_size_mb = sum(
                    e.file_size_bytes for e in self._index.values()
                ) / (1024 * 1024)

            except Exception as e:
                print(f"Erreur chargement index cache: {e}")
                self._index = {}

    def _save_index(self):
        """Sauvegarde l'index du cache."""
        data = {
            "entries": {
                key: {
                    "text_hash": entry.text_hash,
                    "voice_id": entry.voice_id,
                    "speed": entry.speed,
                    "audio_path": str(entry.audio_path),
                    "created_at": entry.created_at,
                    "text_preview": entry.text_preview,
                    "duration_seconds": entry.duration_seconds,
                    "file_size_bytes": entry.file_size_bytes,
                }
                for key, entry in self._index.items()
            },
            "stats": {
                "hits": self._stats.hits,
                "misses": self._stats.misses,
                "time_saved": self._stats.time_saved_seconds,
            }
        }

        with open(self._index_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def get(
        self,
        text: str,
        voice_id: str,
        speed: float
    ) -> Optional[Path]:
        """
        Recupere un segment du cache.

        Returns:
            Chemin vers le fichier audio ou None
        """
        key = self._compute_key(text, voice_id, speed)

        with self._lock:
            if key in self._index:
                entry = self._index[key]
                if entry.audio_path.exists():
                    self._stats.hits += 1
                    return entry.audio_path
                else:
                    # Fichier supprime, retirer de l'index
                    del self._index[key]

            self._stats.misses += 1
            return None

    def put(
        self,
        text: str,
        voice_id: str,
        speed: float,
        audio_data: np.ndarray,
        sample_rate: int = 24000,
        generation_time: float = 0.0
    ) -> Path:
        """
        Ajoute un segment au cache.

        Args:
            text: Texte du segment
            voice_id: ID de la voix
            speed: Vitesse
            audio_data: Donnees audio numpy
            sample_rate: Frequence d'echantillonnage
            generation_time: Temps de generation (pour stats)

        Returns:
            Chemin vers le fichier cache
        """
        import soundfile as sf

        key = self._compute_key(text, voice_id, speed)
        audio_path = self.cache_dir / f"{key}.wav"

        # Sauvegarder l'audio
        sf.write(str(audio_path), audio_data, sample_rate)

        # Calculer les stats
        file_size = audio_path.stat().st_size
        duration = len(audio_data) / sample_rate

        # Creer l'entree
        entry = CacheEntry(
            text_hash=key,
            voice_id=voice_id,
            speed=speed,
            audio_path=audio_path,
            created_at=time.time(),
            text_preview=text[:50],
            duration_seconds=duration,
            file_size_bytes=file_size
        )

        with self._lock:
            self._index[key] = entry
            self._stats.total_entries = len(self._index)
            self._stats.total_size_mb += file_size / (1024 * 1024)
            self._stats.time_saved_seconds += generation_time

            # Nettoyer si necessaire
            self._cleanup_if_needed()

            # Sauvegarder periodiquement
            if len(self._index) % 100 == 0:
                self._save_index()

        return audio_path

    def _cleanup_if_needed(self):
        """Nettoie le cache si necessaire."""
        # Verifier la taille
        total_size = sum(e.file_size_bytes for e in self._index.values())

        if total_size > self.max_size_bytes or len(self._index) > self.max_entries:
            # Trier par date de creation (plus ancien d'abord)
            sorted_entries = sorted(
                self._index.items(),
                key=lambda x: x[1].created_at
            )

            # Supprimer les plus anciens
            to_remove = []
            removed_size = 0
            target_size = self.max_size_bytes * 0.8  # Garder 80%
            target_count = int(self.max_entries * 0.8)

            for key, entry in sorted_entries:
                if (total_size - removed_size > target_size or
                        len(self._index) - len(to_remove) > target_count):
                    to_remove.append(key)
                    removed_size += entry.file_size_bytes

                    # Supprimer le fichier
                    if entry.audio_path.exists():
                        entry.audio_path.unlink()

            # Retirer de l'index
            for key in to_remove:
                del self._index[key]

            print(f"Cache nettoye: {len(to_remove)} entrees supprimees")

    def clear(self):
        """Vide completement le cache."""
        with self._lock:
            for entry in self._index.values():
                if entry.audio_path.exists():
                    entry.audio_path.unlink()

            self._index.clear()
            self._stats = CacheStats()
            self._save_index()

        print("Cache vide")

    def get_stats(self) -> CacheStats:
        """Retourne les statistiques du cache."""
        return self._stats

    def print_stats(self):
        """Affiche les statistiques."""
        print("\n=== Statistiques du Cache ===")
        print(f"  Entrees: {self._stats.total_entries}")
        print(f"  Taille: {self._stats.total_size_mb:.1f} MB")
        print(f"  Hits: {self._stats.hits}")
        print(f"  Misses: {self._stats.misses}")
        if self._stats.hits + self._stats.misses > 0:
            ratio = self._stats.hits / (self._stats.hits + self._stats.misses) * 100
            print(f"  Taux de hit: {ratio:.1f}%")
        print(f"  Temps economise: {self._stats.time_saved_seconds:.1f}s")


class ParallelSynthesizer:
    """
    Generateur TTS parallele.

    Utilise plusieurs threads/processus pour accelerer la synthese.
    """

    def __init__(
        self,
        num_workers: int = None,
        use_processes: bool = False,
        cache: Optional[SynthesisCache] = None
    ):
        """
        Args:
            num_workers: Nombre de workers (defaut: nb CPUs)
            use_processes: Utiliser des processus au lieu de threads
            cache: Cache a utiliser (optionnel)
        """
        self.num_workers = num_workers or max(1, os.cpu_count() - 1)
        self.use_processes = use_processes
        self.cache = cache or SynthesisCache()

        self._executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

    def synthesize_batch(
        self,
        segments: List[dict],
        synthesize_fn: Callable[[dict], np.ndarray],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[np.ndarray]:
        """
        Synthetise un batch de segments en parallele.

        Args:
            segments: Liste de dicts avec 'text', 'voice_id', 'speed'
            synthesize_fn: Fonction de synthese (segment -> audio)
            progress_callback: Callback de progression (done, total)

        Returns:
            Liste des audios generes (dans l'ordre)
        """
        results = [None] * len(segments)
        cached_count = 0
        to_generate = []

        # Verifier le cache d'abord
        for i, seg in enumerate(segments):
            cached = self.cache.get(
                text=seg['text'],
                voice_id=seg['voice_id'],
                speed=seg.get('speed', 1.0)
            )

            if cached:
                # Charger depuis le cache
                import soundfile as sf
                audio, _ = sf.read(str(cached))
                results[i] = audio
                cached_count += 1
            else:
                to_generate.append((i, seg))

        if cached_count > 0:
            print(f"  {cached_count} segments charges depuis le cache")

        if not to_generate:
            return results

        # Generer les segments manquants en parallele
        print(f"  Generation de {len(to_generate)} segments sur {self.num_workers} workers...")

        completed = cached_count

        with self._executor_class(max_workers=self.num_workers) as executor:
            # Soumettre les taches
            future_to_idx = {}
            for idx, seg in to_generate:
                future = executor.submit(synthesize_fn, seg)
                future_to_idx[future] = (idx, seg)

            # Collecter les resultats
            for future in as_completed(future_to_idx):
                idx, seg = future_to_idx[future]

                try:
                    audio = future.result()
                    results[idx] = audio

                    # Ajouter au cache
                    start_time = time.time()
                    self.cache.put(
                        text=seg['text'],
                        voice_id=seg['voice_id'],
                        speed=seg.get('speed', 1.0),
                        audio_data=audio,
                        generation_time=time.time() - start_time
                    )

                except Exception as e:
                    print(f"Erreur generation segment {idx}: {e}")
                    # Silence pour ce segment
                    results[idx] = np.zeros(24000, dtype=np.float32)

                completed += 1
                if progress_callback:
                    progress_callback(completed, len(segments))

        return results


@dataclass
class BatchJob:
    """Job de synthese batch."""
    segments: List[dict]
    output_dir: Path
    voice_id: str
    speed: float = 1.0
    format: str = "wav"


class BatchProcessor:
    """
    Processeur de jobs batch.

    Optimise pour les gros volumes.
    """

    def __init__(
        self,
        num_workers: int = None,
        cache_dir: Path = Path(".synthesis_cache")
    ):
        self.cache = SynthesisCache(cache_dir)
        self.synthesizer = ParallelSynthesizer(
            num_workers=num_workers,
            cache=self.cache
        )

    def process_job(
        self,
        job: BatchJob,
        synthesize_fn: Callable[[dict], np.ndarray],
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[Path]:
        """
        Traite un job batch.

        Returns:
            Liste des chemins des fichiers generes
        """
        import soundfile as sf

        job.output_dir.mkdir(parents=True, exist_ok=True)

        # Preparer les segments
        segments_with_voice = [
            {
                **seg,
                'voice_id': seg.get('voice_id', job.voice_id),
                'speed': seg.get('speed', job.speed),
            }
            for seg in job.segments
        ]

        # Callback adapte
        def internal_progress(done, total):
            if progress_callback:
                seg = segments_with_voice[min(done, total - 1)]
                progress_callback(done, total, seg.get('text', '')[:30])

        # Generer
        audios = self.synthesizer.synthesize_batch(
            segments_with_voice,
            synthesize_fn,
            progress_callback=internal_progress
        )

        # Sauvegarder les fichiers
        output_paths = []
        for i, audio in enumerate(audios):
            output_path = job.output_dir / f"segment_{i:04d}.{job.format}"
            sf.write(str(output_path), audio, 24000)
            output_paths.append(output_path)

        return output_paths

    def get_stats(self) -> CacheStats:
        """Retourne les statistiques."""
        return self.cache.get_stats()


if __name__ == "__main__":
    # Test
    print("=== Test Cache et Parallelisation ===\n")

    cache = SynthesisCache()
    cache.print_stats()

    print("\n--- Test insertion ---")
    test_audio = np.random.randn(24000).astype(np.float32) * 0.1

    path = cache.put(
        text="Ceci est un test",
        voice_id="ff_siwis",
        speed=1.0,
        audio_data=test_audio,
        generation_time=2.5
    )
    print(f"Cache: {path}")

    print("\n--- Test recuperation ---")
    cached = cache.get("Ceci est un test", "ff_siwis", 1.0)
    print(f"Trouve: {cached}")

    cache.print_stats()
