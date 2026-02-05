"""
Traitement par lots pour la conversion de plusieurs livres.

Permet de mettre en file d'attente plusieurs conversions avec
différentes configurations et priorités.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, Any
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time
import threading


class JobPriority(Enum):
    """Priorité d'un job de conversion."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


class JobStatus(Enum):
    """Statut d'un job."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchJob:
    """Un job de conversion dans la file d'attente."""
    id: str
    book_path: Path
    config: dict
    priority: JobPriority = JobPriority.NORMAL
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    error_message: Optional[str] = None
    output_dir: Optional[Path] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    def to_dict(self) -> dict:
        """Convertit le job en dictionnaire."""
        return {
            "id": self.id,
            "book_path": str(self.book_path),
            "config": self.config,
            "priority": self.priority.value,
            "status": self.status.value,
            "progress": self.progress,
            "error_message": self.error_message,
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BatchJob":
        """Crée un job depuis un dictionnaire."""
        return cls(
            id=data["id"],
            book_path=Path(data["book_path"]),
            config=data["config"],
            priority=JobPriority(data.get("priority", 1)),
            status=JobStatus(data.get("status", "pending")),
            progress=data.get("progress", 0.0),
            error_message=data.get("error_message"),
            output_dir=Path(data["output_dir"]) if data.get("output_dir") else None,
            created_at=data.get("created_at", time.time()),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
        )


class BatchProcessor:
    """
    Processeur de conversion par lots.

    Gère une file d'attente de jobs avec priorités et
    exécution parallèle optionnelle.
    """

    def __init__(
        self,
        max_concurrent: int = 2,
        state_file: Optional[Path] = None,
    ):
        """
        Initialise le processeur.

        Args:
            max_concurrent: Nombre maximum de conversions simultanées
            state_file: Fichier pour persister l'état de la file
        """
        self.max_concurrent = max_concurrent
        self.state_file = state_file or Path(".batch_state.json")
        self.queue: list[BatchJob] = []
        self._lock = threading.Lock()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._running = False
        self._job_counter = 0

        # Callbacks
        self._on_job_start: Optional[Callable[[BatchJob], None]] = None
        self._on_job_progress: Optional[Callable[[BatchJob, float], None]] = None
        self._on_job_complete: Optional[Callable[[BatchJob], None]] = None
        self._on_job_error: Optional[Callable[[BatchJob, str], None]] = None

        # Charger l'état précédent si disponible
        self._load_state()

    def _generate_job_id(self) -> str:
        """Génère un ID unique pour un job."""
        self._job_counter += 1
        return f"batch_{int(time.time())}_{self._job_counter:04d}"

    def add_job(
        self,
        book_path: Path,
        config: Optional[dict] = None,
        priority: JobPriority = JobPriority.NORMAL,
    ) -> BatchJob:
        """
        Ajoute un job à la file d'attente.

        Args:
            book_path: Chemin vers le livre à convertir
            config: Configuration de conversion
            priority: Priorité du job

        Returns:
            Le job créé
        """
        job = BatchJob(
            id=self._generate_job_id(),
            book_path=book_path,
            config=config or {},
            priority=priority,
        )

        with self._lock:
            self.queue.append(job)
            self._sort_queue()
            self._save_state()

        return job

    def add_jobs_from_directory(
        self,
        directory: Path,
        config: Optional[dict] = None,
        pattern: str = "*.md",
    ) -> list[BatchJob]:
        """
        Ajoute tous les fichiers d'un dossier à la file.

        Args:
            directory: Dossier contenant les livres
            config: Configuration commune
            pattern: Pattern glob pour filtrer les fichiers

        Returns:
            Liste des jobs créés
        """
        jobs = []
        for file_path in sorted(directory.glob(pattern)):
            job = self.add_job(file_path, config)
            jobs.append(job)
        return jobs

    def add_jobs_from_list(
        self,
        job_list: list[dict],
    ) -> list[BatchJob]:
        """
        Ajoute des jobs depuis une liste de configurations.

        Args:
            job_list: Liste de dicts avec "book_path" et optionnellement "config", "priority"

        Returns:
            Liste des jobs créés
        """
        jobs = []
        for item in job_list:
            priority = JobPriority(item.get("priority", 1))
            job = self.add_job(
                book_path=Path(item["book_path"]),
                config=item.get("config"),
                priority=priority,
            )
            jobs.append(job)
        return jobs

    def _sort_queue(self) -> None:
        """Trie la file par priorité (haute d'abord) puis par date."""
        self.queue.sort(
            key=lambda j: (-j.priority.value, j.created_at)
        )

    def get_job(self, job_id: str) -> Optional[BatchJob]:
        """Récupère un job par son ID."""
        with self._lock:
            for job in self.queue:
                if job.id == job_id:
                    return job
        return None

    def cancel_job(self, job_id: str) -> bool:
        """
        Annule un job en attente.

        Returns:
            True si le job a été annulé
        """
        with self._lock:
            for job in self.queue:
                if job.id == job_id and job.status == JobStatus.PENDING:
                    job.status = JobStatus.CANCELLED
                    self._save_state()
                    return True
        return False

    def remove_job(self, job_id: str) -> bool:
        """
        Supprime un job de la file.

        Returns:
            True si le job a été supprimé
        """
        with self._lock:
            for i, job in enumerate(self.queue):
                if job.id == job_id:
                    if job.status not in (JobStatus.PROCESSING,):
                        self.queue.pop(i)
                        self._save_state()
                        return True
        return False

    def clear_completed(self) -> int:
        """
        Supprime tous les jobs terminés de la file.

        Returns:
            Nombre de jobs supprimés
        """
        with self._lock:
            initial_len = len(self.queue)
            self.queue = [
                j for j in self.queue
                if j.status not in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)
            ]
            removed = initial_len - len(self.queue)
            if removed > 0:
                self._save_state()
            return removed

    def get_pending_jobs(self) -> list[BatchJob]:
        """Retourne les jobs en attente."""
        with self._lock:
            return [j for j in self.queue if j.status == JobStatus.PENDING]

    def get_stats(self) -> dict:
        """Retourne les statistiques de la file."""
        with self._lock:
            return {
                "total": len(self.queue),
                "pending": sum(1 for j in self.queue if j.status == JobStatus.PENDING),
                "processing": sum(1 for j in self.queue if j.status == JobStatus.PROCESSING),
                "completed": sum(1 for j in self.queue if j.status == JobStatus.COMPLETED),
                "failed": sum(1 for j in self.queue if j.status == JobStatus.FAILED),
                "cancelled": sum(1 for j in self.queue if j.status == JobStatus.CANCELLED),
            }

    def _process_job(self, job: BatchJob) -> None:
        """Traite un job individuel."""
        job.status = JobStatus.PROCESSING
        job.started_at = time.time()

        if self._on_job_start:
            self._on_job_start(job)

        try:
            # Import dynamique pour éviter les dépendances circulaires
            from audio_reader import process_book

            def progress_callback(progress: float):
                job.progress = progress
                if self._on_job_progress:
                    self._on_job_progress(job, progress)

            # Traiter le livre
            output_dir = process_book(
                input_file=job.book_path,
                config=job.config,
                progress_callback=progress_callback,
            )

            job.output_dir = output_dir
            job.status = JobStatus.COMPLETED
            job.completed_at = time.time()
            job.progress = 100.0

            if self._on_job_complete:
                self._on_job_complete(job)

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.completed_at = time.time()

            if self._on_job_error:
                self._on_job_error(job, str(e))

        finally:
            with self._lock:
                self._save_state()

    def process_all(
        self,
        on_job_start: Optional[Callable[[BatchJob], None]] = None,
        on_job_progress: Optional[Callable[[BatchJob, float], None]] = None,
        on_job_complete: Optional[Callable[[BatchJob], None]] = None,
        on_job_error: Optional[Callable[[BatchJob, str], None]] = None,
    ) -> dict:
        """
        Traite tous les jobs en attente.

        Args:
            on_job_start: Callback quand un job démarre
            on_job_progress: Callback pour la progression
            on_job_complete: Callback quand un job est terminé
            on_job_error: Callback en cas d'erreur

        Returns:
            Statistiques de traitement
        """
        self._on_job_start = on_job_start
        self._on_job_progress = on_job_progress
        self._on_job_complete = on_job_complete
        self._on_job_error = on_job_error
        self._running = True

        pending = self.get_pending_jobs()
        if not pending:
            return {"processed": 0, "success": 0, "failed": 0}

        self._executor = ThreadPoolExecutor(max_workers=self.max_concurrent)
        futures = []

        for job in pending:
            if not self._running:
                break
            future = self._executor.submit(self._process_job, job)
            futures.append(future)

        # Attendre la fin de tous les jobs
        success = 0
        failed = 0
        for future in as_completed(futures):
            try:
                future.result()
            except Exception:
                pass

        # Compter les résultats
        for job in pending:
            if job.status == JobStatus.COMPLETED:
                success += 1
            elif job.status == JobStatus.FAILED:
                failed += 1

        self._executor.shutdown(wait=True)
        self._executor = None
        self._running = False

        return {
            "processed": len(pending),
            "success": success,
            "failed": failed,
        }

    def stop(self) -> None:
        """Arrête le traitement en cours."""
        self._running = False
        if self._executor:
            self._executor.shutdown(wait=False)

    def _save_state(self) -> None:
        """Sauvegarde l'état de la file."""
        try:
            state = {
                "job_counter": self._job_counter,
                "queue": [job.to_dict() for job in self.queue],
            }
            self.state_file.write_text(json.dumps(state, indent=2))
        except Exception:
            pass  # Ignorer les erreurs de sauvegarde

    def _load_state(self) -> None:
        """Charge l'état précédent."""
        if not self.state_file.exists():
            return

        try:
            state = json.loads(self.state_file.read_text())
            self._job_counter = state.get("job_counter", 0)
            self.queue = [
                BatchJob.from_dict(job_data)
                for job_data in state.get("queue", [])
            ]
            # Remettre les jobs "processing" en "pending" (crash recovery)
            for job in self.queue:
                if job.status == JobStatus.PROCESSING:
                    job.status = JobStatus.PENDING
        except Exception:
            pass  # Ignorer les erreurs de chargement


def load_batch_from_file(file_path: Path) -> list[dict]:
    """
    Charge une liste de jobs depuis un fichier JSON ou TOML.

    Format JSON:
    [
        {"book_path": "livre1.md", "config": {"hq": true}},
        {"book_path": "livre2.md", "priority": 2}
    ]

    Format TOML:
    [[jobs]]
    book_path = "livre1.md"
    [jobs.config]
    hq = true

    [[jobs]]
    book_path = "livre2.md"
    priority = 2
    """
    content = file_path.read_text()

    if file_path.suffix == ".json":
        return json.loads(content)

    elif file_path.suffix == ".toml":
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        data = tomllib.loads(content)
        return data.get("jobs", [])

    else:
        raise ValueError(f"Format non supporté: {file_path.suffix}")


if __name__ == "__main__":
    # Test du batch processor
    print("=== Test Batch Processor ===")

    processor = BatchProcessor(max_concurrent=2)

    # Ajouter quelques jobs de test
    job1 = processor.add_job(Path("livre1.md"), {"hq": True}, JobPriority.HIGH)
    job2 = processor.add_job(Path("livre2.md"), {"hq": False}, JobPriority.NORMAL)
    job3 = processor.add_job(Path("livre3.md"), {}, JobPriority.LOW)

    print(f"\nJobs ajoutés:")
    for job in processor.queue:
        print(f"  {job.id}: {job.book_path} (priorité: {job.priority.name})")

    print(f"\nStatistiques: {processor.get_stats()}")

    # Test annulation
    processor.cancel_job(job3.id)
    print(f"Job {job3.id} annulé: {job3.status.value}")
