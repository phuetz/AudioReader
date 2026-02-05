"""
Interface TTS asynchrone pour streaming audio temps réel.

Permet la génération audio en streaming avec buffer pour playback immédiat.
"""
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator, Optional, Callable
import numpy as np


@dataclass
class StreamingConfig:
    """Configuration pour le streaming TTS."""
    chunk_size_ms: int = 300  # Taille des chunks en ms
    buffer_size_sec: float = 2.0  # Taille du buffer en secondes
    sample_rate: int = 24000
    overlap_ms: int = 10  # Overlap pour éviter les clicks


@dataclass
class AudioChunk:
    """Un chunk audio pour le streaming."""
    audio: np.ndarray
    sample_rate: int
    is_final: bool = False
    timestamp_ms: int = 0
    text_segment: str = ""


class AsyncTTSEngine(ABC):
    """
    Interface abstraite pour un moteur TTS asynchrone avec streaming.
    """

    def __init__(self, config: Optional[StreamingConfig] = None):
        self.config = config or StreamingConfig()

    @abstractmethod
    async def synthesize_stream(
        self,
        text: str,
        voice_id: str = "ff_siwis",
        speed: float = 1.0,
    ) -> AsyncGenerator[AudioChunk, None]:
        """
        Génère l'audio en streaming par chunks.

        Args:
            text: Texte à synthétiser
            voice_id: Identifiant de la voix
            speed: Vitesse de parole

        Yields:
            AudioChunk avec l'audio et les métadonnées
        """
        pass

    @abstractmethod
    async def synthesize_async(
        self,
        text: str,
        output_path: Path,
        voice_id: str = "ff_siwis",
        speed: float = 1.0,
    ) -> bool:
        """
        Synthétise de manière asynchrone (non-bloquante).

        Args:
            text: Texte à synthétiser
            output_path: Chemin du fichier de sortie
            voice_id: Identifiant de la voix
            speed: Vitesse de parole

        Returns:
            True si succès
        """
        pass

    async def warmup(self) -> None:
        """Préchauffe le moteur avec un texte court."""
        pass


class StreamingBuffer:
    """
    Buffer thread-safe pour playback audio temps réel.

    Gère la mise en file des chunks audio pour une lecture fluide.
    """

    def __init__(
        self,
        max_buffer_sec: float = 2.0,
        sample_rate: int = 24000,
    ):
        self.max_buffer_sec = max_buffer_sec
        self.sample_rate = sample_rate
        self.queue: asyncio.Queue[Optional[AudioChunk]] = asyncio.Queue()
        self._total_samples = 0
        self._is_complete = False
        self._lock = asyncio.Lock()

    async def write_chunk(self, chunk: AudioChunk) -> None:
        """
        Ajoute un chunk au buffer.

        Args:
            chunk: Le chunk audio à ajouter
        """
        async with self._lock:
            self._total_samples += len(chunk.audio)
            if chunk.is_final:
                self._is_complete = True

        await self.queue.put(chunk)

    async def read_chunk(self, timeout: float = 1.0) -> Optional[AudioChunk]:
        """
        Lit le prochain chunk du buffer.

        Args:
            timeout: Timeout en secondes

        Returns:
            AudioChunk ou None si timeout/fin
        """
        try:
            chunk = await asyncio.wait_for(
                self.queue.get(),
                timeout=timeout
            )
            return chunk
        except asyncio.TimeoutError:
            return None

    async def mark_complete(self) -> None:
        """Marque le stream comme terminé."""
        await self.queue.put(None)

    @property
    def buffered_duration_sec(self) -> float:
        """Durée audio actuellement en buffer."""
        return self._total_samples / self.sample_rate

    @property
    def is_complete(self) -> bool:
        """Indique si le stream est terminé."""
        return self._is_complete

    def reset(self) -> None:
        """Réinitialise le buffer."""
        self._total_samples = 0
        self._is_complete = False
        # Vider la queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break


class AsyncKokoroWrapper(AsyncTTSEngine):
    """
    Wrapper asynchrone pour le moteur Kokoro TTS.

    Transforme l'API synchrone de Kokoro en interface async/streaming.
    """

    def __init__(
        self,
        config: Optional[StreamingConfig] = None,
        kokoro_engine: Optional["KokoroEngine"] = None,
    ):
        super().__init__(config)
        self._engine = kokoro_engine
        self._executor = None  # ThreadPoolExecutor pour le wrapping sync->async

    def _get_engine(self):
        """Lazy loading du moteur Kokoro."""
        if self._engine is None:
            from src.tts_kokoro_engine import KokoroEngine
            self._engine = KokoroEngine()
        return self._engine

    def _get_executor(self):
        """Lazy loading de l'executor."""
        if self._executor is None:
            from concurrent.futures import ThreadPoolExecutor
            self._executor = ThreadPoolExecutor(max_workers=2)
        return self._executor

    async def synthesize_stream(
        self,
        text: str,
        voice_id: str = "ff_siwis",
        speed: float = 1.0,
    ) -> AsyncGenerator[AudioChunk, None]:
        """
        Génère l'audio en streaming par phrases.

        Le texte est découpé en phrases et chaque phrase est générée
        indépendamment pour permettre le streaming.
        """
        import re

        # Découper le texte en phrases
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return

        engine = self._get_engine()
        executor = self._get_executor()
        loop = asyncio.get_event_loop()

        timestamp_ms = 0

        for i, sentence in enumerate(sentences):
            is_final = (i == len(sentences) - 1)

            # Synthèse dans un thread pour ne pas bloquer
            audio = await loop.run_in_executor(
                executor,
                lambda s=sentence: engine.synthesize(s, voice=voice_id, speed=speed)
            )

            if audio is not None and len(audio) > 0:
                chunk = AudioChunk(
                    audio=audio,
                    sample_rate=self.config.sample_rate,
                    is_final=is_final,
                    timestamp_ms=timestamp_ms,
                    text_segment=sentence,
                )

                # Mettre à jour le timestamp
                duration_ms = int(len(audio) / self.config.sample_rate * 1000)
                timestamp_ms += duration_ms

                yield chunk

    async def synthesize_async(
        self,
        text: str,
        output_path: Path,
        voice_id: str = "ff_siwis",
        speed: float = 1.0,
    ) -> bool:
        """
        Synthétise de manière asynchrone.
        """
        engine = self._get_engine()
        executor = self._get_executor()
        loop = asyncio.get_event_loop()

        try:
            success = await loop.run_in_executor(
                executor,
                lambda: engine.synthesize_chapter(text, str(output_path), voice=voice_id, speed=speed)
            )
            return success
        except Exception:
            return False

    async def warmup(self) -> None:
        """Préchauffe le moteur."""
        engine = self._get_engine()
        # Synthétiser un texte court pour charger le modèle
        async for _ in self.synthesize_stream("Test.", voice_id="ff_siwis"):
            pass


class StreamingPlaybackManager:
    """
    Gère le playback audio en streaming.

    Coordonne le buffer, la synthèse et la lecture.
    """

    def __init__(
        self,
        engine: AsyncTTSEngine,
        buffer: Optional[StreamingBuffer] = None,
    ):
        self.engine = engine
        self.buffer = buffer or StreamingBuffer(
            max_buffer_sec=engine.config.buffer_size_sec,
            sample_rate=engine.config.sample_rate,
        )
        self._synthesis_task: Optional[asyncio.Task] = None
        self._playback_task: Optional[asyncio.Task] = None
        self._is_playing = False
        self._on_chunk_callback: Optional[Callable[[AudioChunk], None]] = None

    def set_on_chunk_callback(self, callback: Callable[[AudioChunk], None]) -> None:
        """Définit un callback appelé pour chaque chunk généré."""
        self._on_chunk_callback = callback

    async def start_streaming(
        self,
        text: str,
        voice_id: str = "ff_siwis",
        speed: float = 1.0,
    ) -> None:
        """
        Démarre le streaming audio.

        Args:
            text: Texte à synthétiser
            voice_id: Identifiant de la voix
            speed: Vitesse de parole
        """
        self.buffer.reset()
        self._is_playing = True

        # Tâche de synthèse
        self._synthesis_task = asyncio.create_task(
            self._synthesis_loop(text, voice_id, speed)
        )

    async def _synthesis_loop(
        self,
        text: str,
        voice_id: str,
        speed: float,
    ) -> None:
        """Boucle de synthèse qui remplit le buffer."""
        try:
            async for chunk in self.engine.synthesize_stream(text, voice_id, speed):
                if not self._is_playing:
                    break

                await self.buffer.write_chunk(chunk)

                if self._on_chunk_callback:
                    self._on_chunk_callback(chunk)

            await self.buffer.mark_complete()
        except Exception as e:
            await self.buffer.mark_complete()
            raise

    async def get_next_chunk(self, timeout: float = 1.0) -> Optional[AudioChunk]:
        """
        Récupère le prochain chunk audio.

        Args:
            timeout: Timeout en secondes

        Returns:
            AudioChunk ou None si terminé/timeout
        """
        return await self.buffer.read_chunk(timeout)

    async def stop(self) -> None:
        """Arrête le streaming."""
        self._is_playing = False

        if self._synthesis_task and not self._synthesis_task.done():
            self._synthesis_task.cancel()
            try:
                await self._synthesis_task
            except asyncio.CancelledError:
                pass

    @property
    def is_complete(self) -> bool:
        """Indique si la synthèse est terminée."""
        return self.buffer.is_complete

    @property
    def is_playing(self) -> bool:
        """Indique si le streaming est actif."""
        return self._is_playing


async def stream_to_file(
    engine: AsyncTTSEngine,
    text: str,
    output_path: Path,
    voice_id: str = "ff_siwis",
    speed: float = 1.0,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> bool:
    """
    Utilitaire pour streamer l'audio vers un fichier.

    Args:
        engine: Moteur TTS async
        text: Texte à synthétiser
        output_path: Chemin de sortie
        voice_id: Identifiant de la voix
        speed: Vitesse de parole
        progress_callback: Callback(current_ms, total_estimated_ms)

    Returns:
        True si succès
    """
    import soundfile as sf

    all_chunks: list[np.ndarray] = []
    total_ms = 0

    async for chunk in engine.synthesize_stream(text, voice_id, speed):
        all_chunks.append(chunk.audio)
        total_ms += int(len(chunk.audio) / chunk.sample_rate * 1000)

        if progress_callback:
            # Estimation grossière: ~150 mots/min = ~2.5 mots/sec
            words = len(text.split())
            estimated_total_ms = int(words / 2.5 * 1000)
            progress_callback(total_ms, max(total_ms, estimated_total_ms))

    if not all_chunks:
        return False

    # Concaténer tous les chunks
    audio = np.concatenate(all_chunks)

    # Sauvegarder
    try:
        sf.write(str(output_path), audio, engine.config.sample_rate)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    # Test du module
    import asyncio

    async def test():
        print("=== Test TTS Async ===")

        # Test du buffer
        buffer = StreamingBuffer(max_buffer_sec=2.0, sample_rate=24000)
        print(f"Buffer créé: max={buffer.max_buffer_sec}s")

        # Test chunk
        audio = np.random.randn(24000).astype(np.float32)  # 1 seconde
        chunk = AudioChunk(
            audio=audio,
            sample_rate=24000,
            is_final=False,
            timestamp_ms=0,
            text_segment="Test"
        )
        print(f"Chunk créé: {len(chunk.audio)} samples, {chunk.sample_rate}Hz")

        # Test write/read
        await buffer.write_chunk(chunk)
        print(f"Buffer duration: {buffer.buffered_duration_sec:.2f}s")

        read_chunk = await buffer.read_chunk(timeout=0.1)
        print(f"Chunk lu: {len(read_chunk.audio) if read_chunk else 0} samples")

        print("\n=== Test complet ===")

    asyncio.run(test())
