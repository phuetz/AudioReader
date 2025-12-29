"""
Tests pour le module synthesis_cache.py

Teste:
- SynthesisCache
- ParallelSynthesizer
- BatchProcessor
- CacheStats et CacheEntry
"""
import pytest
import numpy as np
import sys
from pathlib import Path
import tempfile
import time
import threading

# Ajouter le repertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.synthesis_cache import (
    CacheEntry,
    CacheStats,
    SynthesisCache,
    ParallelSynthesizer,
    BatchProcessor,
    BatchJob
)


class TestCacheEntry:
    """Tests pour CacheEntry."""

    def test_create_entry(self):
        """Test creation d'une entree."""
        entry = CacheEntry(
            text_hash="abc123",
            voice_id="ff_siwis",
            speed=1.0,
            audio_path=Path("/tmp/audio.wav"),
            created_at=time.time(),
            text_preview="Test texte...",
            duration_seconds=2.5,
            file_size_bytes=120000
        )

        assert entry.text_hash == "abc123"
        assert entry.voice_id == "ff_siwis"
        assert entry.speed == 1.0
        assert entry.duration_seconds == 2.5
        assert entry.file_size_bytes == 120000


class TestCacheStats:
    """Tests pour CacheStats."""

    def test_default_values(self):
        """Test valeurs par defaut."""
        stats = CacheStats()

        assert stats.total_entries == 0
        assert stats.total_size_mb == 0.0
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.time_saved_seconds == 0.0


class TestSynthesisCache:
    """Tests pour SynthesisCache."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Cree un repertoire de cache temporaire."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def cache(self, temp_cache_dir):
        """Cree un cache avec repertoire temporaire."""
        return SynthesisCache(
            cache_dir=temp_cache_dir,
            max_size_mb=10.0,
            max_entries=100
        )

    @pytest.fixture
    def sample_audio(self):
        """Cree un signal audio de test."""
        return np.random.randn(24000).astype(np.float32) * 0.1

    def test_init(self, cache, temp_cache_dir):
        """Test initialisation."""
        assert cache.cache_dir == temp_cache_dir
        assert cache.cache_dir.exists()
        assert (temp_cache_dir / "index.json").exists() or len(cache._index) == 0

    def test_compute_key(self, cache):
        """Test calcul de cle."""
        key1 = cache._compute_key("Hello", "ff_siwis", 1.0)
        key2 = cache._compute_key("Hello", "ff_siwis", 1.0)
        key3 = cache._compute_key("Hello", "ff_siwis", 1.1)
        key4 = cache._compute_key("World", "ff_siwis", 1.0)

        # Memes parametres = meme cle
        assert key1 == key2

        # Parametres differents = cles differentes
        assert key1 != key3
        assert key1 != key4

    def test_get_empty_cache(self, cache):
        """Test get sur cache vide."""
        result = cache.get("Test", "ff_siwis", 1.0)
        assert result is None

    def test_put_and_get(self, cache, sample_audio):
        """Test put puis get."""
        text = "Ceci est un test"
        voice = "ff_siwis"
        speed = 1.0

        # Put
        path = cache.put(text, voice, speed, sample_audio)
        assert path.exists()

        # Get
        result = cache.get(text, voice, speed)
        assert result is not None
        assert result == path

    def test_get_increments_hits(self, cache, sample_audio):
        """Test que get incremente les hits."""
        cache.put("Test", "ff_siwis", 1.0, sample_audio)

        initial_hits = cache._stats.hits
        cache.get("Test", "ff_siwis", 1.0)

        assert cache._stats.hits == initial_hits + 1

    def test_get_miss_increments_misses(self, cache):
        """Test que get incremente les misses."""
        initial_misses = cache._stats.misses
        cache.get("Inexistant", "ff_siwis", 1.0)

        assert cache._stats.misses == initial_misses + 1

    def test_put_updates_stats(self, cache, sample_audio):
        """Test que put met a jour les stats."""
        initial_entries = cache._stats.total_entries

        cache.put("Test1", "ff_siwis", 1.0, sample_audio)
        cache.put("Test2", "ff_siwis", 1.0, sample_audio)

        assert cache._stats.total_entries == initial_entries + 2

    def test_cache_different_parameters(self, cache, sample_audio):
        """Test cache avec parametres differents."""
        # Meme texte, voix differente
        cache.put("Hello", "ff_siwis", 1.0, sample_audio)
        cache.put("Hello", "am_adam", 1.0, sample_audio)

        result1 = cache.get("Hello", "ff_siwis", 1.0)
        result2 = cache.get("Hello", "am_adam", 1.0)

        assert result1 is not None
        assert result2 is not None
        assert result1 != result2

    def test_cache_different_speed(self, cache, sample_audio):
        """Test cache avec vitesses differentes."""
        cache.put("Hello", "ff_siwis", 1.0, sample_audio)
        cache.put("Hello", "ff_siwis", 1.2, sample_audio)

        result1 = cache.get("Hello", "ff_siwis", 1.0)
        result2 = cache.get("Hello", "ff_siwis", 1.2)
        result3 = cache.get("Hello", "ff_siwis", 0.8)  # Pas en cache

        assert result1 is not None
        assert result2 is not None
        assert result3 is None

    def test_clear(self, cache, sample_audio):
        """Test vidage du cache."""
        cache.put("Test1", "ff_siwis", 1.0, sample_audio)
        cache.put("Test2", "ff_siwis", 1.0, sample_audio)

        cache.clear()

        assert cache.get("Test1", "ff_siwis", 1.0) is None
        assert cache.get("Test2", "ff_siwis", 1.0) is None
        assert cache._stats.total_entries == 0

    def test_get_stats(self, cache, sample_audio):
        """Test recuperation des stats."""
        cache.put("Test", "ff_siwis", 1.0, sample_audio)
        cache.get("Test", "ff_siwis", 1.0)
        cache.get("Autre", "ff_siwis", 1.0)

        stats = cache.get_stats()

        assert stats.total_entries == 1
        assert stats.hits == 1
        assert stats.misses == 1

    def test_save_and_load_index(self, temp_cache_dir, sample_audio):
        """Test persistance de l'index."""
        # Premier cache
        cache1 = SynthesisCache(cache_dir=temp_cache_dir)
        cache1.put("Persistent", "ff_siwis", 1.0, sample_audio)
        cache1._save_index()

        # Nouveau cache, meme repertoire
        cache2 = SynthesisCache(cache_dir=temp_cache_dir)

        result = cache2.get("Persistent", "ff_siwis", 1.0)
        assert result is not None

    def test_thread_safety(self, cache, sample_audio):
        """Test acces concurrent."""
        results = []
        errors = []

        def worker(i):
            try:
                cache.put(f"Test{i}", "ff_siwis", 1.0, sample_audio.copy())
                result = cache.get(f"Test{i}", "ff_siwis", 1.0)
                results.append(result is not None)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        assert all(results)


class TestParallelSynthesizer:
    """Tests pour ParallelSynthesizer."""

    @pytest.fixture
    def temp_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def synthesizer(self, temp_cache_dir):
        cache = SynthesisCache(cache_dir=temp_cache_dir)
        return ParallelSynthesizer(num_workers=2, cache=cache)

    def test_init_default_workers(self):
        """Test initialisation avec workers par defaut."""
        synth = ParallelSynthesizer()
        assert synth.num_workers >= 1

    def test_init_custom_workers(self):
        """Test initialisation avec workers personnalises."""
        synth = ParallelSynthesizer(num_workers=4)
        assert synth.num_workers == 4

    def test_synthesize_batch_empty(self, synthesizer):
        """Test batch vide."""
        def dummy_synth(seg):
            return np.zeros(1000, dtype=np.float32)

        results = synthesizer.synthesize_batch([], dummy_synth)
        assert results == []

    def test_synthesize_batch_single(self, synthesizer):
        """Test batch avec un seul element."""
        def dummy_synth(seg):
            return np.ones(1000, dtype=np.float32) * 0.5

        segments = [{"text": "Test", "voice_id": "ff_siwis", "speed": 1.0}]
        results = synthesizer.synthesize_batch(segments, dummy_synth)

        assert len(results) == 1
        assert len(results[0]) == 1000

    def test_synthesize_batch_multiple(self, synthesizer):
        """Test batch avec plusieurs elements."""
        def dummy_synth(seg):
            return np.ones(1000, dtype=np.float32) * 0.5

        segments = [
            {"text": f"Test {i}", "voice_id": "ff_siwis", "speed": 1.0}
            for i in range(5)
        ]
        results = synthesizer.synthesize_batch(segments, dummy_synth)

        assert len(results) == 5
        for r in results:
            assert len(r) == 1000

    def test_synthesize_batch_uses_cache(self, synthesizer):
        """Test que le batch utilise le cache."""
        call_count = [0]

        def counting_synth(seg):
            call_count[0] += 1
            return np.ones(1000, dtype=np.float32) * 0.5

        # Premier appel
        segments = [{"text": "Cached", "voice_id": "ff_siwis", "speed": 1.0}]
        synthesizer.synthesize_batch(segments, counting_synth)
        first_count = call_count[0]

        # Deuxieme appel avec les memes segments
        synthesizer.synthesize_batch(segments, counting_synth)
        second_count = call_count[0]

        # Le deuxieme appel devrait utiliser le cache
        assert second_count == first_count, "La synthese n'a pas utilise le cache"

    def test_synthesize_batch_progress(self, synthesizer):
        """Test callback de progression."""
        progress_calls = []

        def dummy_synth(seg):
            return np.ones(1000, dtype=np.float32)

        def progress(done, total):
            progress_calls.append((done, total))

        segments = [
            {"text": f"Test {i}", "voice_id": "ff_siwis", "speed": 1.0}
            for i in range(3)
        ]

        synthesizer.synthesize_batch(segments, dummy_synth, progress_callback=progress)

        # Devrait avoir des appels de progression
        assert len(progress_calls) > 0

    def test_synthesize_batch_handles_errors(self, synthesizer):
        """Test gestion des erreurs."""
        def failing_synth(seg):
            if seg["text"] == "Fail":
                raise ValueError("Intentional error")
            return np.ones(1000, dtype=np.float32)

        segments = [
            {"text": "OK", "voice_id": "ff_siwis", "speed": 1.0},
            {"text": "Fail", "voice_id": "ff_siwis", "speed": 1.0},
            {"text": "Also OK", "voice_id": "ff_siwis", "speed": 1.0},
        ]

        # Ne devrait pas lever d'exception
        results = synthesizer.synthesize_batch(segments, failing_synth)

        assert len(results) == 3
        # Le segment en erreur devrait avoir du silence
        assert np.allclose(results[1], 0)


class TestBatchProcessor:
    """Tests pour BatchProcessor."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def processor(self, temp_dir):
        return BatchProcessor(
            num_workers=2,
            cache_dir=temp_dir / "cache"
        )

    def test_init(self, processor):
        """Test initialisation."""
        assert processor.cache is not None
        assert processor.synthesizer is not None

    def test_process_job(self, processor, temp_dir):
        """Test traitement d'un job."""
        def dummy_synth(seg):
            return np.ones(24000, dtype=np.float32) * 0.5

        job = BatchJob(
            segments=[
                {"text": "Segment 1"},
                {"text": "Segment 2"},
            ],
            output_dir=temp_dir / "output",
            voice_id="ff_siwis",
            speed=1.0,
            format="wav"
        )

        output_paths = processor.process_job(job, dummy_synth)

        assert len(output_paths) == 2
        for path in output_paths:
            assert path.exists()

    def test_get_stats(self, processor):
        """Test recuperation des stats."""
        stats = processor.get_stats()
        assert isinstance(stats, CacheStats)


class TestCacheCleanup:
    """Tests pour le nettoyage automatique du cache."""

    @pytest.fixture
    def small_cache(self):
        """Cree un cache avec taille limitee."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SynthesisCache(
                cache_dir=Path(tmpdir),
                max_size_mb=0.1,  # 100 KB
                max_entries=5
            )
            yield cache

    def test_cleanup_by_entries(self, small_cache):
        """Test nettoyage par nombre d'entrees."""
        audio = np.random.randn(1000).astype(np.float32)

        # Ajouter plus que le max
        for i in range(10):
            small_cache.put(f"Test{i}", "ff_siwis", 1.0, audio)

        # Devrait avoir nettoye
        assert len(small_cache._index) <= small_cache.max_entries


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
