"""
Configuration unifiée pour le support GPU.

Détection automatique de CUDA, MPS (Apple Silicon), ou CPU fallback.
Compatible avec ONNX Runtime et PyTorch.
"""
from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class GPUConfig:
    """
    Configuration GPU pour les moteurs TTS.

    Attributes:
        use_gpu: Activer le GPU si disponible
        device: Type de device ("auto", "cuda", "mps", "cpu")
        cuda_device_id: ID du GPU CUDA à utiliser
        memory_fraction: Fraction de mémoire GPU à utiliser (0.0-1.0)
        mixed_precision: Utiliser FP16 pour réduire la mémoire
    """
    use_gpu: bool = True
    device: str = "auto"  # "cuda", "cpu", "mps", "auto"
    cuda_device_id: int = 0
    memory_fraction: float = 0.8
    mixed_precision: bool = False

    # Cache pour éviter la détection répétée
    _detected_device: Optional[str] = field(default=None, repr=False)

    def get_device(self) -> str:
        """
        Retourne le device optimal à utiliser.

        Returns:
            "cuda", "mps", ou "cpu"
        """
        if self._detected_device is not None:
            return self._detected_device

        if self.device != "auto":
            self._detected_device = self.device
            return self.device

        if not self.use_gpu:
            self._detected_device = "cpu"
            return "cpu"

        # Détection automatique
        try:
            import torch
            if torch.cuda.is_available():
                self._detected_device = "cuda"
                return "cuda"
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._detected_device = "mps"
                return "mps"
        except ImportError:
            pass

        self._detected_device = "cpu"
        return "cpu"

    def get_torch_device(self) -> "torch.device":
        """
        Retourne un objet torch.device configuré.

        Returns:
            torch.device pour PyTorch
        """
        import torch
        device_str = self.get_device()
        if device_str == "cuda" and self.cuda_device_id > 0:
            return torch.device(f"cuda:{self.cuda_device_id}")
        return torch.device(device_str)

    def get_onnx_providers(self) -> list[str]:
        """
        Retourne les providers ONNX Runtime dans l'ordre de priorité.

        Returns:
            Liste des providers pour onnxruntime.InferenceSession
        """
        device = self.get_device()

        if device == "cuda":
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif device == "mps":
            # CoreML pour Apple Silicon (si disponible)
            return ['CoreMLExecutionProvider', 'CPUExecutionProvider']
        else:
            return ['CPUExecutionProvider']

    def get_onnx_provider_options(self) -> list[dict]:
        """
        Retourne les options pour chaque provider ONNX.

        Returns:
            Liste de dicts d'options alignée avec get_onnx_providers()
        """
        device = self.get_device()
        options = []

        if device == "cuda":
            cuda_options = {
                'device_id': self.cuda_device_id,
            }
            if self.memory_fraction < 1.0:
                cuda_options['gpu_mem_limit'] = int(
                    self.memory_fraction * self._get_cuda_memory()
                )
            options.append(cuda_options)
            options.append({})  # CPU fallback
        elif device == "mps":
            options.append({})  # CoreML
            options.append({})  # CPU fallback
        else:
            options.append({})  # CPU

        return options

    def _get_cuda_memory(self) -> int:
        """Retourne la mémoire totale du GPU CUDA en bytes."""
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(self.cuda_device_id)
                return props.total_memory
        except Exception:
            pass
        return 8 * 1024 * 1024 * 1024  # 8 GB par défaut

    def setup_torch_optimizations(self) -> None:
        """
        Configure les optimisations PyTorch selon la config.
        """
        try:
            import torch

            if self.get_device() == "cuda":
                # Limiter la mémoire CUDA si spécifié
                if self.memory_fraction < 1.0:
                    torch.cuda.set_per_process_memory_fraction(
                        self.memory_fraction,
                        self.cuda_device_id
                    )

                # Activer TF32 pour les GPU Ampere+
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

                # Benchmark cudnn pour optimiser les convolutions
                torch.backends.cudnn.benchmark = True

            if self.mixed_precision and self.get_device() in ("cuda", "mps"):
                # Note: La precision mixte doit être gérée au niveau du modèle
                pass

        except ImportError:
            pass

    def is_gpu_available(self) -> bool:
        """Vérifie si un GPU est disponible."""
        return self.get_device() in ("cuda", "mps")

    def get_info(self) -> dict:
        """
        Retourne des informations sur la configuration GPU.

        Returns:
            Dict avec les infos GPU
        """
        info = {
            "device": self.get_device(),
            "use_gpu": self.use_gpu,
            "cuda_available": False,
            "mps_available": False,
            "gpu_name": None,
            "gpu_memory_gb": None,
        }

        try:
            import torch

            info["cuda_available"] = torch.cuda.is_available()
            info["mps_available"] = (
                hasattr(torch.backends, 'mps') and
                torch.backends.mps.is_available()
            )

            if info["cuda_available"] and self.get_device() == "cuda":
                props = torch.cuda.get_device_properties(self.cuda_device_id)
                info["gpu_name"] = props.name
                info["gpu_memory_gb"] = props.total_memory / (1024**3)

        except ImportError:
            pass

        return info

    @classmethod
    def from_env(cls) -> "GPUConfig":
        """
        Crée une config à partir des variables d'environnement.

        Variables supportées:
        - AUDIOREADER_GPU: "true"/"false"
        - AUDIOREADER_DEVICE: "auto"/"cuda"/"mps"/"cpu"
        - AUDIOREADER_CUDA_DEVICE: "0", "1", etc.
        - AUDIOREADER_GPU_MEMORY: "0.8" (fraction)
        - AUDIOREADER_MIXED_PRECISION: "true"/"false"
        """
        return cls(
            use_gpu=os.environ.get("AUDIOREADER_GPU", "true").lower() == "true",
            device=os.environ.get("AUDIOREADER_DEVICE", "auto"),
            cuda_device_id=int(os.environ.get("AUDIOREADER_CUDA_DEVICE", "0")),
            memory_fraction=float(os.environ.get("AUDIOREADER_GPU_MEMORY", "0.8")),
            mixed_precision=os.environ.get("AUDIOREADER_MIXED_PRECISION", "false").lower() == "true",
        )


# Instance globale par défaut
_default_config: Optional[GPUConfig] = None


def get_gpu_config() -> GPUConfig:
    """
    Retourne la configuration GPU globale.

    Returns:
        Instance GPUConfig singleton
    """
    global _default_config
    if _default_config is None:
        _default_config = GPUConfig.from_env()
    return _default_config


def set_gpu_config(config: GPUConfig) -> None:
    """
    Définit la configuration GPU globale.

    Args:
        config: Nouvelle configuration
    """
    global _default_config
    _default_config = config


if __name__ == "__main__":
    # Test de la configuration GPU
    config = GPUConfig()
    print("=== Configuration GPU ===")
    info = config.get_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    print(f"\n  ONNX Providers: {config.get_onnx_providers()}")
