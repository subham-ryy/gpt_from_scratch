"""Lightweight system resource and hardware monitor.

Safely collects GPU VRAM and CPU RAM metrics without heavy synchronization.
All monitoring operations are non-fatal to ensure training never crashes.
"""
import os
import platform
import sys
import torch

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


class SystemMonitor:
    """Collects hardware and runtime metrics for experiment tracking."""

    @staticmethod
    def get_hardware_info() -> dict:
        """Capture static environment and hardware metadata."""
        info = {
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "cpu_count": os.cpu_count(),
        }
        if _HAS_PSUTIL:
            try:
                vm = psutil.virtual_memory()
                info["total_ram_gb"] = round(vm.total / (1024 ** 3), 2)
            except Exception:
                info["total_ram_gb"] = None
        return info

    @staticmethod
    def get_current_metrics() -> dict:
        """Capture dynamic GPU and CPU memory metrics. Never raises."""
        metrics = {
            "gpu_allocated_mb": 0.0,
            "gpu_reserved_mb": 0.0,
            "gpu_peak_mb": 0.0,
            "cpu_ram_used_mb": 0.0,
            "cpu_ram_percent": 0.0,
            "process_ram_mb": 0.0,
        }

        # GPU VRAM metrics
        if torch.cuda.is_available():
            try:
                metrics["gpu_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
                metrics["gpu_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024 * 1024), 2)
                metrics["gpu_peak_mb"] = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
            except Exception:
                pass

        # CPU RAM metrics
        if _HAS_PSUTIL:
            try:
                vm = psutil.virtual_memory()
                proc = psutil.Process()
                metrics["cpu_ram_used_mb"] = round(vm.used / (1024 * 1024), 2)
                metrics["cpu_ram_percent"] = round(vm.percent, 1)
                metrics["process_ram_mb"] = round(proc.memory_info().rss / (1024 * 1024), 2)
            except Exception:
                pass

        return metrics

