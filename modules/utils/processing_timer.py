import time
import torch


class ProcessingTimer:
    def __init__(self, mode="both", sync=True):
        valid_modes = {"cpu", "gpu", "both"}
        self.mode = mode.lower()
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}.")
        self.sync = sync

        if self.mode in {"gpu", "both"} and not torch.cuda.is_available():
            raise RuntimeError("The current device does not support GPU timing, but the mode is set to 'gpu' or 'both'.")

        self.cpu_start = None
        self.cpu_end = None
        self.gpu_start = None
        self.gpu_end = None

        self.elapsed_cpu = None
        self.elapsed_gpu = None

    def start(self):
        if self.mode in {"cpu", "both"}:
            self.cpu_start = time.time()
        if self.mode in {"gpu", "both"}:
            self.gpu_start = torch.cuda.Event(enable_timing=True)
            self.gpu_end = torch.cuda.Event(enable_timing=True)
            self.gpu_start.record()

    def stop(self):
        if self.mode in {"gpu", "both"}:
            self.gpu_end.record()
            if self.sync:
                torch.cuda.synchronize()
            self.elapsed_gpu = self.gpu_start.elapsed_time(self.gpu_end)
        if self.mode in {"cpu", "both"}:
            self.cpu_end = time.time()
            self.elapsed_cpu = (self.cpu_end - self.cpu_start) * 1000

        return self.get_elapsed()

    def get_elapsed(self):
        if self.mode == "cpu":
            return self.elapsed_cpu, None
        elif self.mode == "gpu":
            return None, self.elapsed_gpu
        else:
            return self.elapsed_cpu, self.elapsed_gpu

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
