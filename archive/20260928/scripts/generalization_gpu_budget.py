"""CPU-only GPU memory polling that tolerates transient driver timeouts."""
import json
import subprocess
import time


def free_memory_mib(physical):
    while True:
        try:
            value=subprocess.check_output(['nvidia-smi','--id',str(physical),
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=15)
            return int(value.strip())
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as exc:
            print(json.dumps(dict(gpu_memory_poll_retry=str(physical),error=type(exc).__name__)),flush=True)
            time.sleep(5)
