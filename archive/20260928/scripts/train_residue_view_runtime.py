#!/usr/bin/env python3
"""Certified residue transport and live timing for the fixed K ablation.

Reuse the original trainer, sampling, and values. A complete immutable-store
finite scan replaces repetitive per-protein finite scans on every access.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import lightning.pytorch as pl
from horizyn.config import load_config
from horizyn.datasets.residue_hdf5 import (
    validate_residue_hdf5_finite, verify_residue_hdf5_finite_certificate,
    residue_finite_certificate_path)
import horizyn.training_io as transport
import train_protein_pooling_fast_io as original


class CertifiedResidues(transport.StoragePrecisionResidues):
    def __init__(self, file_path, **kwargs):
        kwargs['validate_finite_on_access'] = False
        kwargs['allow_uncertified_finite_skip'] = False
        super().__init__(file_path, **kwargs)


def ensure_certificate(path):
    sidecar = residue_finite_certificate_path(path)
    with sidecar.with_suffix('.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not sidecar.exists():
            print(json.dumps(dict(stage='certifying_frozen_residue_values', path=str(path))), flush=True)
            validate_residue_hdf5_finite(path, workers=8, progress_every_chunks=0)
        return verify_residue_hdf5_finite_certificate(path)


def write(path, value):
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def canonicalize_gradient_layout(module):
    """Preserve gradient values while matching each parameter's exact strides.

    A singleton query gets a [1, 1]-strided gradient for a [256, 1]-strided
    parameter. Both are contiguous, so .contiguous() does not fix it, while
    fused AdamW requires matching strides. No optimizer or gradient values change.
    """
    corrected = []
    for name, parameter in module.named_parameters():
        gradient = parameter.grad
        if gradient is not None and gradient.layout == torch.strided and gradient.stride() != parameter.stride():
            value = torch.empty_strided(parameter.shape, parameter.stride(),
                                        dtype=gradient.dtype, device=gradient.device)
            value.copy_(gradient)
            parameter.grad = value
            corrected.append(name)
    return corrected


class RuntimeProgress(pl.Callback):
    def __init__(self, output):
        self.output = output
        self.started = time.monotonic()
        self.last_step_time = None
        self.last_step = 0

    def record(self, trainer, module, stage):
        now = time.monotonic()
        step = int(trainer.global_step)
        elapsed = now - self.started
        row = dict(stage=stage, unix_time=time.time(), epoch=min(int(trainer.current_epoch)+1, trainer.max_epochs),
                   step=step, elapsed_seconds=elapsed, batch_size=512)
        if self.last_step_time is not None and step > self.last_step:
            row['recent_seconds_per_step'] = (now-self.last_step_time)/(step-self.last_step)
        if module.device.type == 'cuda':
            row.update(allocated_mib=torch.cuda.memory_allocated(module.device)/1024**2,
                       reserved_mib=torch.cuda.memory_reserved(module.device)/1024**2,
                       peak_allocated_mib=torch.cuda.max_memory_allocated(module.device)/1024**2)
        write(self.output/'runtime_progress.json',row)
        with (self.output/'runtime_history.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        self.last_step_time, self.last_step = now, step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step == 1 or trainer.global_step % 20 == 0:
            self.record(trainer, pl_module, 'training')

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        corrected = canonicalize_gradient_layout(pl_module)
        if corrected and trainer.global_step == 0:
            write(self.output/'gradient_layout_receipt.json',dict(
                corrected_parameters=corrected, gradient_values_unchanged=True,
                optimizer_unchanged='fused AdamW', reason='singleton-query gradient stride'))

    def on_validation_start(self, trainer, pl_module):
        self.record(trainer, pl_module, 'validation')

    def on_train_end(self, trainer, pl_module):
        self.record(trainer, pl_module, 'training_complete')


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config',required=True)
    parser.add_argument('--io-output-dir',required=True)
    args,_=parser.parse_known_args()
    config=load_config(args.config)
    source=Path(config.data.protein_residue_embeds_path)
    certificate=ensure_certificate(source)
    output=Path(args.io_output_dir).resolve()
    # The original entry point owns creation of the output directory.
    write(output.parent/'runtime_optimization.json',dict(
        source=str(source), certificate=str(certificate),
        certificate_sha256=hashlib.sha256(certificate.read_bytes()).hexdigest(),
        runtime_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_config_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        sampler_model_loss_values_unchanged=True, finite_check='complete certified immutable store'))
    trainer=pl.Trainer

    class InstrumentedTrainer(trainer):
        def __init__(self,**kwargs):
            kwargs['callbacks']=list(kwargs.get('callbacks',[]))+[RuntimeProgress(output)]
            super().__init__(**kwargs)

    transport.StoragePrecisionResidues=CertifiedResidues
    pl.Trainer=InstrumentedTrainer
    original.main()


if __name__=='__main__':
    main()
