"""Ordered CPU prefetch for exact FP32 residue export; no model changes."""
import torch
from torch.utils.data import DataLoader, Dataset
from horizyn.utils.collate import residue_collate_fn


class OrderedResidues(Dataset):
    def __init__(self,source,keys):
        self.source=source;self.keys=keys

    def __len__(self):return len(self.keys)

    def __getitem__(self,index):
        key=self.keys[index]
        return dict(self.source[key],target_id=key)


def initialize_worker(_):
    torch.set_num_threads(1)


@torch.inference_mode()
def batches(module,dataset,keys,device,batch_size=128,workers=4):
    """Yield contiguous original-order batches, preserving padding and dtype."""
    dtype=next((p.dtype for p in module.model.parameters() if p.is_floating_point()),torch.float32)
    options=dict(batch_size=batch_size,shuffle=False,drop_last=False,num_workers=workers,
        collate_fn=residue_collate_fn,pin_memory=True)
    if workers:
        # Workers perform CPU HDF5 reads only. Fork shares the immutable, large
        # sequence-key index; spawn serializes it independently for every worker.
        # StoragePrecisionResidues reopens inherited HDF5 handles by process ID.
        options.update(multiprocessing_context='fork',prefetch_factor=1,worker_init_fn=initialize_worker)
    loader=DataLoader(OrderedResidues(dataset,keys),**options)
    start=0
    for batch in loader:
        residues=batch['residue_embeddings'].to(device=device,dtype=dtype,non_blocking=True)
        mask=batch['residue_padding_mask'].to(device=device,non_blocking=True)
        encoded=module.model.encode_targets(residues,residue_padding_mask=mask,
            score_residue_embeddings=None,score_residue_padding_mask=None,
            capability_vectors=None,capability_mask=None,text_vectors=None,text_mask=None,
            retrieval_direction='reaction_to_enzyme').detach()
        if encoded.ndim!=2 or encoded.shape[0]!=len(residues) or not bool(torch.isfinite(encoded).all()):
            raise ValueError('Invalid ordered residue export')
        yield start,encoded.float().cpu()
        start+=len(residues)
    if start!=len(keys):raise ValueError('Export omitted candidates')
