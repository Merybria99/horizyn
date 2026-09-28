"""Train-only relational distillation from fixed input chemistry descriptors."""
import torch
import torch.nn.functional as F


def reaction_geometry_loss(embeddings, chemistry, available=None):
    """Match off-diagonal reaction neighborhoods, without enzyme/rule labels.

    Call on the batch's distinct training reactions. Both temperatures are
    fixed at 0.1. Missing/zero descriptors are excluded, and teacher features
    are detached. This is an auxiliary training loss, never a scoring change.
    """
    if chemistry.ndim != 2 or chemistry.shape[0] != embeddings.shape[0]:
        raise ValueError("Chemistry must have one descriptor per reaction embedding")
    with torch.autocast(device_type=embeddings.device.type, enabled=False):
        teacher = chemistry.detach().float()
        valid = torch.isfinite(teacher).all(dim=1) & (teacher.norm(dim=1) > 0)
        if available is not None:
            valid &= available.to(device=valid.device, dtype=torch.bool).reshape(-1)
        student = embeddings.float()[valid]
        if student.shape[0] < 3:
            return embeddings.float().sum() * 0.0
        student = F.normalize(student, dim=-1)
        teacher = F.normalize(teacher[valid], dim=-1)
        n = student.shape[0]
        diagonal = torch.eye(n, device=student.device, dtype=torch.bool)
        student_logits = (student @ student.T / 0.1).masked_fill(diagonal, -1e4)
        teacher_logits = (teacher @ teacher.T / 0.1).masked_fill(diagonal, -1e4)
        return F.kl_div(F.log_softmax(student_logits, dim=-1),
                        F.softmax(teacher_logits, dim=-1), reduction="batchmean")
