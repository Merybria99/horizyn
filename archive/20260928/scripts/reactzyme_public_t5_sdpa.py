"""T5 encoder attention with fused SDPA, preserving unscaled logits and bias.

Only the uncached encoder self-attention path is supported. Decoder, attention
output, and cache requests fail explicitly. The original module parameters and
state-dict keys are unchanged.
"""
from types import MethodType
import torch
import torch.nn.functional as F


def forward(self,hidden_states,mask=None,key_value_states=None,position_bias=None,
            past_key_values=None,output_attentions=False,**kwargs):
    if self.is_decoder or key_value_states is not None or past_key_values is not None or output_attentions:
        raise ValueError('This SDPA implementation supports uncached T5 encoder self-attention only')
    batch,length=hidden_states.shape[:2]
    shape=(batch,length,self.n_heads,self.key_value_proj_dim)
    q=self.q(hidden_states).view(shape).transpose(1,2)
    k=self.k(hidden_states).view(shape).transpose(1,2)
    v=self.v(hidden_states).view(shape).transpose(1,2)
    if position_bias is None:
        if self.has_relative_attention_bias:
            position_bias=self.compute_bias(length,length,device=q.device)
        else:
            position_bias=torch.zeros((1,self.n_heads,length,length),device=q.device,dtype=q.dtype)
            if self.gradient_checkpointing and self.training:position_bias.requires_grad_(True)
        if mask is not None:position_bias=position_bias+mask[:,:,:,:length]
    # T5 deliberately has no sqrt(d) scale. Casting is necessary for the fused
    # BF16 kernel; the native implementation also accumulates its bias into
    # BF16 scores under autocast. Position bias is shared across encoder layers.
    context=F.scaled_dot_product_attention(q,k,v,attn_mask=position_bias.to(q.dtype),
        dropout_p=self.dropout if self.training else 0.,is_causal=False,scale=1.)
    return self.o(context.transpose(1,2).contiguous().view(batch,length,-1)),position_bias


def enable_t5_sdpa(model):
    count=0
    for layer in model.encoder.block:
        attention=layer.layer[0].SelfAttention
        if attention.is_decoder:raise ValueError('Encoder required')
        attention.forward=MethodType(forward,attention);count+=1
    return count


def checkpoint_feed_forward(self,hidden_states):
    if not self.training or not torch.is_grad_enabled():return self._original_forward(hidden_states)
    from torch.utils.checkpoint import checkpoint
    return checkpoint(self._original_forward,hidden_states,use_reentrant=False)


def enable_t5_ffn_checkpointing(model):
    """Recompute only large FFN activations; retain fused attention activations."""
    for layer in model.encoder.block:
        ffn=layer.layer[-1]
        ffn._original_forward=ffn.forward
        ffn.forward=MethodType(checkpoint_feed_forward,ffn)
