"""Family-weighted bootstrap retaining the same annotation classes in each draw."""
import numpy as np
from scipy import sparse
from cersei_embedding_metrics import summarize as original_summary,labels_matrix
from cersei_embedding_organization import SEED

def summarize(values,labels,families,replicates=1000):
    result=original_summary(values,labels,families,replicates)
    result['legacy_family_resampling_ci95']=result['ci95']
    result['uncertainty']='1000 family-level exponential multiplier draws; fixed observed classes and candidate bank'
    y=np.asarray(values,dtype=float);keep=np.isfinite(y)&np.array([bool(x) for x in labels],dtype=bool)
    if not keep.any():return result
    labs=[labels[i] for i in np.flatnonzero(keep)];y=y[keep]
    membership,classes=labels_matrix(labs);_,fi=np.unique(np.asarray(families)[keep],return_inverse=True);nf=int(fi.max())+1
    if nf<2:return result
    row,col=membership.nonzero()
    total=sparse.csr_matrix((y[row],(fi[row],col)),shape=(nf,len(classes)))
    counts=sparse.csr_matrix((np.ones(len(row)),(fi[row],col)),shape=(nf,len(classes)))
    # A shared positive weight for every member of a sampled family retains all
    # observed classes, avoiding the changing-class estimand of dropped classes.
    weights=np.random.default_rng(SEED).exponential(size=(replicates,nf))
    numerator=np.asarray(total.T@weights.T).T;denominator=np.asarray(counts.T@weights.T).T
    stats=(numerator/denominator).mean(1)
    result['ci95']=np.quantile(stats,[.025,.975]).tolist()
    return result
