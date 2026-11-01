import numpy as np
from pclean.files import require

def matched_threshold(scores,target_accepted):
    """One global threshold, no dev labels. Ties stay accepted (a <= tau)."""
    a=np.asarray(scores,np.float32)
    require(a.ndim==1 and len(a)>0 and np.isfinite(a).all(),'Invalid legal calibration scores')
    require(0<=target_accepted<=len(a),'Invalid calibration acceptance count')
    # Finite threshold just below min permits JSON-safe rejection of all.
    candidates=np.r_[np.nextafter(a.min(),np.float32(-np.inf)),np.unique(a)].astype(np.float32)
    require(np.isfinite(candidates).all(),'Threshold overflow')
    rows=[(abs(int((a<=t).sum())-target_accepted),abs(float(t)),float(t)) for t in candidates]
    _,_,tau=min(rows) # Closest acceptance count, then smallest boundary shift, then smaller tau.
    return dict(threshold=tau,target_accepted=int(target_accepted),actual_accepted=int((a<=tau).sum()),n=len(a),
                rule='known if original_a <= threshold; original scores stay frozen',
                tie_rule='min count error, then abs(threshold), then smaller threshold')


def boundary_values(raw,original,active,tau):
    logits=raw['known_logits'];prob=np.exp(logits-logits.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True)
    labels=np.asarray(active)[prob.argmax(1)].copy();labels[original['a']>tau]=-1
    obj=original['obj'];u=original['u']
    return dict(keep=original['keep'],labels=labels,scores=np.where(labels==-1,obj*u,obj*(1-u)*prob.max(1)))

