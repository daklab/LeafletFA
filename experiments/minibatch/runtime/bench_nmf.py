import sys, time, numpy as np, h5py, scipy.sparse as sp
from sklearn.decomposition import NMF, MiniBatchNMF
TRAIN=sys.argv[1]; N_SUB=int(sys.argv[2]) if len(sys.argv)>2 else 0
with h5py.File(TRAIN,"r") as h:
    g=h["layers/junc_ratio"]; ip=g["indptr"][:]
    n=(N_SUB if N_SUB else len(ip)-1)
    ip=ip[:n+1]; a,b=int(ip[0]),int(ip[-1])
    jr=sp.csr_matrix((g["data"][a:b],g["indices"][a:b],ip-ip[0]),
                     shape=(n,int(g.attrs["shape"][1]))).toarray().astype(np.float32)
X=np.clip(np.nan_to_num(jr),0,1); del jr
print(f"matrix {X.shape}  ({X.nbytes/1e9:.1f} GB)",flush=True)
print(f"{'method':16} {'K':>3} {'seconds':>9} {'minutes':>8}",flush=True)
for K in (10,20,50):
    t=time.time(); MiniBatchNMF(n_components=K,init="nndsvda",batch_size=2048,max_iter=100,random_state=0).fit(X); dt=time.time()-t
    print(f"{'MiniBatchNMF':16} {K:>3} {dt:>9.1f} {dt/60:>8.2f}",flush=True)
for K in (10,20,50):
    t=time.time(); NMF(n_components=K,init="nndsvda",max_iter=200,random_state=0).fit(X); dt=time.time()-t
    print(f"{'NMF(full)':16} {K:>3} {dt:>9.1f} {dt/60:>8.2f}",flush=True)
