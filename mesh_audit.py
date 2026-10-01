import numpy as np
from scipy.spatial import cKDTree, Delaunay
 
def audit_mesh(P, V, H=1.0, k=28,Dn_obs=None,
               Dx_UU=None, Dy_UU=None, DL_UU=None,
               Dx_PU=None, Dy_PU=None, Dx_UP=None, Dy_UP=None,
               Dx_PP=None, Dy_PP=None, DL_PP=None,
               p_sets=None, U_ref=1.5, Re=100.0):
    """P = pressure nodes (vertices), V = velocity nodes (edge midpoints).
       Operators optional: pass them and the deeper checks run too."""
    out=[]
    def chk(name, val, good, warn, fmt='{:.3g}', lower_is_better=True):
        if lower_is_better: s = 'PASS' if val<=good else ('WARN' if val<=warn else 'FAIL')
        else:               s = 'PASS' if val>=good else ('WARN' if val>=warn else 'FAIL')
        out.append((s, name, fmt.format(val)))
        return s
 
    dV = cKDTree(V).query(V,k=2)[0][:,1]
    dP = cKDTree(P).query(P,k=2)[0][:,1]
 
    # --- 1. resolution: can a stencil even fit inside the channel? -----------
    rows = H/np.median(dV)
    chk('rows of V-nodes across the height', rows, 12, 8, '{:.1f}', lower_is_better=False)
 
    tree=cKDTree(V); interior=(V[:,1]>0.2*H)&(V[:,1]<0.8*H)
    span=[]
    for i in np.where(interior)[0][::7]:
        st=V[tree.query(V[i],k=k)[1]]-V[i]
        span.append((st[:,1].max()-st[:,1].min())/H)
    chk('fraction of the height a %d-node stencil spans'%k, float(np.mean(span)), 0.40, 0.60, '{:.2f}')
 
    # --- 2. local quality ---------------------------------------------------
    chk('min/median V spacing (slivers)', dV.min()/np.median(dV), 0.5, 0.35, '{:.2f}', lower_is_better=False)
    chk('min/median P spacing (slivers)', dP.min()/np.median(dP), 0.5, 0.35, '{:.2f}', lower_is_better=False)
    nb = cKDTree(V).query(V,k=7)[1][:,1:]
    grad = np.max(np.abs(np.log(dV[nb]/dV[:,None])),axis=1)
    chk('worst neighbour size jump (grading smoothness)', float(np.exp(np.percentile(grad,99))), 1.35, 1.6, '{:.2f}')
 
    # --- 3. staggering ------------------------------------------------------
    ratio=len(V)/len(P)
    out.append((('PASS' if abs(ratio-3)<=0.5 else 'WARN' if abs(ratio-3)<=1.0 else 'FAIL'),
                'N_V / N_P  (edge midpoints -> ~3; centroids -> ~2)', '{:.2f}'.format(ratio)))
 
    # --- 4. boundary bookkeeping -------------------------------------------
    if p_sets is not None:
        allb=np.concatenate([np.asarray(p_sets[k_]).ravel()
                             for k_ in p_sets if k_ != 'interior'])
        dup = len(allb)-len(np.unique(allb))
        miss = len(P)-len(allb)-len(np.asarray(p_sets['interior']).ravel())
        chk('P-nodes with two BCs', dup, 0, 0, '{:.0f}')
        chk('P-nodes with no equation', abs(miss), 0, 0, '{:.0f}')
 
    # --- 5. operator sanity (polynomial reproduction: exact by construction) --
    if Dx_UU is not None:
        one=np.ones(len(V))
        chk('Dx_UU @ 1        (must be 0)', np.abs(Dx_UU@one).max(), 1e-9, 1e-7, '{:.1e}')
        chk('Dx_UU @ x  - 1   (must be 0)', np.abs(Dx_UU@V[:,0]-1).max(), 1e-9, 1e-7, '{:.1e}')
        chk('DL_UU @ x^2 - 2  (must be 0)', np.abs(DL_UU@(V[:,0]**2)-2).max(), 1e-8, 1e-6, '{:.1e}')
 
    # --- 6. operator accuracy on a resolved smooth field --------------------
    if Dx_UU is not None:
        kk=3.0
        f=lambda q: np.sin(kk*q[:,0])*np.cos(kk*q[:,1])
        fx=lambda q: kk*np.cos(kk*q[:,0])*np.cos(kk*q[:,1])
        fl=lambda q: -2*kk*kk*f(q)
        chk('Dx_UU rel err on sin(3x)cos(3y)', np.abs(Dx_UU@f(V)-fx(V)).max()/np.abs(fx(V)).max(), 1e-3, 1e-2, '{:.1e}')
        chk('DL_UU rel err on sin(3x)cos(3y)', np.abs(DL_UU@f(V)-fl(V)).max()/np.abs(fl(V)).max(), 1e-2, 5e-2, '{:.1e}')
        if DL_PP is not None:
            chk('DL_PP rel err on sin(3x)cos(3y)', np.abs(DL_PP@f(P)-fl(P)).max()/np.abs(fl(P)).max(), 2e-2, 1e-1, '{:.1e}')
 
    # --- 7. weight magnitudes (a bad stencil shows up here first) -----------
    for nm,M in [('Dx_UU',Dx_UU),('DL_UU',DL_UU),('Dx_PU',Dx_PU),('Dx_UP',Dx_UP),('DL_PP',DL_PP)]:
        if M is None: continue
        rs=np.asarray(np.abs(M).sum(axis=1)).ravel()
        chk('%s weight ratio max/median'%nm, rs.max()/np.median(rs), 10, 20, '{:.1f}')
 
    # --- 8. the assembled systems ------------------------------------------
    if DL_PP is not None and p_sets is not None:
        import scipy.sparse as sps
        from scipy.sparse.linalg import splu, norm as spnorm
        s = p_sets; N = len(P)
        interior = np.asarray(s['interior']).ravel()
 
        # every boundary set, with the direction of its outward normal
        bsets = []
        for nm_, dir_ in [('top','y'),('bottom','y'),('inlet','x'),('outlet','x'),
                          ('extra_x','x'),('extra_y','y')]:
            if nm_ in s and len(np.asarray(s[nm_]).ravel()):
                bsets.append((np.asarray(s[nm_]).ravel(), dir_))

        # the consistent (composite) Laplacian actually inverted by the projection
        Lc = (Dx_PU @ Dx_UP + Dy_PU @ Dy_UP).tocsr()

        A = sps.lil_matrix((N, N))
        A[interior, :] = Lc[interior]
        for idx_, dir_ in bsets:
            A[idx_, :] = Dx_PP[idx_] if dir_ == 'x' else Dy_PP[idx_]
        if 'obstacle' in s and Dn_obs is not None:          # <-- the only addition
            A[np.asarray(s['obstacle']).ravel(), :] = Dn_obs
        # Dirichlet p = 0 at the outlet -> non-singular, no Lagrange border
        if 'outlet' in s:
            for kk_ in np.asarray(s['outlet']).ravel():
                A[kk_, :] = 0; A[kk_, kk_] = 1
        A = sps.csc_matrix(A)
 
        # does it factorise, and does it solve?
        try:
            lu = splu(A)
            rng = np.random.default_rng(0)
            b = rng.standard_normal(N); b[np.concatenate([i for i,_ in bsets])] = 0
            x = lu.solve(b)
            resid = np.abs(A @ x - b).max() / max(np.abs(b).max(), 1e-30)
            chk('pressure system solve residual', resid, 1e-9, 1e-6, '{:.1e}')
        except Exception as e:
            out.append(('FAIL', 'pressure system factorisation', str(e)[:40]))
 
        # how far the compact Laplacian is from the one the projection inverts
        d = spnorm(Lc[interior] - DL_PP[interior]) / spnorm(Lc[interior])
        chk('||D.G - DL_PP|| / ||D.G||   (use D.G in the Poisson rows)',
            d, 0.05, 0.20, '{:.3f}')
 
        # boundary-to-boundary coupling in the Neumann rows (should be 0)
        allb = np.concatenate([i for i, _ in bsets])
        bset = set(allb.tolist())
        nbb = 0
        for bi in allb:
            row = Dx_PP.indices[Dx_PP.indptr[bi]:Dx_PP.indptr[bi+1]]
            nbb += sum(1 for c in row if c in bset and c != bi)
        chk('boundary-to-boundary stencil entries (build rows from interior only)',
            nbb, 0, 0, '{:d}')
 
    # --- 9. what dt this mesh admits ---------------------------------------
    dt_cfl = 0.25*dV.min()/U_ref
    out.append(('INFO','recommended dt (CFL 0.25)','{:.4g}'.format(dt_cfl)))
    out.append(('INFO','N_P, N_V','{:d}, {:d}'.format(len(P),len(V))))
    out.append(('INFO','dense V-operator memory','{:.0f} MB each'.format(len(V)**2*8/1e6)))
 
    w=max(len(r[1]) for r in out)
    print(f"{'':6}{'check':<{w}}  value")
    print('-'*(8+w+12))
    for s_,n_,v_ in out:
        mark={'PASS':'  ok  ','WARN':' warn ','FAIL':' FAIL ','INFO':'      '}[s_]
        print(f"{mark}{n_:<{w}}  {v_}")
    nf=sum(1 for r in out if r[0]=='FAIL'); nw=sum(1 for r in out if r[0]=='WARN')
    print('-'*(8+w+12))
    print(f"{nf} FAIL, {nw} WARN  ->  " +
          ("usable" if nf==0 else "fix the FAILs before trusting any result"))