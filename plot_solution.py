"""
plot_solution.py -- visualisation for the RBF-FD staggered-node solver.

Drop this next to your notebook and:

    from plot_solution import plot_solution, plot_flux_and_profiles

    plot_solution(nodes_u, centers_p, u[:, k], v[:, k], p[:, k],
                  obstacle=(2.75, 3.25, 0.0, 0.5), L=6.0, H=1.0,
                  title='Re = 100, t = 30')

Two things make this work on your scattered nodes:

  * CONTOURS are drawn with matplotlib.tri.Triangulation directly on the
    scattered points -- no interpolation, so nothing is smeared.  Triangles
    whose centroid falls inside the obstacle are masked out.

  * STREAMLINES cannot use scattered data: streamplot() needs a regular
    grid.  So the velocity is interpolated onto a meshgrid with griddata()
    and the obstacle is then set to NaN.  streamplot terminates a line when
    it hits NaN, which is exactly the behaviour you want at a solid body.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.interpolate import griddata


def _masked_tri(pts, obstacle):
    """Triangulate scattered points, hiding triangles inside the obstacle."""
    tri = mtri.Triangulation(pts[:, 0], pts[:, 1])
    if obstacle is not None:
        x1, x2, y1, y2 = obstacle
        cx = pts[:, 0][tri.triangles].mean(axis=1)
        cy = pts[:, 1][tri.triangles].mean(axis=1)
        tri.set_mask((cx > x1) & (cx < x2) & (cy > y1) & (cy < y2))
    return tri


def _grid(pts, f, obstacle, L, H, nx=900, ny=160):
    """Interpolate a scattered field onto a regular grid, NaN inside the body."""
    xg = np.linspace(0.0, L, nx)
    yg = np.linspace(0.0, H, ny)
    XG, YG = np.meshgrid(xg, yg)                 # shape (ny, nx)
    FG = griddata(pts, f, (XG, YG), method='linear')
    if obstacle is not None:
        x1, x2, y1, y2 = obstacle
        FG[(XG > x1) & (XG < x2) & (YG > y1) & (YG < y2)] = np.nan
    return xg, yg, FG


def _add_body(ax, obstacle):
    if obstacle is None:
        return
    x1, x2, y1, y2 = obstacle
    ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                               facecolor='0.75', edgecolor='k',
                               linewidth=1.0, zorder=5))


def plot_solution(nodes_u, centers_p, u, v, p=None,
                  obstacle=(2.75, 3.25, 0.0, 0.5), L=6.0, H=1.0,
                  title='', xlim=None, density=(4.0, 1.6), fname=None):
    """Four stacked panels: speed, u with the u=0 line, pressure, streamlines."""
    npanel = 4 if p is not None else 3
    fig, ax = plt.subplots(npanel, 1, figsize=(13, 2.9 * npanel))

    tri_u = _masked_tri(nodes_u, obstacle)
    speed = np.hypot(u, v)

    c = ax[0].tricontourf(tri_u, speed, levels=40, cmap='viridis')
    plt.colorbar(c, ax=ax[0], label='|U|')
    ax[0].set_title(f'{title}   —   speed' if title else 'speed')

    lv = np.linspace(min(u.min(), -1e-6), u.max(), 45)
    c = ax[1].tricontourf(tri_u, u, levels=lv, cmap='RdBu_r')
    
    ax[1].tricontour(tri_u, u, levels=[0.0], colors='k', linewidths=1.4)
    plt.colorbar(c, ax=ax[1], label='u')
    ax[1].set_title(f'u   (black line: u = 0, edge of the recirculation)   '
                    f'u_min = {u.min():+.4f}')

    k = 2
    if p is not None:
        tri_p = _masked_tri(centers_p, obstacle)
        c = ax[2].tricontourf(tri_p, p, levels=40, cmap='coolwarm')
        plt.colorbar(c, ax=ax[2], label='p')
        ax[2].set_title('pressure')
        k = 3

    xg, yg, UG = _grid(nodes_u, u, obstacle, L, H)
    _,  _,  VG = _grid(nodes_u, v, obstacle, L, H)
    ax[k].streamplot(xg, yg, UG, VG, density=list(density),
                     color='k', linewidth=0.6, arrowsize=0.7)
    ax[k].set_title('streamlines')
    ax[k].set_xlabel('x')

    for a in ax:
        _add_body(a, obstacle)
        a.set_xlim(xlim if xlim else (0.0, L))
        a.set_ylim(0.0, H)
        a.set_aspect('equal')
        a.set_ylabel('y')

    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=130, bbox_inches='tight')
    return fig, ax


def reattachment(nodes_u, u, obstacle, band=0.05):
    """Last x on the bottom wall downstream of the body where u < 0."""
    x1, x2, y1, y2 = obstacle
    m = (nodes_u[:, 1] < band) & (nodes_u[:, 1] > 0.004) & (nodes_u[:, 0] > x2)
    o = np.argsort(nodes_u[m, 0])
    xb, ub = nodes_u[m, 0][o], u[m][o]
    neg = np.where(ub < 0)[0]
    if len(neg) == 0:
        return None, None
    xr = xb[neg[-1]]
    return xr, (xr - x2) / (y2 - y1)


def plot_flux_and_profiles(nodes_u, u, obstacle=(2.75, 3.25, 0.0, 0.5),
                           L=6.0, H=1.0, stations=(1.0, 2.6, 3.0, 3.6, 5.0,5.5),
                           fname=None):
    """Left: mass flux vs x (a flat line means mass is conserved).
       Right: u profiles at a few stations."""
    x1, x2, y1, y2 = obstacle
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))

    def prof(X, n=400):
        ylo = y2 if (x1 <= X <= x2) else 0.0
        yy = np.linspace(ylo, H, n)
        uu = griddata(nodes_u, u, np.column_stack([np.full(n, X), yy]),
                      method='linear')
        return yy, np.nan_to_num(uu)

    xs = np.linspace(0.02, L - 0.02, 120)
    q = np.array([np.trapezoid(*prof(X)[::-1]) for X in xs])
    ax[0].plot(xs, q, 'b-')
    ax[0].axhline(q[0], color='r', ls='--', label=f'inlet flux {q[0]:.4f}')
    ax[0].axvspan(x1, x2, color='0.85')
    ax[0].set_xlabel('x'); ax[0].set_ylabel(r'$\int u\,dy$'); ax[0].legend()
    ax[0].set_title('mass flux vs x   (variation %.2f%%)'
                    % (100 * (q.max() - q.min()) / q[0]))

    for X in stations:
        yy, uu = prof(X, 200)
        ax[1].plot(uu, yy, label=f'x = {X}')
    ax[1].axvline(0, color='k', lw=0.6)
    ax[1].set_xlabel('u'); ax[1].set_ylabel('y'); ax[1].legend()
    ax[1].set_title('u profiles')

    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=130, bbox_inches='tight')
    return fig, ax
