# sim.py — simulate R lambs, one lion each, compute S1(t) and plot it
import numpy as np
from scipy.special import erf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
R = 20000
T = 10000
d0 = 10
batch = 2000

S1_sum = np.zeros(T+1, dtype=np.float64)

for start in range(0, R, batch):
    n = min(batch, R-start)
    lamb_steps = rng.choice(np.array([-1,1], dtype=np.int8), size=(n,T))
    lion_steps = rng.choice(np.array([-1,1], dtype=np.int8), size=(n,T))
    diff = (lion_steps - lamb_steps).astype(np.int32)  # values in {-2,0,2}
    D = np.empty((n,T+1), dtype=np.int32)
    D[:,0] = d0
    np.cumsum(diff, axis=1, out=D[:,1:])
    D[:,1:] += d0
    alive = (D != 0)
    alive_cum = np.cumprod(alive, axis=1, dtype=np.int8)
    S1_sum += alive_cum.sum(axis=0)
    del lamb_steps, lion_steps, diff, D, alive, alive_cum

S1 = S1_sum / R
np.save('S1.npy', S1)

# --- fit and plot ---
t = np.arange(1, T+1)
S1_t = S1[1:]

mask = (t >= 100) & (t <= 10000)
logt = np.log(t[mask])
logS = np.log(S1_t[mask])
slope, intercept = np.polyfit(logt, logS, 1)
beta1 = -slope

S1_pred = erf(d0/(2*np.sqrt(t)))  # no fitted prefactor

fig, ax = plt.subplots(figsize=(7,5))
idx = np.unique(np.logspace(0, np.log10(T), 400).astype(int))
idx = idx[idx>=1]
ax.loglog(t[idx-1], S1_t[idx-1], '.', ms=3, label='simulation', alpha=0.6)
ax.loglog(t[idx-1], S1_pred[idx-1], 'r-', lw=1.5, label=r'$\mathrm{erf}(d_0/2\sqrt{t})$')
ax.loglog(t[mask], np.exp(intercept)*t[mask]**slope, 'k--', lw=1.5, label=f'fit slope={slope:.3f}')
ax.set_xlabel('t')
ax.set_ylabel(r'$S_1(t)$')
ax.set_title(rf'N=1 capture: fitted beta1={beta1:.3f} (fit window 10^2 <= t <= 10^4)')
ax.legend()
ax.grid(True, which='both', alpha=0.3)
fig.tight_layout()
fig.savefig('S1_survival.png', dpi=150)
print('beta1 =', beta1)