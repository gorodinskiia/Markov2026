# lamb_capture.py — N=1 and N=2 lion capture simulation, combined plot
import numpy as np
from scipy.special import erf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = 20000
T = 10000
d0 = 10
batch = 2000
t = np.arange(1, T+1)

def run_sim(n_lions, seed):
    rng = np.random.default_rng(seed)
    S_sum = np.zeros(T+1, dtype=np.float64)
    for start in range(0, R, batch):
        n = min(batch, R - start)
        lamb_steps = rng.choice(np.array([-1,1], dtype=np.int8), size=(n,T))
        lion_steps_list = [rng.choice(np.array([-1,1], dtype=np.int8), size=(n,T))
                            for _ in range(n_lions)]

        alive = np.ones((n, T+1), dtype=bool)
        for lion_steps in lion_steps_list:
            diff = (lion_steps - lamb_steps).astype(np.int32)
            D = np.empty((n, T+1), dtype=np.int32)
            D[:,0] = d0
            np.cumsum(diff, axis=1, out=D[:,1:])
            D[:,1:] += d0
            alive &= (D != 0)

        alive_cum = np.cumprod(alive, axis=1, dtype=np.int8)
        S_sum += alive_cum.sum(axis=0)
        del lamb_steps, lion_steps_list, alive, alive_cum
    return S_sum / R

S1 = run_sim(n_lions=1, seed=0)
S2 = run_sim(n_lions=2, seed=1)
np.save('S1.npy', S1)
np.save('S2.npy', S2)

S1_t = S1[1:]
S2_t = S2[1:]
S1sq_t = S1_t**2

mask = (t >= 100) & (t <= 10000)

def fit(y):
    logt = np.log(t[mask]); logy = np.log(y[mask])
    slope, intercept = np.polyfit(logt, logy, 1)
    return -slope, intercept

beta1, ic1 = fit(S1_t)
beta2, ic2 = fit(S2_t)

S1_pred = erf(d0/(2*np.sqrt(t)))  # continuum N=1 prediction, no fitted prefactor

fig, ax = plt.subplots(figsize=(8,6))
idx = np.unique(np.logspace(0, np.log10(T), 400).astype(int))
idx = idx[idx>=1]

ax.loglog(t[idx-1], S1_t[idx-1], '.', ms=3, color='C0', alpha=0.5, label='S1(t) sim')
ax.loglog(t[idx-1], S1_pred[idx-1], '-', color='C0', lw=1.5, label=r'erf($d_0/2\sqrt{t}$)')
ax.loglog(t[mask], np.exp(ic1)*t[mask]**(-beta1), '--', color='C0', lw=1.2, label=f'S1 fit beta1={beta1:.3f}')

ax.loglog(t[idx-1], S2_t[idx-1], '.', ms=3, color='C1', alpha=0.5, label='S2(t) sim')
ax.loglog(t[mask], np.exp(ic2)*t[mask]**(-beta2), '--', color='C1', lw=1.2, label=f'S2 fit beta2={beta2:.3f}')

ax.loglog(t[idx-1], S1sq_t[idx-1], '.', ms=3, color='C2', alpha=0.5, label='S1(t)^2')

ax.set_xlabel('t')
ax.set_ylabel('survival probability')
ax.set_title(f'Lamb capture: N=1 beta1={beta1:.3f}, N=2 beta2={beta2:.3f} (fit 10^2<=t<=10^4)')
ax.legend(fontsize=8)
ax.grid(True, which='both', alpha=0.3)
fig.tight_layout()
fig.savefig('S1_S2_survival.png', dpi=150)

print('beta1 =', beta1)
print('beta2 =', beta2)
print()
print(f"{'t':>8} {'S2(t)':>12} {'S1(t)^2':>12}")
for tt in [100, 1000, 10000]:
    print(f"{tt:>8} {S2[tt]:>12.5f} {S1[tt]**2:>12.5f}")