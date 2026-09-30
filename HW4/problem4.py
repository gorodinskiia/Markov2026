import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# lumped chain, states U,I,M,F,A
P = np.array([[0,.5,.5,0,0],
              [.25,0,0,.5,.25],
              [.75,0,0,0,.25],
              [0,0,0,1,0],
              [0,0,0,0,1]])
cum = np.cumsum(P, axis=1)

def run(x):
    t = 0
    while x < 3:
        x = int(np.searchsorted(cum[x], rng.random(), side='right'))
        t += 1
    return x, t

N = 10**4
res = {}
for x, name in enumerate("UIM"):
    out = [run(x) for _ in range(N)]
    f = np.array([o[0] for o in out])
    t = np.array([o[1] for o in out])
    res[name] = (f, t)

# exact values: (h, g, tauF, tauA)
ex = {'U': (.5, 4, 4, 4), 'I': (.625, 2, 1.8, 7/3), 'M': (.375, 4, 5, 3.4)}

print("x   h_hat  h     | g_hat  g     | tauF_hat tauF  | tauA_hat tauA")
for name in "UIM":
    f, t = res[name]
    fo = f == 3
    print(name, f"  {fo.mean():.4f} {ex[name][0]:.3f} | {t.mean():.3f} {ex[name][1]:.3f} | "
                f"{t[fo].mean():.3f}  {ex[name][2]:.3f} | {t[~fo].mean():.3f}  {ex[name][3]:.3f}")

Q, R = P[:3, :3], P[:3, 3:]
f, t = res['I']
hI = {3: 5/8, 4: 3/8}
nmax = 25
n = np.arange(1, nmax + 1)

fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for k, (fate, lab) in enumerate([(3, 'Folded'), (4, 'Aggregated')]):
    ts = t[f == fate]
    axs[k].hist(ts, bins=np.arange(0.5, nmax + 1.5), density=True, alpha=.5, label='simulation')
    pmf = np.array([(np.linalg.matrix_power(Q, m - 1) @ R)[1, k] for m in n]) / hI[fate]
    axs[k].plot(n, pmf, 'ro', ms=4, label='exact')
    axs[k].set_title(f'{lab} runs, start I')
    axs[k].set_xlabel('T')
    axs[k].legend()
    print(lab, "even-T count:", (ts % 2 == 0).sum(), "n =", len(ts))

plt.tight_layout()
plt.savefig('hist_I.png', dpi=130)  # still saves a copy
plt.show()                          # opens the window