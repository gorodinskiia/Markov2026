import numpy as np
import matplotlib.pyplot as plt

# Monte Carlo estimate of P(X^2+Y^2 < Z and Z^2 > X*Y)
# for X,Y,Z ~ Uniform[0,1]

rng = np.random.default_rng(0) # random number generator
Nmax = 1000000
X = rng.random(Nmax)
Y = rng.random(Nmax)
Z = rng.random(Nmax)

# indicator for the event, evaluated once on the full pool of samples
hit = (X**2 + Y**2 < Z) & (Z**2 > X * Y)

# cumulative estimate: running mean of "hit" as N grows
cum_hits = np.cumsum(hit)
Ns = np.arange(1, Nmax + 1)
p_hat = cum_hits / Ns

# analytic value
p_true = np.pi / 8
plot_N = np.unique(np.round(np.logspace(1, np.log10(Nmax), 200)).astype(int))

plt.figure(figsize=(8, 6))
plt.semilogx(plot_N, p_hat[plot_N - 1], 'b-', linewidth=1.5, label='Monte Carlo estimate')
plt.axhline(p_true, color='r', linestyle='--', linewidth=2, label=r'analytic value')
plt.xlabel(r'$N$ (sample size)', fontsize=16)
plt.ylabel('estimated probability', fontsize=16)
plt.title(r'Monte Carlo estimate of $P(X^2+Y^2<Z,\ Z^2>XY)$ vs sample size')
plt.legend(loc='best', fontsize=12)
plt.tick_params(labelsize=14)
plt.tight_layout()
plt.show()

print(f"Final MC estimate (N={Nmax}): {p_hat[-1]:.5f}")
print(f"Analytic value: {p_true:.5f}")