import numpy as np                            # numerical arrays and vectorized math
import matplotlib.pyplot as plt                # plotting

rng = np.random.default_rng()  # real entropy-seeded generator

a = 0.9                                        # mixing weight: probability of the fast component
lam_f = 1000.0   # 1/s                         # rate of the fast (brief) closed state
lam_s = 10.0     # 1/s                         # rate of the slow (long) closed state

N = 10**5                                      # number of dwell times to sample

# --- Composition sampling: one Bernoulli draw + one inverse-transform draw ---
B = rng.random(N) < a                     # Bernoulli(a): True with prob a -> fast component
U = rng.random(N)                              # uniforms used for the inverse-transform step
T = np.where(B, -np.log(U) / lam_f, -np.log(U) / lam_s)   # pick fast or slow inversion formula per sample, based on B

# --- Empirical statistics ---
empirical_mean = T.mean()                      # sample mean of the N dwell times
empirical_P_gt_50ms = np.mean(T > 0.050)        # fraction of samples exceeding 50 ms

# --- Theoretical values ---
theory_mean = a / lam_f + (1 - a) / lam_s                              # exact mixture mean, a/λf + (1-a)/λs
theory_P_gt_50ms = a * np.exp(-lam_f * 0.050) + (1 - a) * np.exp(-lam_s * 0.050)   # exact tail probability at 50 ms

print(f"Empirical mean E[T]      = {empirical_mean*1000:.4f} ms")      # print sample mean in ms
print(f"Theoretical mean E[T]    = {theory_mean*1000:.4f} ms")         # print exact mean in ms
print()
print(f"Empirical  P(T > 50 ms)  = {empirical_P_gt_50ms:.5f}")          # print measured tail probability
print(f"Theoretical P(T > 50 ms) = {theory_P_gt_50ms:.5f}")             # print exact tail probability

# --- Plot: normalized histogram on semilog-y axes with f(t) overlaid ---
fig, ax = plt.subplots(figsize=(7, 5))         # create a single figure/axes pair

# Fine linear bins over a range that captures both the fast (~1ms) decay
# near t=0 and the slow (~100ms) tail, so both components are visible on log-y.
t_max_plot = 0.30                              # cap the plotted time range at 300 ms
bins = np.linspace(0, t_max_plot, 300)          # 300 evenly spaced linear bins from 0 to t_max_plot
ax.hist(T[T <= t_max_plot], bins=bins, density=True, alpha=0.6, color='steelblue',
        label='Empirical (composition samples)')   # normalized histogram of samples within plotting range

t_grid = np.linspace(1e-6, t_max_plot, 5000)    # fine grid of t values for plotting the true density curve
f_vals = a * lam_f * np.exp(-lam_f * t_grid) + (1 - a) * lam_s * np.exp(-lam_s * t_grid)   # evaluate true mixture density f(t) on the grid
ax.plot(t_grid, f_vals, 'r-', lw=2, label=r'$f(t)=a\lambda_f e^{-\lambda_f t}+(1-a)\lambda_s e^{-\lambda_s t}$')  # overlay true density in red

ax.set_yscale('log')                           # log scale on the y-axis, so both components are visible together
ax.set_xlabel('t (s)')                          # x-axis label
ax.set_ylabel('density (log scale)')            # y-axis label
ax.set_title('Closed dwell time mixture: composition sampling vs theory')   # plot title
ax.legend()                                     # show the legend (histogram vs curve)
ax.set_ylim(bottom=1e-2)                        # clip the lower y-limit so the log scale doesn't blow up near zero

plt.tight_layout()                              # adjust spacing so labels/titles don't overlap
plt.show()                                      # render the figure to screen