import numpy as np                            # numerical arrays and vectorized math
import matplotlib.pyplot as plt                # plotting

rng = np.random.default_rng()  # real entropy-seeded generator

mu = 1.0                                       # redirection parameter (marginal case)
N_max = 10**4                                  # simulate the process out to this many nodes
R = 10**3                                      # number of independent realizations run in parallel

# C(3) = 1 for every realization
C = np.ones(R, dtype=np.int64)                 # length-R array: current core count for each realization, starting at C(3)=1

# Step from N=3 up to N=N_max-1, updating C(N+1) from C(N).
# At each step, draw ONE length-R Bernoulli vector (vectorized across realizations).
for N in range(3, N_max):                      # loop over each step of network growth (one loop, not nested over realizations)
    p = mu * C / N                      # length-R vector of success probabilities   # per-realization growth probability, since C differs across realizations
    U = rng.random(R)                   # single length-R uniform draw               # one uniform draw per realization, all at once
    C += (U < p).astype(np.int64)       # increment where the Bernoulli succeeded    # add 1 to C wherever the Bernoulli trial succeeded, leave others unchanged

# Now C holds C(N_max) for each of the R realizations
z = C / N_max                                  # convert each realization's final core count into a core fraction

empirical_mean = z.mean()                       # sample mean of z across the R realizations
empirical_sd = z.std()                          # sample standard deviation of z across the R realizations
empirical_cv = empirical_sd / empirical_mean     # empirical coefficient of variation (SD/mean)

theory_mean = 1/3                               # exact mean of the limiting density h(z)=2(1-z)
theory_cv = 1/np.sqrt(2)                        # exact coefficient of variation of h(z)=2(1-z)

print(f"Empirical mean of z       = {empirical_mean:.5f}")     # print measured mean
print(f"Theoretical mean (1/3)    = {theory_mean:.5f}")        # print exact mean for comparison
print()
print(f"Empirical SD/mean (CV)    = {empirical_cv:.5f}")        # print measured CV
print(f"Theoretical SD/mean (CV)  = {theory_cv:.5f}")           # print exact CV for comparison
print()
print(f"Smallest core size seen (min C at N={N_max}) = {C.min()}")   # smallest final core count across all realizations
print(f"Largest core size seen  (max C at N={N_max}) = {C.max()}")   # largest final core count across all realizations

# --- Plot ---
fig, ax = plt.subplots(figsize=(7, 5))          # create a single figure/axes pair
ax.hist(z, bins=40, density=True, alpha=0.6, color='steelblue', label='Empirical (simulated z=C/N)')  # normalized histogram of the R final z values

z_grid = np.linspace(0, 1, 500)                 # fine grid of z values for plotting the true density curve
h_vals = 2 * (1 - z_grid)                       # evaluate the true limiting density h(z)=2(1-z) on the grid
ax.plot(z_grid, h_vals, 'r-', lw=2, label=r'$h(z)=2(1-z)$')   # overlay the true density in red

ax.set_xlabel('z = C/N')                        # x-axis label
ax.set_ylabel('density')                        # y-axis label
ax.set_title(f'Core fraction z=C/N at N={N_max}, R={R} realizations ($\\mu=1$)')   # plot title with parameters
ax.legend()                                     # show the legend (histogram vs curve)

plt.tight_layout()                              # adjust spacing so labels/titles don't overlap
plt.show()                                      # render the figure to screen