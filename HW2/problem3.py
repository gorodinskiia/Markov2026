import numpy as np                          # numerical arrays and vectorized math
import matplotlib.pyplot as plt              # plotting
import time                                  # timing the sampling loop
rng = np.random.default_rng()                # real entropy-seeded random number generator

def sample_gamma21_rejection(N, lam, c):
    # N: number of accepted samples wanted, lam: proposal rate, c: envelope constant
    accepted = np.empty(N)                   # preallocate array to hold accepted X values
    n_accepted = 0                           # running count of how many have been accepted so far
    n_iter = 0                               # running count of total proposals drawn (accepted + rejected)

    start = time.perf_counter()              # start the timer
    batch_size = 200                         # draw this many candidate proposals per loop iteration
    while n_accepted < N:                    # keep going until we have N accepted samples
        U1 = rng.random(batch_size)          # draw batch_size uniforms, used to generate the proposal X
        X = -np.log(U1) / lam                # inversion sample from Exponential(lam): X = -ln(U1)/lam
        U2 = rng.random(batch_size)          # draw a second, independent batch of uniforms for the accept/reject test
        ratio = X * np.exp(-X) / (c * lam * np.exp(-lam * X))   # compute f(X) / (c*g(X)) for each candidate
        accept_mask = U2 < ratio             # boolean array: True where the candidate passes the accept/reject test

        n_iter += batch_size                 # every proposal in this batch counts as an iteration, accepted or not
        accepted_this_batch = X[accept_mask] # pull out just the X values that were accepted this batch

        take = min(len(accepted_this_batch), N - n_accepted)   # don't overshoot: only take as many as still needed
        accepted[n_accepted:n_accepted+take] = accepted_this_batch[:take]  # write the accepted values into the output array
        n_accepted += take                   # update the running accepted count

    elapsed = time.perf_counter() - start    # stop the timer, compute total elapsed time
    return accepted, n_iter, elapsed         # return the samples, total proposals drawn, and time taken


N = 10**4                                    # target number of accepted samples per lambda

results = {}                                 # dictionary to store results, keyed by lambda
for lam, c_theory in [(0.5, 4/np.e), (0.2, 1/(0.16*np.e))]:   # loop over the two (lambda, envelope constant) pairs
    samples, n_iter, elapsed = sample_gamma21_rejection(N, lam, c_theory)   # run the sampler
    empirical_accept_frac = N / n_iter                        # fraction of proposals that were accepted, measured
    theoretical_accept_frac = 1 / c_theory                     # fraction predicted by theory (1/c)
    mean_time_per_accept = elapsed / N                         # average wall-clock time per accepted sample
    results[lam] = dict(samples=samples, n_iter=n_iter, elapsed=elapsed,
                         empirical=empirical_accept_frac, theoretical=theoretical_accept_frac,
                         mean_time=mean_time_per_accept, c=c_theory)   # store everything for this lambda
    print(f"lambda={lam}: c={c_theory:.4f}  iterations={n_iter}  accepted={N}")   # print summary line
    print(f"  empirical acceptance fraction = {empirical_accept_frac:.4f}")        # print measured accept rate
    print(f"  theoretical acceptance fraction (1/c) = {theoretical_accept_frac:.4f}")  # print predicted accept rate
    print(f"  total elapsed = {elapsed*1000:.2f} ms")                              # print total time in milliseconds
    print(f"  mean time per accepted sample = {mean_time_per_accept*1e6:.3f} microseconds")  # print per-sample timing
    print()                                                                        # blank line between lambda blocks

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))   # create a figure with 2 side-by-side subplots
x_grid = np.linspace(0, 12, 500)                  # fine grid of x values for plotting the true density curve
f_vals = x_grid * np.exp(-x_grid)                 # evaluate the true target density f(x) = x*e^{-x} on that grid

for ax, lam in zip(axes, [0.5, 0.2]):             # loop over each subplot and its corresponding lambda
    samples = results[lam]['samples']             # pull the accepted samples for this lambda
    ax.hist(samples, bins=60, density=True, alpha=0.6, color='steelblue', label='Empirical (accepted samples)')  # normalized histogram of samples
    ax.plot(x_grid, f_vals, 'r-', lw=2, label=r'$f(x)=xe^{-x}$')   # overlay the true density curve in red
    ax.set_title(f"$\\lambda={lam}$,  c={results[lam]['c']:.4f}\n"
                 f"empirical accept={results[lam]['empirical']:.4f}, theory 1/c={results[lam]['theoretical']:.4f}")  # title with stats
    ax.set_xlabel('x')                            # x-axis label
    ax.set_ylabel('density')                      # y-axis label
    ax.legend()                                   # show the legend (histogram vs curve)
    ax.set_xlim(0, 12)                             # fix the x-axis range so both subplots are comparable

plt.tight_layout()                                # adjust spacing so titles/labels don't overlap
plt.show()                                        # render the figure to screen