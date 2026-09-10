import numpy as np                            # numerical arrays and vectorized math

rng = np.random.default_rng()  # real entropy-seeded generator

N = 10**5                                      # number of samples to generate
U = rng.random(N)                              # draw N uniforms on (0,1)
z = 1 - np.sqrt(U)          # inversion: z = H^{-1}(U)   # apply the inverse-CDF formula to convert each U into a sample from h(z)

sample_mean = z.mean()                          # compute the sample mean of the generated z values
sample_var = z.var()                            # compute the sample variance of the generated z values

theory_mean = 1/3                               # exact mean of h(z)=2(1-z), derived analytically
theory_var = 1/18                               # exact variance of h(z)=2(1-z), derived analytically

print(f"Sample mean       = {sample_mean:.5f}")             # print the empirical mean
print(f"Theoretical mean  = {theory_mean:.5f}  (1/3)")      # print the exact mean for comparison
print()
print(f"Sample variance      = {sample_var:.5f}")           # print the empirical variance
print(f"Theoretical variance = {theory_var:.5f}  (1/18)")   # print the exact variance for comparison