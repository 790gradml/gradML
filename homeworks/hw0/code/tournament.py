import numpy as np
from scipy.stats import norm

# Feel free to change the parameters below
theta = np.linspace(3, 16, 16)
sigma = np.linspace(1, 2, 16)

# We'll start by getting our CDF solution

# get our top seed player's parameters
top_seed_index = np.argmax(theta)
top_seed_theta = theta[top_seed_index]
top_seed_sigma = sigma[top_seed_index]
# get the fifteen 'difference' random variable parameters
all_other_theta = np.delete(theta, top_seed_index)
all_other_sigma = np.delete(sigma, top_seed_index)
x_mean = all_other_theta - top_seed_theta
x_var = top_seed_sigma**2 + all_other_sigma**2
# x is the array holding the prob. of top seed player
# winning against each of the 15 opponents
x = [norm.cdf(0, loc=i, scale=np.sqrt(j)) for (i, j) in zip(x_mean, x_var)]
ans = np.sum(x) / 15
print(f"Top seed player's chance of winning is {ans}")

# We then run some simulations to see if the proportion of
# wins agree with the solution above
M = int(1e5)
count = 0


def one_simulation(all_other_theta, all_other_sigma):
    # choose a random opponent index
    j = np.random.choice(range(15))
    sj = norm.rvs(all_other_theta[j], all_other_sigma[j])
    top_seed_s = norm.rvs(top_seed_theta, top_seed_sigma)
    if top_seed_s < sj:
        return False
    return True


for i in range(M):
    if one_simulation(all_other_theta, all_other_sigma):
        count += 1
print(f"Top seed player wins {count/M} of the total simulated games.")
