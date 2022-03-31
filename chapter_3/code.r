p_grid <- seq(from=0, to=1, length.out=1000)
prob_p <- rep(1, 1000)
prob_data <- dbinom(6, size=9, prob=p_grid)
posterior <- prob_data * prob_p
posterior <- posterior / sum(posterior)

set.seed(100)
samples <- sample(p_grid, prob=posterior, size=1e4, replace=TRUE)

# 3E1
sum(samples < .2) / 1e4
# [1] 4e-04

# 3E2
sum(samples > .8) / 1e4
# [1] 0.1116

# 3E3
sum(samples > .2 & samples < .8) / 1.e4
# [1] 0.888

# 3E4
quantile(samples, .2)
# 20%
# 0.5185185

# 3E5
quantile(samples, .8)
#       80%
# 0.7557558

# 3E6
HPDI(samples, prob=.66)
#     |0.66     0.66|
# 0.5085085 0.7737738

# 3E7
PI(samples, prob=.66)
#       17%       83%
# 0.5025025 0.7697698

# 3M1
p_grid <- seq(from=0, to=1, length.out=1000)
p_prior <- rep(1, 1000)
likelihood <- dbinom(8, size=15, prob=p_grid)
posterior <- likelihood * p_prior
posterior <- posterior / sum(posterior)

# 3M2
samples <- sample(p_grid, 1e4, replace=TRUE, prob=posterior)
HPDI(samples, .9)
#      |0.9      0.9|
# 0.3293293 0.7167167

# 3M3
# Weighted by the posterior samples.
w <- rbinom(1e4, size=15, prob=samples)

table(w)[9] / 1e4
#      8
# 0.1483

# 3M4
# So I've got this posterior distribution of `p`.
# That's actually not related to the size of the trial.
# Let's sample binomially, randomly, at this size, with these parameter
# likelihoods.

# Sample the parameter space (done)
samples <- sample(1e4, p_grid, prob=posterior, replace=TRUE)
# Do a random binomial trial for all of those parameters
w <- rbinom(1e4, size=9, prob=samples)
# Events over event space:
sum(w == 6) / 1e4
# [1] 0.1777

# 3M5
## 3M1*
p_grid <- seq(from=0, to=1, length.out=1000)
p_prior <- rep(1, 1000)
p_prior[1:500] = 0
likelihood <- dbinom(8, size=15, prob=p_grid)

posterior <- p_prior * likelihood
posterior <- posterior / sum(posterior)

## 3M2*
samples <- sample(p_grid, 1e4, prob=posterior, replace=TRUE)
HPDI(samples, prob=.9)
#      |0.9      0.9|
# 0.5005005 0.7107107

## 3M3*
w <- rbinom(1e4, size=15, prob=samples)
mean(w == 8)
# [1] 0.1588

## 3M4*
w <- rbinom(1e4, size=15, prob=samples)
mean(w == 6)
# [1] 0.233

# 3M6
# This seems really suspect. But all the solutions I've been finding are
# using the .7 true probability to get the likelihood.
#
# I found a solution that did a complicated (to me lol) R thing with maps and
# declarative coding using tidyverse stuff
# (https://sr2-solutions.wjakethompson.com/bayesian-inference.html). I didn't
# want to blindly copy-paste that, or go that deep into R yet. Another solution
# did guess and check with a function.
# This is my linear-search solution. It agrees with the fancy R one:

assumed_water_percentage <- .7
prior <- rep(1, 1000)
size <- 1
p_grid <- seq(from=0, to=1, length.out=1000)
p_99_interval <- 1
while (p_99_interval > .05) {
	if (size > 10000) {
		break;
	}

	size <- size + 1
	likelihood <- dbinom(floor(size * assumed_water_percentage), size, p_grid)
	posterior <- likelihood * prior
	posterior <- posterior / sum(posterior)
	samples <- sample(p_grid, 1e4, prob=posterior, replace=TRUE)
	interval <- PI(samples, prob=.99)
	p_99_interval <- interval[2] - interval[1]
}

size
# 2169
# I still don't understand why it's reasonable to choose .7 as the binomial
# probability for the simulation?
#
# What is this problem trying to teach? Strategies for simulation?

# 3H1
# Birth probability, we will model as bionmial.
p_grid <- seq(from=0, to=1, length.out=1000)
num_births <- length(birth1) + length(birth2)
num_boys <- (sum(birth1) + sum(birth2))

prior <- rep(1, length(p_grid))
# Binomial distribution on the total number of boys observed out of the number
# of births.
likelihood <- dbinom(num_boys, size=num_births, prob=p_grid)
posterior <- likelihood * prior
posterior <- posterior / sum(posterior)

# Value for `p` probability of male birth, with maximal posterior density.
p_grid[which.max(posterior)]
#[1] 0.5545546

# 3H2
samples <- sample(p_grid, 1e4, prob=posterior, replace=TRUE)

HPDI(prob= .5, samples)
HPDI(prob= .89, samples)
HPDI(prob= .97, samples)
#      |0.5      0.5|
# 0.5305305 0.5775776

#     |0.89     0.89|
# 0.4994995 0.6106106

#     |0.97     0.97|
# 0.4794795 0.6296296

# 3H3
simulated <- rbinom(1e4, 200, prob=samples)

simplhist(simulated, xlab="binomial trials, randomly performed, with probabilities as weighted samples from the posterior distribution")
# Yup, looks good

# 3H4
# Do the same binomial trial, but only one birth, since we're looking at first
# borns.
simulated <- rbinom(1e4, 200, prob=samples)
dens(simulated)

# Looks like the model is skewed male. Mean birth value is .554 on the model,
# vs .51 in birth1. Birth2 is at .6, inidicating that the second birth skews
# male.
# 3H5
# Second births that followed female first borns
# Recall, male = 1, female = 0, so we can identify sequence female -> male by
# counting 1s in the subtraction.
sequence <- birth2 - birth1
female_after_male <- sum(sequence == 1)

# We want to check the independence of the first/second births.

# Ok damn. I wasn't reading literally.
# Get the number of firstborn girls
female_first_borns <- sum(birth1 == 0)

# Still using our sampled posterior for the random binomial trial.

# So what this says: From the count of female first borns, simulate male birth
# binomial trials. I.e. consider each second birth an event in a binomial trial
# of size female_first_borns.
male_second_birth_simulations <- rbinom(1e4, female_first_borns, prob=samples)
#
median(male_second_birth_simulations)
# 27
chainmode(male_second_birth_simulations)
# [1] 27.1417

# Actual
sum((birth2 - birth1) == 1)
# 39

So our model is underestimating male births after female births.
