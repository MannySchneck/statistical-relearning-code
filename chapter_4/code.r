# 4E1

# The likelihood is y_i ~ Normal(mu, sigma)

# 4E2

# There are 2 parameters

# 4E3

# Bayes' Theorem:
# p(a | b) = p(b | a) p(a) / p(b)

# Model:
# y_i ~ Normal(mu, sigma)
# mu ~ Normal(0, 10)
# sigma ~ Exponential(1)

# pr(y_i | mu, sigma) = Normal(y_i | mu, sigma)
# pr(mu) = Normal(mu | 0, 10)
# pr(sigma) = Exponential(sigma | 1)

# So:

# pr(mu, sigma | y_i ) = (pr(y_i | mu, sigma) * pr(mu) * pr(sigma))
#                        / ( integral(pr(y_i | mu, sigma)
#                          * pr(mu) * pr(sigma))d y )

# pr(mu, sigma | y_i ) = (Normal(y | mu, sigma)
#                       * Normal(mu | 0, 10) *
#                       * Exp(sigma | 1))
#            / integral(Normal(y | mu, sigma)
#             * Normal(0, 10)
#             * Exp(1)) dmu dsigma

# Yeah I don't understand this.

# 4E4

# The linear model is u_i = a + b * x_i

# 4E5

# 3 parameters, a, b, sigma in the posterior.
# The mean is defined by a and b, as such isn't a parameter.

# 4M1

# Model:
# y_i ~ Normal(mu, sigma)
# mu ~ Normal(0, 10)
# sigma ~ Exponential(1)

# Recall: The prior predictive simulation is plugging in the prior parameter
# samples into the *likelihood estimator*.

# So:

# Likelihood estimator

size <- 1e4

likelihood <- function(mu_samples, sigma_samples) {
	 rnorm(size, mu_samples, sigma_samples)
}

mu_samples <- rnorm(size, 0, 10)
sigma_samples <- rexp(size, 1)

prior_predictive_samples <- likelihood(mu_samples, sigma_samples)
dens(prior_predictive_samples)

# Note that we end up sampling through layers when doing Posterior pred sim.

# 4M4
# y_i ~ dnorm(mu, sigma)
# mu ~ dnorm(0, 10)
# sigma ~ dexp(1)

# 4M3
# y ~ Normal(mu, sigma)
# mu = a + b*x
# a ~ Normal(0, 10)
# b ~ Uniform(0, 1),
# sigma ~ Exponential(1)

# 4M4
# Measure student heights for 3 years. Linear regression: height vs year.
# h_i ~ dnorm(mu_i, sigma)
# mu_i = a + b * ( year_i - year_bar)
# a <- dnorm(100, 20)
# b <- dunif(5, 20)
# sigma <- dunif(0, 15)

# prior-pd:
plot(NULL, xlim=c(0, 3), ylim=c(50, 250), xlab="year", ylab="height")
N <- 100
year_bar <- 2
a_sample <- rnorm(N, 100, 20)
b_sample <- log_normal(N, 5, 20)
sigma_sample <- runif(N, 0, 8)

for (i in 1:N) curve(
	a[i] + b[i] * (x - year_bar),
	from=0,
	to=3,
	add=TRUE,
	col=col.alpha('black', .2)
)

# 4M7
library(rethinking)
data(Howell1); d <- Howell1; d2 <- d[ d$age >= 18, ]

m4.3_revised <- quap(
  alist(
    # likelihood
    height ~ dnorm(mu, sigma),
    # Linear predictor
    mu <- alpha + beta * ( weight),
    # priors
    alpha ~ dnorm(178, 20),
    beta ~ dlnorm(0, 1),
    sigma ~ dunif(0, 50)
  ), data=d2
)

# > vcov(m4.3_revised)
#              alpha          beta         sigma
# alpha  3.601407563 -0.0784370308  0.0093616751
# beta  -0.078437031  0.0017437116 -0.0002043995
# sigma  0.009361675 -0.0002043995  0.0365751541


# > vcov(m4.3)
#               alpha          beta         sigma
# alpha  7.306632e-02 -4.238715e-08  6.151697e-05
# beta  -4.238715e-08  1.757593e-03 -2.517741e-05
# sigma  6.151697e-05 -2.517741e-05  3.654026e-02



analyze <- m4_7analyze(model) {
	weight.seq <- seq(from=25, to=70, by=1)
	mu_revised <- link(model, data=data.frame(weight=weight.seq))
	mu_revised.mean <- apply(mu_revised, 2, mean)
	mu_revised.pi <- apply(mu_revised, 2, PI, prob=.89)


	heights <- sim(model, data=list(weight=weight.seq))
	heights.pi <- apply(heights, 2, PI, prob=.89)

	plot(
		d2$height ~ d2$weight,
		xlim=range(d2$weight),
		col=col.alpha(rangi2, .8)
	)

	lines(weight.seq, mu_revised.mean)
	shade(mu_revised.pi, weight.seq)
	shade(heights.pi, weight.seq)
}
