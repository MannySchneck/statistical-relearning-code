# 4: Geocentric models

Ptolemaic epicycles are equivalent to fourier series! That's really cool. And
it makes sense. Ever smaller-weighted sin/cos functions, smaller circles.

DEF
*Linear Regression* is always wrong, but is useful.

Bayesian linear regression uses a probability interpretation.

Linear regression corresponds to many process models. Recall that process model
!= hypothesis.

## 4.1 On the "normality" of the normal distribution

### 4.1.1 Normal by addition

```
trials <- 1e4
tonorm_uniform <-  function () {
	sums_over_random_center_0 <- replicate(1000, sum(runif(trials, -1, 1)))
	plot(density( sums_over_random_center_0 ))
}

tonorm_squared <- function () {
	sums_over_random_center_0 <- replicate(1000, sum(runif(trials, -1, 1)^2))
	plot(density( sums_over_random_center_0 ))
}

```

Looks like squaring produces a 'thinner-tailed' normal.

So... Summing a symmetric uniform distribution, leads to the normal
distribution when repeated?

So why does this work?

Sums over a distribution, that has a mean, will end up as sums of the mean,
most often. Fluctuations above the mean will cancel out with those below, the
more terms that there are. The peak, the most frequent value, will be the mean.

SO! Any process adding together random values from a distribution converges to
normal (wut).

Note that this doesn't work for distributions that can produce "dominating
outliers", i.e. frequently the mean won't be the converging/telescoping sum
choice.

# 4.1.2 Normal by multiplication

```
small <- replicate(1e4, prod(1 + runif(12, 0, .1)))
dens(small)

big <- replicate(1e4, prod(1 + runif(12, 0, .5)))
dens(big)
```

Error for small-deviant multiplication, approximated by adding just the
multiplicative increase percent is small:

```
1.1 * 1.1 = (1 + .1) (1 + .1) = 1 + .2 + .01 = 1.21 ~= 1.2
```

This falls apart for large deviates

# 4.1.2 Normal by log-multiplication

But!

If we measure our observations on a log-scale, then repeated
multiplicative-effect interactions converge to a gaussian!

```
log.big <- replicate(1000, log(prod(1 + runif(12, 0, 0.5))))
dens(log.big)
```

There we go!

Recall:
```
log(a * b) = log(a) + log(b)
```

Gaussians are effective at modeling sums of fluctuations, which is a lot of
stuff. But Gaussians aren't good at modelling underlying micro-process.

DEF
The *Exponential Family* of distributions, of which Gaussians are a member is a
useful set of distributions for modeling natural processes.

Gaussians have an epistemological advantage: indifference/ignorance. All we
state is variance and mean: it's a two-parameter distribution.

They also have the ontological advantage of fitting a lot of natural processes
(lol).

If a measure has finite variance, the gaussian assumes the least about the
shape of the data. That's the only assumption.

The Gaussian:
```
p(gamma | mu, sigma) =
	1 / squrt(2 pi sigma^2)
	* exp (-(gamma - mu)^2 / (2 * sigma^2))
```

Where
* `gamma` is probability density
* `mu` is the mean
* `sigma` is the standard deviation, whereas `sigma^2` is the variance.

Note the parabola with `(y - mu)^2`.

DEF
A *Probability Mass Function* is a probability distribution with discrete
outcomes, e.g. the binomial distribution, `dbinom`. Denoted `Pr`

DEF
A *Probability Density Function* is a probability distribution with continuous
outcomes. Denoted `p`.

Alternative Gaussian formulation

let `tau = 1 / sigma^2`

```
p(gamma | mu, tau) =
	sqrt(tau / 2pi) exp( - (tau / 2) * (gamma - mu)^2)
```

## 4.2 Language for describing models

1. Identify the variables in the model Observable variables are *data*.
   Unobservable variables are *parameters*.

2. Define each variable in terms of either other variables or in terms of a
   probability distribution. Distributional variables are "leaf nodes of the
   dag".

3. This, the variables and their associated distributions defines the *joint
   generative model*. This can be used both to generate and analyze data.


The challenge is in knowing which variables matter and the relationships
between them, not in the math.

DEF
A *model* is a mapping of one set of variables to another through a probability
distribution.

## 4.2.1 Re-working the globe model
```
W ~ Binomial(N, p)
p ~ Uniform(0, 1)
```

i.e.

The count `W` is distributed binomially, with sample size `N` and probability
`p`.

The prior for `p` is assumed to be uniform between 0 and 1.

DEF
A *Stochastic* relationship of a variable, denoted with `~`:
```
Variable ~ Distribution(other, variables)
```

indicates that no particular value of `Variable` is described by the
`Distribution`, or even known. Some values are more _plausible_ than others,
but many values are possible under the model.

So what does this look like in Bayes' theorem?
```
Pr(p | w, n) = Binomial(w | n, p) * uniform( p | 0, 1)
	/ Integral(Binomial(w | n, p) * uniform(p | 0, 1), dp)
uniform)
```

Notice that substituting in the variables `n`, and `p` is "the data". Then we
get the parameter posterior distribution.
```
w <- 6
n <- 9
p_grid <- seq(from=0, to=1, length.out=100)
posterior <- dbinom(w, n, p_grid) * dunif(p_grid, 0, 1)
posterior <- posterior / sum(posterior)
```

## 4.3 A Gaussian model of height

Mapping the relative plausibilities of values of `mu` and `sigma`, after
observing the data.

The estimate is this posterior distribution of the two parameters. So we end up
with a distribution of possible gaussians! A distribution of distributions!

```
h_i ~ Normal(mu, sigma)
   iid
```

Where `iid` means "independent and identically distributed". This is an
epistemological statement about the model we're using, not an ontological
statement about the world. "This is what we know or are willing to assume", not
"what is".

Parameters are:
```
mu
theta
```

So we need priors to estimate.

```
Pr(mu, sigma) = pr(mu)pr(sigma)
```
Which assumes that `mu` and `sigma` are independent. I.e priors are specified
idependently.

```
h_i ~ Normal(mu, sigma)
mu ~ Normal(178, 20)
sigma ~ uniform(0, 50)
```

These are from "esitmated guesses on height".

```
# Mu prior
curve(dnorm(x, 178, 20), from=100, to=250)
```

This is an assumption about our average, not about individual heights.
Interesting... so we're chaining distributions already. Normal on normal
averages.

```
# Sigma prior
curve(dunif(x, 0, 50), from=-10, to=60)
```

So our `sigma` states that 95% of variation lies within `100`cm of the average.

DEF
*Prior Predictive Simulation*

```
sample_mu <- rnorm(1e4, 178, 20)
sample_sigma <- runif(1e4, 0, 50)

# What the fuck is this???
prior_h <- rnorm(1e4, sample_mu, sample_sigma)
dens(prior_h)
```

This is the distribution of relative plausibilities of different heights,
before adding any data to the model.

Try with a less precise estimate for `mu`:
```
mu <- rnorm(1e4,178,100)
sigma <- runif(1e4,0,50)
prior_h <- rnorm(1e4, mu, sigma)
dens(prior_h)
```

This gives a much wider prior predictive distribution.

These predictions are a little weird.
```
mean(prior_h > 272)
# [1] 0.1831
mean(prior_h < 0 )
# [1] 0.0429
```

The better the prior model, in general, the better things are gonna go. This is
why it's SCIENCE.

Where `h` is a _set_ of heights.
```
Pr(mu, sigma | h) =
  Product_i(Normal(h_i | mu, sigma))
  * Normal(mu | 178, 20)
  * Uniform(sigma | 0, 50)
  /
  Integral(
    Product_i(
      Normal(h_i | mu, sigma)
      * Normal(mu | 178, 20)
      * Uniform(sigma | 0, 50)
    )
  ) d(mu) d(sigma)
```

`mu, sigma` are the dependent variables, and `h` is the data, the observation.

`mu, sigma` are parameter variables, `h` is the data variable.

# 4.3.3 Grid approximation of the posterior distribution

```
mu.list <- seq(from=150, to=160, length.out=100)
sigma.list <- seq(from=7, to=9, length.out=100)
post <- expand.grid(mu=mu.list, sigma=sigma.list)

post$LL <- sapply(1:nrow(post), function(i) sum(
    dnorm(d2$height, post$mu[i], post$sigma[i], log=TRUE)
  ))

post$prod <- post$LL + dnorm(
	post$mu,
	178,
	20,
	log=TRUE
) + dunif(
	post$sigma,
	0,
	50,
	log=TRUE
)

post$prob <- exp(post$prod - max(post$prod))

contour_xyz(post$mu, post$sigma, post$prob)

image_xyz(post$mu, post$sigma, post$prob)

```

```
sample.rows <- sample(1:nrow(post), size=1e4, replace=TRUE, prob=post$prob)
sample.mu <- post$mu[sample.rows]
sample.sigma <- post$sigma[sample.rows]

plot(sample.mu, sample.sigma, cex=1, pch=16, col=col.alpha(rangi2, .1))

dens(sample.mu)
PI(sample.mu)

dens(sample.sigma)
PI(sample.sigma)

```

```

d3 <- sample( d2$height , size=20 )

mu.list <- seq( from=150, to=170 , length.out=200 )
sigma.list <- seq( from=4 , to=20 , length.out=200 )
post2 <- expand.grid( mu=mu.list , sigma=sigma.list )
post2$LL <- sapply( 1:nrow(post2) , function(i)
sum( dnorm( d3 , mean=post2$mu[i] , sd=post2$sigma[i] ,
log=TRUE ) ) )
post2$prod <- post2$LL + dnorm( post2$mu , 178 , 20 , TRUE ) +
dunif( post2$sigma , 0 , 50 , TRUE )
post2$prob <- exp( post2$prod - max(post2$prod) )
sample2.rows <- sample( 1:nrow(post2) , size=1e4 , replace=TRUE ,
prob=post2$prob )
sample2.mu <- post2$mu[ sample2.rows ]
sample2.sigma <- post2$sigma[ sample2.rows ]
plot( sample2.mu , sample2.sigma , cex=0.5 ,
col=col.alpha(rangi2,0.1) ,
xlab="mu" , ylab="sigma" , pch=16 )

dens(sample2.sigma, norm.comp=TRUE)

```

Basically: estimates for sigma are right-biased, since the variance skews
positive (can't be negative).

# 4.3.5 Finding the p;osterior distribution with quap

DEF
The *Quadratic Approximation* hill climbs to the *Maximum A Posteriori*
estimate, and using the curvature there, estimates the shape of the
distribution.

```
library(rethinking)
data(Howell1)
d <- Howell1
d2 <- d[ d$age >= 18 ]
```

```
flist <- alist (
	height ~ dnorm(mu, sigma),
	mu ~ dnorm(178, 20),
	sigma ~ dunif(0, 50)
      )

m4.1 <- quap(flist, data=d2)
precis(m4.1)

```

```
=>
        mean   sd   5.5%  94.5%
mu    154.61 0.41 153.95 155.27
sigma   7.73 0.29   7.27   8.20
```

DEF
These are approximations for each parameter's *marginal distribution*.
That is to say: averaging over all the distributions of `mu` for each
plausibility of `sigma` we get that the plausibility of `mu` is given by a
gaussian of mean=`154.61` sd=`.41`
