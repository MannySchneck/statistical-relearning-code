---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    formats: md
    main_language: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.6
---

# 2 Small Worlds and Large Worlds

## Intro
DEF
The *Small World* is the world of the model.

DEF
The *Large World* is the world we live in. We are modeling in the small world.
To Understand this large world. To gain control over it. To Intervene.
Causality is a statement about cohort matching re the subset of the Large world
that we assume is the small world. Lol.


Large world relevance must be demonstrated, neither assumed nor deduced.

Intuition is heuristic. Bayesianism is expensive computationally.

IDEA:
CONSIDER:
Intuition as the stable reduction from the cross-basis of the eigenvectors of
multiple models as bases.
What would it mean to train a model by minimizing that error?
*Khaneman THINKING TYPE 1*

## 2.1 The Garden of Forking Data

Bayesian Foundations.

TO READ: Garden of forking paths, Jorge Luis Borges

Bayesian: Quantitative ranking of hypothesis on which branches. Very Feynman
QM.

## 2.1.1: A review of basic probability tilting Bayesian

The usual marble in a bag.

SYSTEM
Four marbles. Colored Blue and white.

There are 5 possibilities on order-independent coloring. This is because we can
represent that as a single partition that has 5 positions, left of all, and
then 4 increments to be rightmost. This is the color divider.

This is a logically consistent small-world model for how many there are.

Now. Byesian probability Probability.

Let
- O=white
- X=colored

OBSERVATION
Draw from bag: XOX

ASSUME: marble has one blue and three white, marbles.
[Example of the gorgious diagrams in the book](./count_01.svg). See 2.1.1 for
these. Recursive choice, fully drawn out.

SIMULATION VIA TREE: DRAW THREE MARBLES W/ REPLACEMENT:
Each chance is equal for which marble we have drawn. The book iterates a
diagram out to three choices, assuming one Blue and three White.

Each choice, with replacement has 4 options. Iterate as exponent 3 times,
giving 64 possibilities of choosing three marbles with replacement under
assumption _white=3, blue=1_.

We observed X0X. There are three such ways on the diagram, starting from first
blue, and then going to the white trees, 3, and taking the one blue choice. 1 *
3 * 1 -> 3.

Did I get that right?

I did! Cool. Maybe my mental model of this book is accurate so far. I'll have
to read more to know. Meta.

We can eliminate conjecture 0000, because X0X needs X marbles.

Figure 2.4 is epic. Repeats above procedure for XX|00 XXX|0 and XXXX|.


SO:
|0000 -> 0*4*0 -> 0
X|000 -> 1*3*1 -> 3
XX|00 -> 2*2*2 -> 8
XXX|0 -> 3*1*3 -> 9
XXXX| -> 4*0*4 -> 0
20

Grokkage achieved. Didn't look at the book. Walk the path, multiply at each
stage, ways to choose. Recursive choice multiplication similar to Big(O). Both
combinatoric, but constant factors matter here.

Strategy: start drawing the tree if confused. Multiply down the path and
condense as you go to avoid living your life drawing the diagram.

## 2.1.2 Combining other information

Now: assume each possibility is equal. This is the prior. Count the ways that
the data is consistent with these possibilities for each.

OBSERVATION:
Draw another marble, sequence is now X0XX

This is our prior, we already drew 3.

|0000 -> 0*4*0 -> 0
X|000 -> 1*3*1 -> 3
XX|00 -> 2*2*2 -> 8
XXX|0 -> 3*1*3 -> 9
XXXX| -> 4*0*4 -> 0


Update:

    X0X            PRIOR  Draw X
|0000 -> 0*4*0 ->   0   -> 0 -> 0
X|000 -> 1*3*1 ->   3   -> 1 -> 3
XX|00 -> 2*2*2 ->   8   -> 2 -> 16
XXX|0 -> 3*1*3 ->   9   -> 3 -> 27
XXXX| -> 4*0*4 ->   0   -> 4 -> 0

43 / 4^4 -> 43/256

NOTE: Draw X is _independent_ because we're doing replacement.

The above is equivalent from doing the analysis from scratch with 4-seq
observation.

Language:
These are Conjectures
|0000
X|000
XX|00
XXX|0
XXXX|

These are Observations
X0X
X0XX

Sum the viable ways for observations given the conjectures. Compare viable
counts to get relative probability among conjectures.

TO UPDATE:
Count the ways the new observation could update the prior number of ways,
multiply.

Ok. Let's skim until code. I can't keep up this level of density.


```
Bayes number ways for conjecture with new info = WaysPrior * WaysNew\_underObservationNew
```

DEF
*Principle of Indifference*: if we know nothing, assign equal weights to all
conjectures. We can do better by looking at the problem for the priors.

THEOREM[Bayes]:
plausibility of conjecture after observation
is proportional to:
[ways the conjecture can produce observation] * [the prior].

DEF
Probability is restating the relative plausibilities so they sum to one. You
can do this by getting _all_ the ways things can happen in a system
(combinatorics) as the denominator or by adding up the Bayesian way
plausibilities.

THEOREM[Bayes restate]:
```
plausibility of p (conjecture) after observation D_new =

ways p can produce D_new * prior plausibility p
----------------------------------------
                sum of products
```
WHERE:
- Sum of products is sum of all `ways` for all `p`
- `p` is a "plausibility". Normalized to add to `1`, it is a probability.

In the example:
`p` is proportion of blue marbles. `D_new` is X0X.

Names in Applied Probability Theory:
- `p` -> PARAMETER value, describes possible conjectures, explanations of our
  system
- `number of ways that p can produce the data` -> LIKELYHOOD Enumerate all
  possible system sequences, eliminate those inconsistent with data.
- `P(p)` (probability of a given `p`) prior plausibility -> `PRIOR PROBABILITY`
- `P(p | OBSERVATION)` -> `POSTERIOR PROBABILITY` wrt an observation.

DEF
*Information Entropy* You get a lot of it when you shuffle a deck of cards.
Also eliminates knowledge about state of system.

## 2.2 Building a Model

Bayesian inference with probabilities, as opposed to the previous,
count-oriented examples, is a leap in abstraction.

Example: Monte Carlo globe sampling Water/Land:

Observation: W L W W W L W L W

ALG: Bayesian Modeling:
1. Motivate the model by narrating how the data might arise
1. Update: educate the model by feeding it data
1. Evaluate: examine the results; perhaps update the model.

### 2.2.1: A data sotry

DEF
A *Descriptive* model specifies associations that can be used for prediction.

DEF
A *Causal* model describes how some events lead to other events.

Models specify an algorithm to create new data that is consistent with the
model.

Motivation: how does the system create data?
1. The true proportion of water covering the globe is `p`
1. Tossing the globe has probability `p` of producing water (W) (assumes random
   spin, likely axial on one dimension, simplification), and `1 - p` of
   producing land (L).
1. Each globe toss is independent (w00t we can multiply).


Recall: resolving the ambiguity between the data story (statistical model), and the
hypothesis is many to many and generally hard.

### 2.2.2 Bayesian Updating

1. Start by assigning paulsibilities to each conjecture.
1. Update these in light of the data; this is *Bayesian Updating*.

Is this a perceptron?

Recall: `W L W W W L W L W`

Assume: plausibility of `p` is uniform. `p` could be `0` to `1`, no assumption.

After seeing `W`, we know that `p` cannot be `0`.

`p` > .5 has increased, since we have evidence for `W`, but none for `L`, yet.

DEF:
`plausibility`: point on the probability density function, for some parameter
`p`.

After seeing `L`, plausibility of `p = 1` is fixed to `0`.

More values of `W` shift the peak right. As more observations are collected,
the peak shifts higher, since fewer values of `p` amass more plausibility
(collecting mass on the PDF).

Each updated set of plausibilities is the set of initial plausibilities for the
next observation.

This inferrence process is bidirectional. We can subtract (divide out) the last
observation to get the previous set of plausabilities.

Bayesian estimates are interprable at any data size. Caveat: reliance on the
prior.

### 2.2.3 Evaluate

Assuming the Bayesian model is correct in the large world, it's learning is
"optimal" (huh?).

More data creates more certainty in the model. Do not interpret this as
certainty in the world. The dependence on the correctness of the model
increases as you feed it more data.

All models are false, some models are useful. Don't try to satisfy yourself by
poking holes in models. Models are for processing information

What we want: is the model adequate for a purpose?

## 2.3 Components of the (Bayesian) model

1. The number of ways each conjecture could produce an observation
1. Accumulated number of ways each conjecture could produce the entire data
   (Bayesian update). Each observation updates, becoming the new prior.
1. The initial plausibility distribution (PDF).

### 2.3.1 Variables

DEF
A *Parameter* is a variable that is not observed, but inferred from observation
of other variables.

Example:

In globe model:

N = L + W

Where:
N is the number of spins
L is the Land spins
W is the Water spins

We haven't defined the values of these variables, but stated a relation
between them.

We do this to count all the ways data could arise, given our assumptions. These
are our assumptions. The statement of the variables and their interrelations.
The words we choose and the names we give.

Worked example:

#### 2.3.2.1: Observed Variables

For the count of `W` and `L`, we define plausibility for all possible `p`.

DEF
Instead of counting, we use a function to give us plausibilities for `p`.
Such a distribution function is called a *Likelyhood* (Bayesian distributions
expand this a little).

When:
- Each event is a binary decision
- The probability of A/B is constant over the stream

i.e.

Two possible events, probabilities `p` and `( 1 - p )`, over the event stream.
```
Pr(W,L|p) = [(W + L) ! / W!L!] * [p ^ W]  * [(1 - p) ^ L]
```
i.e: The counts of `W` and `L` are distributed binomially, with probability `p`
of `W` for each event.

NOTE: The likelyhood function is the most important part of a Bayesian model.

#### 2.3.2.2 Unobserved Variables

DEF
Unobserved variables are *Parameters*

Examples:
- What is the difference between the treatment groups?
- How strong is the causal curative effect of a treatment?
- Are there relevant covariant factors for treatment (e.g. exercise)
- What is the sub-population variation?

DEF
For every parameter you intend your bayesian model to estimate, you must
provide a *Prior*.

Prior choice reflects both what we know and how we want the system to learn in
response to new data.

DEF
When we just pick numbers that feel right that is *Subjective Bayesian*
modeling.

Experiment with priors. It's fun!

NOTE:
Uniform prior PDF over `[a, b]`:
`1 / ( b - a )`
where `b` is the end of the domain, and `a` the start. Note that this
integrates to 1, trivially.

### 2.3.3 The model

`W ~ Binomial(N, p)`
`p ~ Uniform(0, 1)`

Which states:
`W` and `L` are binomially distributed, `W + L = N`, and `p` is chosen with a
uniform flat prior.

Ok, so we have binomially distributed events with a flat prior.

## 2.4 Making teh model go (yeeting with Thomas Bayes

DEF
The updated distribution, attaching new observations to the system, is the
*POSTERIOR DISTRIBUTION* (hehe, posterior).

For all tuples (data, likelyhood, parameters, prior) there exists a unique
posterior distribution (yeet). Relative plausibility distribution (pdf), for
the parameter space, given the assumed likelyhood, prior, and observation
(data).

```
Pr(p | W, L)
```
is "probability of possible values for paramter `p` (distribution) given
observed `W` and `L`".

### 2.4.1 Bayes' Theorm (oh noe maths)

Derivation: using probability math theory to do the garden of forking paths
counting thing (update on possible sets given set off priors).

It's all combinatorics.

Now:

Notation:
`Pr(A,B ... ,Z)` is the *Joint Probability Distribution* of variables
`A,B ..., Z`, which is to say, commalist funcall `Pr` is a joint probability
distribution. Commalist order irrelevant.


```
Pr(W,L,p) = Pr(W, L | p) * Pr(p)
```

So:

```
Pr(p,W,L) = Pr(p | W, L) * Pr(W, L)
```

Which smells associative/commutative, have we an algebra?

INTUITION:
This is the general idea that AND-ing probabilities is multiplication. I.e. set
combinatorics for convolution (all possible combos) is multiplication. I.e. for
any given choice of set Y, I have "choices of set X" ways to extend. Grafting
garden fork trees.

Stack em:
```
Pr(W,L,p) = Pr(W, L | p) * Pr(p)
Pr(p,W,L) = Pr(p | W, L) * Pr(W, L)
```
Run the equivalence
```
Pr(p | W, L) * Pr(W, L) = Pr(W, L | p) * Pr(p)
```
Divide:

```
Pr(p | W, L) = Pr(W, L | p) * Pr(p)
                   -------------
		      Pr(W, L)
```

Don't let `Pr(W,L)` be zero. Which would state that an impossible event has
occurred.

With words:

```
Posterior = [ Probability of data * Prior ] / Average probability of data
```

So what is "Average probability of data"?

Called "evidence", "average likelyhood". Literally, this is the average
probability of observing `(W, L)` for some `(W, L)`. It normalizes the prior.

Notation:

`E(X)` is "The expectation of `X`"

```
Pr(W,L) = E(Pr(W,L | p)) = Integral Pr(W,L | p) Pr(p) dp
```

Integrate over all values of the parameter `p` to get `Pr(W,L)`. Average over
the parameter space (Not the joint probability distribution of `W, L`?).

DEF
This sort of average (over the parameter space), is called a *marginal*
AKA *marginal likelyhood*

Posterior IS Product(prior, data), ok but why? Tree extension combinatorics.
Number of ways of combining the likelyhood count sets.

The paths through the forking garden is:
```
cardinality(Prior Paths) * cardinality(Paths to produce new data)
```

See above intuition for the set combinatorics in play.

The denominator in Bayes' Theorem is just a normalizer for the rules of
probability.

### 2.4.2 Motors: So how do we make it go?

How to condition the prior on the data, the update step to make new curves for
the PDF (such that they integrate to one blah blah blah).

So don't do this with algebra, 2 hard lolz.

Approximations:
1. Grid approximation
1. Quadratic approximatin
1. Markov Chain Monte Carlo w00t (MCMC)

Model -> Prior and a likelyhood

Actuality --> Model -> Prior, likelyhood, how we calculate teh update.

The numerical technique matters. Things can totally go wrong here.

### 2.4.3 Grid approximation

Most parameters are continuous. Even sampling can work pretty well, and then we
can just program discrete integrals (sums of products).

DEF
*Grid approximation*: multiply the prior probability at `p'` by the likelyhood
function at `p'`.

Grid approximation is impractical at large nubmers of parameters (duh).

Let's do grid approximation for globe spinning!

1. Define the grid: how many points to use in estimating the posterior, list
   the parameter values on the grid
1. Compute the value of the prior at each parameter value on the grid
1. Compute the likelihood at each parameter value
1. Compute the unstandardized posterior, by chain-multiplying through the
   priors on the likelihoods
1. Standardize the posterior, by dividing it by the sum of the observed
   variables (??? not sure I'm following this).

   ``` r

GRID_SIZE=100
# define the grid
# Make a grid from 0 to 1 inclusive, 20 points
p_grid <- seq(from=0, to=1, length.out=GRID_SIZE)

# Uniform prior
prior <- rep(1, GRID_SIZE)

# compute the likelihood at each value in the grid (binomial likelihood
# estimator/function), this is given 6 observations of water on 9 spins.
likelihood <- dbinom(6, size=9, prob=p_grid)

# Product of prior and likelihood -> posterior
unstd.posterior <- likelihood * prior

# standardize the posterior, so it sums to 1
posterior <- unstd.posterior / sum(unstd.posterior)
# Note: sum(posterior) -> 1

# Plot the probability against the parameter grid:
plot(p_grid,
  posterior,
  type="b",
  xlab="probability of water",
  ylab="posterior probability"
  )
  ```

Different priors!
```
# Step prior
prior <- ifelse(p_grid < .5, 0, 1)

# binomial likelihood, over all the paramter grid for `Pr(p)`
likelihood <- dbinom(6, size=9, prob=p_grid)

unstd.posterior <- prior * likelihood

# Normalize with average probability over the grid
posterior <- unstd.posterior / sum(unstd.posterior)

plot(
	p_grid,
	posterior,
	type="b",
	xlab="paramter values",
	ylab="posterior probability"
)
mtext("Cuts the left side of the binomial likelihood to zero")
```

### 2.4.4 Quadratic Approximation

Grid approximation is easy for a single-paramter model (what is proportion of
water?) like globe spinning.

We get trivial combinatoric explosion when considering larger models.

A grid with 2 paramters, computed for 100 values, is 10,000. For 10 paramters,
it's 100^10, tooo big.


Take as read
Logarithm of the Gaussian distribution is a parabola.
```
Log(Gaussian(x)) ~ x^2
```

The quadratic approximation assumes that the shape of the posterior
distribution near the peak is Gaussian.

Gaussian distributions have 2 parameters:
- Spread (the variance), how flat it is
- The peak (the mean)

So. The strategy is using a parabola to approximate `log(posterior)`.

This is how linear regression works (there's some linear algebra matrix
minimization estimator thing that happens).

HOW TO:
1. Find the posterior mode. Gradient ascent to find the peak is a simple way to
   do this
1. Estimate the curvature at the peak, this gives the 2nd derivative, and
   defines the parabola.

   ``` r
library(rethinking)
globe.qa <- quap(
		alist(
			W ~ dbinom(W+L, p), # binomial likelihood
			p ~ dunif(0,1) # uniform prior
		     ),
		data = list(W=6, L=3)
	)
   ```

How to use quap:
Provide a formula and a list of data. The *formula* is the likelihood estimator
and the prior. The likelihood esitmator uses the list of variables in the data.
What whacked unbound variable object differentiation is this?

The quadratic approximation is better when it has lots of data. Yup.

Quadratic bayesian approximation is often equivalent (math-terms) to a *Maximum
Likelihood Estimator* and its *Standard Error*. MLEs have some drawbacks, which
also apply to bayesian approximation.

Recall:
The *Hessian* is a square matrix of second derivatives, on the paramter grid.
In the estimator it is a square of
`second_derivative(log(posterior_priobability))`. This describes a gaussian,
because the `log(Gaussian(x)) ~ x^2`.

### 2.4.5 Markov chain Monte Carlo

Heirarchical models don't work with grid estimators. Puter go boom. Parabola
fragile? What do?

Throw darts at the problem and see where they tend to land: random methods.

MCMC: Fuck math, let's just work with the plausibilities of the posterior.

YEET:
```
# Set up the lattice
n_samples <- 1000
p <- rep(NA, n_samples)
p[1] <- 0.5

# Write down our data
W <- 6
L <- 3

for ( i in 2:n_samples ) {
	# Grab randoms on a normal, mode is prior, stddev is .1
	p_new <- rnorm(1, p[i-1], 0.1)

	# Why is this needed? Are values on the normal not [0, 1]?
	if ( p_new < 0 ) p_new <- abs(p_new)
	if ( p_new > 1 ) p_new <- 2 - p_new

	q0 <- dbinom(W, W + L, p[i - 1]) # prob on prior
	q1 <- dbinom(W, W+L, p_new) # Update

	p[i] <- ifelse(runif(1) < q1 / q0, p_new, p[i-1])
}

# Now plot the chain:
dens(p, xlim=c(0,1)) # plotter for density estimates
```

PROBLEM SET YEET:
2E1. -> 1
2E2. -> 3
2E3. -> 1
2E4. To state "the probability of water is 0.7" is to state that in the
inffinite set of all possible events, given that each event occurs equally, a
proportion, 0.7 of these events are water. The other 0.3, given how we set up
the system, are land.

2M1. Globe tossing, uniform prior, Grid approximate:
1. `W,W,W`
2. `W,W,W,L`
3. `L,W,W,L,W,W,W`

1.
``` r
GRID_SIZE = 100
p_grid = seq(from=0, to=1, length.out=GRID_SIZE)
prior <- rep(1, GRID_SIZE)

likelihood = dbinom(3, size=3, p_grid)

unstd.posterior <- prior *  likelihood

posterior = unstd.posterior / sum(unstd.posterior)

plot(
	p_grid,
	posterior,
	xlab='parameter grid',
	ylab='posterior distribution density',
	type="b"
    )

```

2.
``` r
GRID_SIZE = 100
p_grid = seq(from=0, to=1, length.out=GRID_SIZE)
prior <- rep(1, GRID_SIZE)

likelihood = dbinom(3, size=4, p_grid)

unstd.posterior <- prior *  likelihood

posterior = unstd.posterior / sum(unstd.posterior)

plot(
	p_grid,
	posterior,
	xlab='parameter grid',
	ylab='posterior distribution density',
	type="b"
    )
```

3.
``` r
GRID_SIZE = 100
p_grid = seq(from=0, to=1, length.out=GRID_SIZE)
prior <- rep(1, GRID_SIZE)

likelihood = dbinom(5, size=7, p_grid)

unstd.posterior <- prior *  likelihood

posterior = unstd.posterior / sum(unstd.posterior)

plot(
	p_grid,
	posterior,
	xlab='parameter grid',
	ylab='posterior distribution density',
	type="b"
    )
```

2M2.
