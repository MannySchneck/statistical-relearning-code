# Chapter 3: Sampling the imaginary

Consider posterior inference. Medical test for vampirism

`Pr(positive test | vampire) = .95`
`Pr(positive test | mortal) = .01`

`Pr(vampire) = .001`

Bayes rewrite:
```
Pr(vampire | positive test) = [Pr(positive test | vampire) * Pr(vampire)]
                                   -----------------
                                   Pr(positive test)
```

Which is to say: The event conditional * the event generally, divided by the
diagnostic event space.

This is how we calculate the event space:
```
Pr(positive test) =   Pr(positive | vampire) * Pr(vampire)
                    + Pr(positive | mortal) * Pr(1 - Pr(vampire))
```

The co-probability of the conditional event on the diagnostic targets and the
diagnostic target events. i.e. Weighted average of the condition space. Chain
rule notation pseudo-multiply is in effect.


DEF
Confidence interval: an interval of defined probability mass. E.g. From .6 to
.8 has 50% of the probability density for a some posterior we care about.

AKA: Compatibility Interval

DEF
Highest Posterior Density Interval: narrowest interval containing the specified
probability mass.

DEF
Loss function: returns the "cost" of choosing a single value summary point for
the distributaiton

A simple loss function: linear distance:

```
loss = sum(posterior * abs(d - p_grid))
```

Weight the linear loss of any given `p` against point estimate `d` against the
likelihood density, and sum it up.

Join weighted average lol!

R for running this calculation over the whole `p_grid` (effectively a
convolution?)

``` R
loss <- sapply( p_-grid, function(d) sum( posterior - abs( d - p_grid) ) )
```

Then:

```
best_p = p_grid[which.min(loss)]
```

Where `which.min` is the `argmin` on the index.

Linear (absolute) loss leads to the `median` on the samples as the point
estimate.

Quadratic `(d - p) ^ 2` leads to the `mean` on the samples as the point
estimate.

## 3.3 Sampling to simulate prediction

1. Model design: We can sample from the prior to understand how our prior will
   influence the model.

2. Model checking: generating from teh model and sanity checking is another
   cross checking point.

3. Validation: Generate data from a _known_ set of parameters, and compare to
   reality. Check the parameters that we recover from the model Against the
   known model.

4. Research design: use a hypothesis -> simulate observations under a process
   -> statistical model binding. (power analysis?) Determine if the research
   design can be effective in the "small world" sense.

5. Forecasting: prediction, but also to check if the model works lol.

### 3.3.1 Dummy data

Glob tossing: Some fixed proportion `p` of land and water. Ergo, the ratio of
spin-results is going to be `p` to `1 - p`.

What we did: infer plausibility for values of `p` given observations.

What we can do: generate observations implied by the model.

Bayesian models are *generative*: We can sample the likelihood function to
generate data.

DEF
Simulated data AKA *Dummy Data*

Recall: binomial likelihood:
```
Pr(W|N, p) = [ N! / (W!(N-W)!) ] * p^W (1 - p)^(N - W)
```

There's a lot of symmetry here. Basically:

Num trials factorial over the product of the result sides (wins * losses). All
that times `p` and complement exponentiated respectively.

Recall: binomial distribution is the probability of observing `W` water spins
over `N` trials, given probability `p` for `W` of each trial (`1 - p` for the
complement event `L`).

Kinda tricky here. The total `W` proportion on a sampled dataset ends up being
`sum(Number of "wins") / number of trials * _trial size_`

### 3.3.2 Model checking

DEF
*Model checking* is
1. making sure the model fitting worked
2. Evaluating the fitness of the model for some purpose.

Bayesian models are generative, so we can generate data after conditioning.

#### 3.3.2.1 Did we fuck up?

Generate data and see if it matches the training data lol. o


#### 3.3.2.2

Does this model describe the data? Imperfect preditiction can be a sign of the
"learning"--that we're not overfitting.

Sampling parameters from the posterior AND sampling simulated observations ->
uses the whole distribution, captures uncertainty i.e. "flatness".

Ways that predictions are uncertain:
1. Observational: Even knowing `p`, we can't predict the next globe spin.
2. Uncertainty about `p`: The _distribution_, means that there's uncertainty.
   We don't really _know_ what the true `p` water/land is.

So: We need to propagate this uncertainty about `p` to the predictions--since
this will affect everything (we don't really know the model's parameters).

DEF
*Posterior Predictive Distribution*: Average the prediction of each parameter
value `p` by the posterior distribution density. Sum these predictions to get
the PPD.

Example:

```
prior <- rep(1, 1000)
p_grid <- seq(from=0, to=1, length.out=1000)
likelihood <- dbinom(6, size=9, p=p_grid) # Observe 6 W in a trial of size 9
posterior <- prior * likelihood
posterior <- posterior / sum(posterior)

samples <- sample(p_grid, size=1e4, replace=TRUE, prob=posterior)

# Samples are proportionate to posterior probabilities, effectively an
# average.
w <- rbinom(1e4, size=9, prob=samples)
```

`prob=samples` propagates the parameter uncertainty. Using each sampled
parameter value to make a binomial prediction. Effectively averaging over the
distribution because of the sampling process. Note that the number of samples
has to be.... how big? Has to dominate something but not sure what.

Model checking:
* The binomial model assumes no correlation between subsequent tosses.

To check: look at "flips" in sequence in the data vs what we sample.

Look at "longest run".

