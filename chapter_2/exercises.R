# PROBLEM SET YEET
# 2E1. -> 1
# 2E2. -> 3
# 2E3. -> 1
# 2E4. To state "the probability of water is 0.7" is to state that in the -->
# infinite set of all possible events, given that each event occurs equally, a -->
# proportion, 0.7 of these events are water. The other 0.3, given how we set up -->
# the system, are land. -->

########################################
# 2M1. Globe tossing, uniform prior, Grid approximate: -->

binom_grid <- function(slug, grid_size, binom_successes, binom_size) {
	p_grid = seq(from=0, to=1, length.out=grid_size)
	prior <- rep(1, grid_size)

	likelihood = dbinom(binom_successes, size=binom_size, p_grid)
	unstd.posterior <- prior *  likelihood
	posterior = unstd.posterior / sum(unstd.posterior)


	plot(
		p_grid,
		posterior,
		xlab='parameter grid',
		ylab='posterior distribution density',
		type="b"
	    )

	mtext(slug)
}

# 1. `W,W,W` -->
binom_grid("subprob 1", 100, 3, 3)

# 2. `W,W,W,L` -->
binom_grid("subprob 2", 100, 3, 4)

# 3. `L,W,W,L,W,W,W` -->
binom_grid("subprob 3", 100, 5, 7)

########################################
# 2M2. Step Prior

step_bgrid <- function(slug, b_success, b_size) {
	p_grid = seq(from=0, to=1, length.out=100)
	prior <-  ifelse(p_grid < .5, 0, 1)
	prior <- prior / sum(prior)


	likelihood <- dbinom(b_success, b_size, prob=p_grid)
	unstd.posterior <- prior * likelihood
	posterior <- unstd.posterior / sum(unstd.posterior)

	plot(
	     p_grid,
	     posterior,
	     xlab='parameter grid',
	     ylab='posterior density',
	     type='b',
	)
}

# 1. `W,W,W` -->
step_bgrid("subprob 1",  3, 3)

# 2. `W,W,W,L` -->
step_bgrid("subprob 2",  3, 4)

# 3. `L,W,W,L,W,W,W` -->
step_bgrid("subprob 3",  5, 7)


########################################
# 2M3.
# Parameter: Earth or Mars?

# Flat prior
GRID_SIZE = 100
p_grid <- seq(from=0, to=1, length.out=GRID_SIZE)

# Since we don't know which planet, assume flat prior
prior <- rep(1, GRID_SIZE)

# Likelihood is binomial, since it's earth/mars

# Pr(W | Earth) = .7
# Pr(W | Mars) = 0

# Want: Pr(Earth | L)

## Prior of "Earth"", update with observation, normalize.
# Pr(Earth | L) = [Pr(Earth) * Pr(L | Earth)] / Pr(L)
# > .5 * .3
# [1] 0.15
# > (.5 * .3) / (1.3 / 2)
# [1] 0.2307692

########################################
# 2M4.
# Three cards
# A: Two black sides
# B: One black, one white
# C: Two white sides

# Observation: black is facing up.
# Find: probability that other side is black.

# Six sides,
# Count(C) <- 0
# Count(B) <- 1
# Count(A) <- 2

# Situation: A or B, We're done. No further choices.
# Two ways to get A, so it's (2/3), since A has twice the ways to get to our
# observation

# 2M5
# Cards: B/B, B/W, W/W, B/B

# Observation: draw, black face up. Probability that other side is black?

# Count(B/B) <- 2
# Count(B/B) <- 2
# Count(B/W) <- 1
# Count(W/W) <- 0

# -> 4 / 5, since top 2 can produce a black flip, and no others can.

# 2M6

# Ways to pull | Ways to be black face up | Multiply
# B/B -> 1    -> 2                       -> 2
# B/W -> 2    -> 1                       -> 2
# W/W -> 3    -> 0                       -> 0

# Pr(Black | Pulled black) = 2 / 4 =  .5

# 2M7
#   draw  black up   draw B/W White up
# B/B -> 1 -> 2   ->  1
#                    W/W White up
#                 ->  2
#                    W/W White up
# B/W -> 1 -> 1   ->  2
#                    B/W White up
# W/W -> 1 -> 0 ->   1

# Ways Other side black = 2 + 4 = 6
# Other ways -> 2

# 6 / 8 = 0.75

# 2H1.

# Parameter: What species of panda?

# Pr(Twins | A) = 0.1
# Pr(Twins | B)  = 0.2

# Want: Pr(Twins | LastBirth == Twins)

# Pr(B | Twins) = [ Pr(Twins | B) * Pr(B) ] / Pr(Twins)
# Pr(B | Twins) = (0.2 * 0.5) / 0.15 = 2 / 3

# Pr(A | Twins) = [ Pr(Twins | A) * Pr(A) ] / Pr(Twins)
# Pr(A | Twins) = [ 0.1 * .5 ] / 0.15 = 1 / 3

# Pr(Twins | LastBirth == Twins)

# Priors   Twins       Twins again
#  A  .5    -> 1/ 3    -> [(1 / 3) * .1] -> 1 / 30
#  B  .5    -> 2 / 3   -> [(2 / 3) * .2] -> 4 / 30

# 5 / 30 -> 1 / 6

# 2H2
# 1/ 3

# 2H3.

# Observed: twins, not twins
# Want: P(A | (twins, not twins))

prior <- c(1, 1)

a_likelihood <- (.1 * (1 - .1))
b_likelihood <- (.2 * (1 - .2))

p_grid <- c(1, 1) # "Pandaness"
likelihood <- c(a_likelihood, b_likelihood)

birth_posterior <- ( prior * likelihood  )
birth_posterior <- birth_posterior / sum(birth_posterior)

birth_posterior[1]

# 2H4

# P(A | tests A) = P(A) * P(Tests A | A) / P(Tests A)

p_a <- (.5 * .8) / (.5 * .8 + .5 * .35) # Denom is weighted average of "A"

p_grid <- c(1, 1) # "Species-icity"
prior <- c(p_a, 1 - p_a)

# Single round of inference
a_likelihood <- (.1 * (1 - .1))
b_likelihood <- (.2 * (1 - .2))
likelihood <- c(a_likelihood, b_likelihood)
posterior <- prior * likelihood
posterior <- posterior / sum(posterior)


posterior[1]
