 # PROBLEM SET YEET
 # 2E1. -> 1
 # 2E2. -> 3
 # 2E3. -> 1
 # 2E4. To state "the probability of water is 0.7" is to state that in the -->
 # infinite set of all possible events, given that each event occurs equally, a -->
 # proportion, 0.7 of these events are water. The other 0.3, given how we set up -->
 # the system, are land. -->

 # 2M1. Globe tossing, uniform prior, Grid approximate: -->
 # 1. `W,W,W` -->
 # 2. `W,W,W,L` -->
 # 3. `L,W,W,L,W,W,W` -->

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

# 2.
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

# 3.
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

2M2.
