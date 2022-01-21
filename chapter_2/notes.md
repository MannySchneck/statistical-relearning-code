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


