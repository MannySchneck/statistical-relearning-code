# The Golem of Prague

## A decision tree for statistics:

* Parametric Assumptions:
1. Independent samples
2. Data normally distributed
3. Variances are equal


Figure 1.1:
1. Type of data?
	1. continuous
		1. Relationship question
			1. Have dependent/indepent variables
				Regression Analysis
			1. Correlation Analysis:
				1. Parametric: Pearson's R
				1. Nonparametric: Spearman's rank correlation

		1. Difference question
			1. Between Means
			1. Between Variances
			1. Between Multiple means; single variable
				1. Two groups
				 TODO
				1. More than two
					1. Parametric assumptions satisfied
						1. One Way ANOVA
					1. Parametric assumptions NOT satisfied
						1. Transform data?:
						Treat as parametric?
						1. Not Transform Data:
						Kruskal-Wallis test
					1. If one way ANOVA or Kruskall-Wallis
					   is significant, do post-hoc test
					   [Bonferroni's, Dunn's, Tukey's]

	1. discrete
		1. Counts < 5
			Chi-square test, one and two sample.
		1. Counts > 5
			Fisher's exact test.

Tests:
TODO: Consider a spaced repetition algorithm for this material

Continuous:
Easy correlation analysis (already know dependence/indepedence relationships)
* Regression Analysis: Continuous with independent variables, for analyzing
  relationships

Complex correlation analysis
* Pearson's R (parametric)
* Spearman's Rank Correlation (nonparametric)


Discrete:
* Chi Squared test for discrete analysis on count < 5
* Fisher's exact test for discrete analysis on count > 5

## 1.2 Rethinking Statistics

Rejection of Popperism regarding falsifiying hypotheses as the way to do
science.

Instead: falsify the null hypothesis with statistics.

The first is a folk Popperism. Popper realized that falsifying the hypothesis
is not how science is actually practiced.

1. The model is not the hypothesis. This is a many<-\> many. So we're just
   whacking entries in the mapping table.

1. The data correctness is always uncertain (measurement error).


NOTE: Process models as mapping tables in a relational sense, between
statistical models and the hypothesis, is a useful lens

Process models are causal, giving us places to probe.

Statistical models (these, anyway) are not causal.

Correspondance problem:
This complicates using statistical models in relation to hypotheses.

Usual: Check a statistical model against the null hypothesis, to reject it; but
the null statistical model is not tightly bound to the null hypothesis.

Identify more tightly bound hypotheses <-> process <-> stat model chains. Find
the ones that will break the ambiguities in your SAT system.

Compare predictions of more than one model; process and statistical.

Process models let us take into account hidden variables and sampling bias.

### Rethinking: entropy and model identification
Distributions: normal, binomial, poisson.

:points-up: exponential distributions. THese tend toward maximum entropy.

We can't use these to get at process models.

### 1.2.2. Measurement

H -> D

If not d then not that implication. This is modus tollens.

Measurement is hard. People can look at the same thing and disagree (is the
dress gold or blue?)

Measurement complicates falsification. E.g. faster than light nutrinos.

Hypothesis: Black swans exist
False positive: white swan with dirty feathers
False negative: black swan with confusing specular highlights (shiny feathers)

Statistical hypotheses let us do degree-rather-than-kind tests.

## 1.3 So what do we do?

- Bayesian analysis
- Model comparison
- Multilevel models
- Graphical causal models

### Bayesian
Under a set of assumptions, how many ways could this data happen?

Probabalistic, subset of set of outcomes fraction.

Frequentist: Only measure frequency within a large datset to estimate
probability. Assuiming that if we were able to repeat the measurement, the
pattern would hold.

The only thing that matters is the sampling distribution. This is useful in
some contexts where you're collecting a lot of measurement.

Just as absurd as Bayes probability uncertainties (opinions lol?)

Image reconstruction is "educated guessing", Bayesian for that reason.

"Bayesian" metaphysics is just that; not a logical claim to rationality (lol).

Bayesian more intuitive--less likely to be used wrong. Not an inconsiderable
advantage in scientific statistics.

This book: Laplace-Jeffreys-Cox-Jaynes Bayesian probability.

### Yeah, so?

When there is more than one model:

- Cross-Validation
- Information Criteria

Keep reading to find out what the fuck these are.

Get too smart and you start Over-Fitting; occam's razor, empirical here.

i.e. Don't Validate On Your Training Data, but more mathy and philosophical
than that.

"Fitting is easy; prediction is hard". Nice, pithy.

Cross-validation and information criteria are better than fit for assessing
prediction.

Apparently information criteria are new. Fun. They're over-used because they
can be used as arguments that "I'm right".

### Multilevel models

Regard parameters as missing sub-models.

Multilevel models: model how a parameter gets its value, embed that in the
model.

Cascading levels of uncertainty up to the root.
MultiLevel Models AKA:
- hierarchical
- random effects
- varying effects, mixed effects

DEF
*Multilevel models*

Multilevel models have a natural bayesian representation.

Multilevel models help with overfitting; forcing into the simple model at the
top; discard irrelevance.

Partial Pooling: sharing _some_ information across units (what are "units"?) of
the data.

Examples:
- Adjusting estimates for repeat samples: i.e. same patient, same claim,
  multiple dx codes
- Adjusting for different levels of sampling: i.e. we have more data from
  Chicago.
- To study variation: multilevel models explicitly model variation in the
  choice of sub-models.
- To avoid averaging: Multilevel models can be used to avoid pre-aggregation,
  and preserve/communicate that uncertainty/variation through the model as a
  system. I.e. preaggregation is the root of all evil.

Clusters/groups of measurements that differ. Differ in _kind_, in the way they
need to be sampled and treated because of the human contextual knowledge
associated with devising the grouping.

Examples:
- models for missing data
- time series models
- factor analysis
- Some spatial and network regressions

Paired t-tests are multi-level?

TODO: understand the multilevel modeling paradigm. Level wanted: Grok.

Multilevel model brings engineering into statistics.

TODO: make my default paradigm multi-model, which I think it has been
implicitly. Commit to whatever this book is thinking and then assess later once
I Grok.

Fitting and interpreting multi-level regression is a non-reproducible (yet)
human activity.

## 1.3.4 Graphical causal models

Data does not have cause and effect. Association can be either way or neither.

Paradox(?): causally incorrect models can have better predictions than causally
correct models.

Implicit? Causal models are more valuable than correct prediction: they help us
structure our thinking.

The most important information for interpretation, almost tautologically, is
outside of the data--otherwise it would just be part of the model; there would
be no need to interpret. No hubris; don't that.

DEF
The *Identification* problem is not the same as raw prediction. Model
"identification", ask the question, which model? and you will find this problem
staring back at you.

Focusing on prediction can mislead us. Be aware of your goals as part of your
interpretation. Are you trying to Understand, or to predict.

Prediction may improve when we use causally misleading info. E.g. When
branchs sway there is wind, does the first thing cause wind? No. Why though?
Well that's ludicrous. It violates what we know about wind. CounterFactual: I
can wave the branches with my arms. It won't make it windy. Causality as human
agency. Causality as anthropomophism?

A Kind of Scientific Animism. It's useful. William James; pragmatism.

Scientific models contain more information than the statistical models that
describe them can contain.

To find interventions that are effective, we need to think causally, since we
want to do actions to cause things.

So:

Causal model that can be used to design a statistical model (or set of them
(multilevel?)) for the purpose of understanding causes in the system.

XREF: Systems theory leverage points, Donella R. Meadows.

Things are complicated; leverage heuristics; vague hypotheses; vague models;
pragmatism.

DEF
*Graphical Causal Models*, a simple form being the *DAG*,

DEF
*Causal Salad*: Tossing "control" variables into a statistical model, observing
how the estimates change, and then telling a story about causation, and testing
reality to see if it matches.

Be careful. No guarantees that we're doing this correctly.


