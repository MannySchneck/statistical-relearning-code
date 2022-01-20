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


