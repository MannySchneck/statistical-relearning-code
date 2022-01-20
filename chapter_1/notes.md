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

##

