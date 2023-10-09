# IndependentTest2d
To test whether two continuous variables (observations) are independent or not.

Based on a [``R script documentation``](https://search.r-project.org/CRAN/refmans/robusTest/html/indeptest.html), according to the author's understanding.
Paper: [``Distribution Free Tests of Independence Based on the Sample Distribution Function. J. R. Blum, J. Kiefer and M. Rosenblatt, 1961.``](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-32/issue-2/Distribution-Free-Tests-of-Independence-Based-on-the-Sample-Distribution/10.1214/aoms/1177705055.full)

Basic principle:
![](https://github.com/YAO-Shuyang/IndependentTest2d/blob/main/illustration1.png)

We provide two method for providing a standard comparison: "monta-carlo" and "permutation".

The "monta-carlo" method is based on this rationale:
Under H0 the test statistic is distribution-free and is equivalent to the same test statistic computed for two independent continuous uniform variables in [0,1], where the supremum is taken for t1,t2 in [0,1]. Using this result, the distribution of the test statistic is obtained using Monte-Carlo simulations.

The "permutation" is simply shuffling the input x, y observations to disrupt any implicitly linked relationships between these two variables.

Generally, the "monta_carlo" is too rigorous and strict, as it uses two standard independent distributions. Thereby, it might not suit experimental data, which inevitably contains noises and some sort of systematic error.
The "permutation", however, uses true data instead of standard data. It would be more tolerable to any experimental cases than "monte_carlo", therefore it is recommended.
