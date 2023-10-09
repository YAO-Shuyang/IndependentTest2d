import numpy as np
from numba import jit
import copy as cp
from scipy.stats import ttest_1samp

@jit(nopython=True)
def bi_ecdf(
    x: np.ndarray,
    y: np.ndarray,
    t1: float,
    t2: float
) -> float:
    """bi_ecdf: Bivariate Empirical Cumulative Distribution Function

    Parameters
    ----------
    x : np.ndarray
        Dimension 1 of the observation.
    y : np.ndarray
        Dimension 2 of the observation.
    t1 : float
        Threshold for the dimension 1.
    t2 : float
        Threshold for the dimension 2.

    Returns
    -------
    float
        Empirical cumulative distribution function at (t1, t2).
        
    Notes
    -----
    Fn(t1, t2) = sum Indicator(xi <= t1, yi <= t2)/n, while n is the total number of observation
    """
    return np.where((x<=t1) & (y<=t2))[0].shape[0]/x.shape[0]

@jit(nopython=True)
def si_ecdf(
    x: np.ndarray,
    t: float
) -> float:
    """si_ecdf: Empirical Cumulative Distribution Function

    Parameters
    ----------
    x : np.ndarray
        Observation.
    t : float
        Threshold.

    Returns
    -------
    float
        Empirical cumulative distribution function at t.
        
    Notes
    -----
    F(t) = sum Indicator(xi <= t)/n, while n is the total number of data
    """
    return np.where(x<=t)[0].shape[0]/x.shape[0]

@jit(nopython=True)
def supremum(
    diff: np.ndarray
) -> float:
    """supremum: the definition of the test statistic.

    Parameters
    ----------
    diff : np.ndarray
        The difference between Fn(t1, t2) and Fx(t1)*Fy(t2)

    Returns
    -------
    float
        The supremum of the |Fn(t1, t2) - Fx(t1)*Fy(t2)|
    """
    return np.max(np.abs(diff))

@jit(nopython=True)
def test_stats(
    x: np.ndarray,
    y: np.ndarray,
    x_bin_num: int = 1000,
    y_bin_num: int = 1000
):
    x_bins = np.linspace(np.min(x), np.max(x), x_bin_num+1)
    y_bins = np.linspace(np.min(y), np.max(y), y_bin_num+1)
    
    diff = np.zeros((x_bins.shape[0], y_bins.shape[0]), np.float64)
    
    for i in range(x_bins.shape[0]):
        for j in range(y_bins.shape[0]):
            diff[i, j] = bi_ecdf(x, y, x_bins[i], y_bins[j]) - si_ecdf(x, x_bins[i])*si_ecdf(y, y_bins[j])
            
    return supremum(diff)

from tqdm import tqdm
def monte_carlo(
    simu_times: int = 1000,
    simu_len: int = 1000,
    x_bin_num: int = 100,
    y_bin_num: int = 100
):
    simu_statistics = np.zeros(simu_times, np.float64)
    
    for i in tqdm(range(simu_times)):
        x = np.random.rand(simu_len)
        y = np.random.rand(simu_len)
        simu_statistics[i] = test_stats(x, y, x_bin_num, y_bin_num)
        
    return simu_statistics

def permutation(
    x: np.ndarray,
    y: np.ndarray,
    x_bin_num: int = 1000,
    y_bin_num: int = 1000,
    simu_times: int = 1000
):
    simu_statistics = np.zeros(simu_times, np.float64)
    
    for i in tqdm(range(simu_times)):
        np.random.shuffle(x)
        np.random.shuffle(y)
        simu_statistics[i] = test_stats(x, y, x_bin_num, y_bin_num)
        
    return simu_statistics

def indeptest(x, y, simu_times=1000, x_bin_num=100, y_bin_num=100, shuffle_method: str = "monte_carlo"):
    """indeptest
    
    See "Distribution Free Tests of Independence Based on the Sample Distribution Function. J. R. Blum, J. Kiefer and M. Rosenblatt, 1961."
    https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-32/issue-2/Distribution-Free-Tests-of-Independence-Based-on-the-Sample-Distribution/10.1214/aoms/1177705055.full
    
    According to R document of function indeptest https://search.r-project.org/CRAN/refmans/robusTest/html/indeptest.html

    Parameters
    ----------
    x : _type_
        Dimension 1 of observation
    y : _type_
        Dimension 2 of observation
    simu_times : int, optional
        Times of Monte-Carlo simulation, by default 1000
    x_bin_num : int, optional
        The number of bins to divide the range of x observation, by default 1000
    y_bin_num : int, optional
        The number of bins to divide the range of y observation, by default 1000
    shuffle_class : str, optional
        The method of shuffling, by default "monte_carlo".
        Options: "monte_carlo" or "permutation"

    Returns
    -------
    test_statistics: float
        The test statistics of independent task.
    res: scipy.stats._stats_py.TtestResult
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x ({x.shape}) and y ({y.shape}) must have the same length!")
    
    print("It needs several minutes to perform monte-carlo simulation.")
    test_statistics = test_stats(x, y, x_bin_num, y_bin_num)
    
    if shuffle_method == "monte_carlo":
        simu_statistics = monte_carlo(simu_times, x.shape[0], 100, 100)
    elif shuffle_method == "permutation":
        simu_statistics = permutation(cp.deepcopy(x), cp.deepcopy(y), x_bin_num, y_bin_num, simu_times)
    else:
        raise ValueError(f"shuffle_method must be 'monte_carlo' or 'permutation', not {shuffle_method}")
    
    return test_statistics, ttest_1samp(simu_statistics, test_statistics, alternative='less')


if __name__ == "__main__":
    from mylib.statistic_test import mkdir, Clear_Axes, figpath, figdata
    # You can annotate this line if you cannot import mylib.
    import os
    import pickle
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr, linregress
    
    # You can manually set a "figpath" to save the figures, and a "figdata" directory to save the results
    # in the forms of both PKL file or EXCEL sheet.
    
    code_id = "0061 - Independent Test (Blum et al., 1961)"
    loc = os.path.join(figpath, code_id)
    mkdir(loc)
    
    test_time = 41
    noise_amp = np.linspace(0.96, 1, test_time)
    pearson_corr = np.zeros(test_time, np.float64)
    pearson_pvalue = np.zeros(test_time, np.float64)
    
    linreg_slope = np.zeros(test_time, np.float64)
    linreg_rvalue = np.zeros(test_time, np.float64)
    linreg_pvalue = np.zeros(test_time, np.float64)
    
    indep_statistics = np.zeros(test_time, np.float64)
    indep_pvalue = np.zeros(test_time, np.float64)
    ttest_statistics = np.zeros(test_time, np.float64)
    
    # Create canvas to avoid too much memory use.
    fig = plt.figure(figsize = (4,4))
    ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    # You can annotate this line if you cannot import mylib. Use ax = plt.axes() as an alternative.
    ax.set_aspect("equal")
    ax.axis([0,1,0,1])
    ax.plot([0,1], [0,1], ':', color='gray', linewidth=0.5)
    
    for i, n in enumerate(noise_amp):
        # Generate data with different degree of noise.
        x = np.random.randn(1000)
        y = x*(1-n) + np.rans
        
        # Comparing the indeptest with pearson correlation and linear regression to test its efficiency.
        pearson_corr[i], pearson_pvalue[i] = pearsonr(x, y)
        linreg_slope[i], intercept, linreg_rvalue[i], linreg_pvalue[i], _ = linregress(x, y)
        indep_statistics[i], tteststruct = indeptest(x, y)
        ttest_statistics[i], indep_pvalue[i] = tteststruct.statistic, tteststruct.pvalue
        
        print("Noise Amplitude: ", n)
        print(f"Pearson: {pearson_corr[i]}, {pearson_pvalue[i]}")
        print(f"Linreg: {linreg_slope[i]}, {linreg_pvalue[i]}")
        print(f"Independent: {indep_statistics[i]}, {indep_pvalue[i]}", end='\n\n')
        
        # Visualizing...
        b = ax.plot([0, 1], [intercept, intercept + linreg_slope[i]], color='orange', linewidth=0.5)
        a = ax.plot(x, y, 'o', markeredgewidth = 0, markersize = 1, color = "black")
        ax.set_title(f"Pearson: {round(pearson_corr[i], 2)}, {round(pearson_pvalue[i], 3)}\n"+
                     f"Linreg: {round(linreg_slope[i], 2)}, {round(linreg_pvalue[i], 3)}\n"+
                     f"Independ: {round(indep_statistics[i], 2)}, {round(indep_pvalue[i], 3)}\n")
        plt.tight_layout()
        plt.savefig(os.path.join(loc, f"Noise Amplitude - {n}.png"), dpi=600)
        plt.savefig(os.path.join(loc, f"Noise Amplitude - {n}.svg"), dpi=600)
        
        for j in a+b:
            j.remove()
    
    # Cleaning data dots and the regression-fitted line while keeping the diagonal gray dotted line.     
    plt.close()
        
    Data = pd.DataFrame(
        {
            "Noise Amplitude": noise_amp,
            "Pearson Correlation": pearson_corr,
            "Pearson P-value": pearson_pvalue,
            "Linear Regression Slope": linreg_slope,
            "Linear Regression P-value": linreg_pvalue,
            "Linear Regression R-value": linreg_rvalue,
            "Independent Test Statistic": indep_statistics,
            "Independent Test P-value": indep_pvalue,
            "1 Sample T-test Statistic": ttest_statistics
        }
    )
    
    # Save the data
    Data.to_excel(os.path.join(figdata, code_id+" [tail].xlsx"), index=False)
    
    with open(os.path.join(figdata, code_id+" [tail].pkl"), 'wb') as handle:
        pickle.dump(Data, handle)