# MCUP
Monte Carlo uncertainty propagation in regression. Have you ever wondered how to estimate the uncertainty of your regression parameters correctly? MCUP will help you to get it right. 

## Status
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/mcup.svg)](https://pypi.org/project/mcup/) [![Build Status](https://travis-ci.org/detrin/MCUP.svg?branch=master)](https://travis-ci.org/detrin/MCUP) [![PyPI version shields.io](https://img.shields.io/pypi/v/mcup.svg)](https://pypi.org/project/mcup/) [![Documentation Status](https://readthedocs.org/projects/mcup/badge/?version=latest)](https://readthedocs.org/projects/mcup/?badge=latest) [![codecov](https://codecov.io/gh/detrin/MCUP/branch/master/graph/badge.svg?token=Dx6elQkztR)](https://codecov.io/gh/detrin/MCUP)


## Abstract
We aim to provide a regression parameter error estimator (PEE) that provides information about the error of regression parameters based on the estimated errors of measured points. Our method ought to be reliable in a variety of real-world experiments, with precission being our primary goal.

## Keywords
> Regression parameter errors, errors in variables, estimation of regression uncertainty, confidence intervals of fit parameters, standard error of least squares, propagation of uncertainty from data to parameter space, likelyhood in parameter space

## Scope
This repository will contain implementations of several PEE algorithms as well as a meta testing suite which evaluates the algorithms on various sets of data and shows how good they were. When some of the PEE prove to be good for real-world usage, they will get their own repository.

## Terminology
* PEE – Parameter Error Estimator, an algorithm which, based on measured data (points with error bars), estimates some statistic properties (standard errror, mean, ...) of the parameters returned by a regression algorithm (be it OLS, Deming or something else)

* VE – Virtual Experiment, a program which from a given relationship between two variables produces a set of “measured points” with simulated random (and possibly also systematic) error. It can also return error bars for individual measured points which can either correspond precisely to the error of measurement, or can be intensionally off to simulate a bad estimate of measurement precission that can happen in real-life scenarios.

* DistGen – Parameter Distribution Generator, a program that runs a VE many times and for each produced set of data it runs a regression and saves its best-fit parameters. Then out of these sets of parameters it constructs a probability distribution determining how likely it is that a particular set of parameters will be the best fit for the VE. It is this probability distribution that we want to estimate with PEE.

* PPD – Parameter Probability Distribution, the distribution that is generated by DistGen and estimated by PEE

* QDD – Quantative Distribution Descriptor, a meta-analysis tool which extracts useful information from a PPD, such as the difference of the true value and the mean of PPD, the standard deviation of PDD and how Gaussian/Lorentzian it is. It is necessary to extract only few important bits of information from the PPD in order to compare many of them at once.

## Roadmap
1. Implement a naive PEE, a DistGen and QDD
2. Improve the performance of least-squares regression to speed up Monte-Carlo
3. Study the results of PEE
   
   * on polynomials up to degree 3
   * for true values equidistant on the x axis
   * without systematic error
   * with random error being of the same order on all data points
   * with errorbars exactly matching the measurement error

4. Implement more PEEs and compare them
5. Compare on a wider variety of experiments
  


