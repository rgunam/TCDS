# TCDS: Time-Constant-Domain Spectroscopy

## This code provide the useful methods and functions for computing the time-constant-domain spectrum (TCDS) from impedance spectroscopy experimental data.

The TCDS is an impedance-based method which take ```N``` measurements on the frequency-domain and transforms the impedance spectrum to a time-constant-domain spectrum, via de Distribution of Relaxation Times Model (DRT).

The algorithm solves the DRT using ridge regression, as it is an ill-posed problem.

The TCDS script runs under Python 3 along with the following packages:

- numpy
- scipy
- matplotlib (for debugging)
- cvxpy - Go to [Home page](https://www.cvxpy.org/index.html)

------

### Example

You can test the code using an impedance dataset for a dummy RC circuit. The file is in .csv format. 


1) Input -> Impedance data : 
![](https://github.com/rgunam/TCDS/blob/master/nyquist_RC.png)

2) Output -> TCDS: 
![](https://github.com/rgunam/TCDS/blob/master/nyquist_RC.png)
