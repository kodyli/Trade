from __future__ import print_function
import datetime
import numpy as np
from pandas_datareader import data as web
from scipy.stats import norm


def var_cov_var(P,c,mu,sigma):
	alpha = norm.ppf(1-c,mu,sigma)
	return P-P*(alpha + 1)
	
if __name__ == "__main__":
	start = datetime.datetime(2010,1,1)
	end = datetime.datetime(2014,1,1)
	citi = web.DataReader("C","yahoo",start,end)
	citi["rets"]=citi["Adj Close"].pct_change()
	print(citi["rets"])
	P = 1e6
	c = 0.99
	mu = np.mean(citi["rets"])
	sigma = np.std(citi["rets"])
	var = var_cov_var(P,c,mu,sigma)
	print(var)