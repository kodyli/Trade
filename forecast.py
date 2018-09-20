from __future__ import print_function
import datetime
import numpy as np
import pandas as pd
import sklearn

from pandas_datareader import data as web
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.svm import LinearSVC, SVC


def createLaggedSeries(symbol,startDate,endDate,lags=5):
	ts = web.DataReader(symbol, "yahoo", startDate-datetime.timedelta(days=365), endDate)
	tsLag = pd.DataFrame(index = ts.index)
	tsLag["Today"] = ts["Adj Close"]
	tsLag["Volume"] = ts["Volume"]
	
	for i in range(0,lags):
		tsLag["Lag%s" % str(i+1)]=ts["Adj Close"].shift(i+1)
		
	tsRet = pd.DataFrame(index = ts.index)
	tsRet["Volume"] = tsLag["Volume"]
	tsRet["Today"] = tsLag["Today"].pct_change()*100.0
	
	for i , x in enumerate(tsRet["Today"]):
		if(abs(x)<0.0001):
			tsRet["Today"][i] = 0.0001
			
	for i in range(0,lags):
		tsRet["Lag%s" %str(i+1)]=tsLag["Lag%s" %str(i+1)].pct_change()*100
	
	tsRet["Direction"] = np.sign(tsRet["Today"])
	tsRet = tsRet[tsRet.index>=startDate]
	return tsRet

	
if __name__ =="__main__":
	print("Hit Rates/Confusion Matrices:\n")
	
	for i in range(0,10):
		year = 2017-i
		print("Year:%s" %year)
		snpret = createLaggedSeries("^GSPC",datetime.datetime(year-5,1,10),datetime.datetime(year,12,31),5)
		x = snpret[["Lag1","Lag2"]]
		y = snpret["Direction"]

		startTestDate = datetime.datetime(year,1,1)
		
		xTrain=x[x.index<startTestDate]
		yTrain = y[y.index<startTestDate]
		
		xTest =x[x.index>=startTestDate]
		yTest =y[y.index>=startTestDate]
		

		
		models=[("LR",LogisticRegression())]
		for m in models:
			m[1].fit(xTrain,yTrain)
			pred = m[1].predict(xTest)
			
			print("%s:\n%0.3f" % (m[0], m[1].score(xTest,yTest)))
			print("%s\n" % confusion_matrix(pred,yTest))
