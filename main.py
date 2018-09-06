import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as mpl
import scipy.stats as sps
import math

#Student Details
print("\n"+"UBitName = ss623")
print("personNumber = 50247317")

Path=os.path.dirname(__file__)
relativepath='DataSet\\university data.xlsx'
finalpath=os.path.join(Path,relativepath)
df = pd.read_excel(finalpath)
x1=df.iloc[0:-1,2]
x2=df.iloc[0:-1,3]
x3=df.iloc[0:-1,4]
x4=df.iloc[0:-1,5]

#calculating mean of variables
mu1=np.mean(x1)
print ("\n"+"mu1 = " +str(round(mu1,2)))
mu2=np.mean(x2)
print ("mu2 = " +str(round(mu2,2)))
mu3=np.mean(x3)
print ("mu3 = " +str(round(mu3,2)))
mu4=np.mean(x4)
print ("mu4 = " +str(round(mu4,2)))

#calculating variance of variables
var1=np.var(x1)
print("\n"+"var1 = " +str(round(var1,2)))
var2=np.var(x2)
print("var2 = " +str(round(var2,2)))
var3=np.var(x3)
print("var3 = " +str(round(var3,2)))
var4=np.var(x4)
print("var4 = " +str(round(var4,2)))

#calculating standard deviation of variables
sigma1=np.std(x1)
print("\n"+"sigma1 = " +str(round(sigma1,2)))
sigma2=np.std(x2)
print("sigma2 = " +str(round(sigma2,2)))
sigma3=np.std(x3)
print("sigma3 = " +str(round(sigma3,2)))
sigma4=np.std(x4)
print("sigma4 = " +str(round(sigma4,2)))

#calculating covariance
rounder= lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind': rounder})
rr = df.iloc[0:-1,2:6].T
covarianceMat=np.cov(rr)
print("\n"+"covarianceMat =")
print(covarianceMat)

#calculating correlation
correlationMat=np.corrcoef(rr)
print("\n"+"correlationMat =")
print(correlationMat)

#plotting graph for variables
print("\n"+"PLOTS OF PAIRWISE DATA:")
mpl.scatter(x1,x2)
mpl.xlabel("US Score(US News)")
mpl.ylabel("Research Overhead")
mpl.show()

mpl.scatter(x1,x3)
mpl.xlabel("US Score(US News)")
mpl.ylabel("Admin Base Pay$")
mpl.show()

mpl.scatter(x1,x4)
mpl.xlabel("US Score(US News)")
mpl.ylabel("Tution(out-state)$")
mpl.show()
 
mpl.scatter(x2,x3)
mpl.xlabel("Research Overhead")
mpl.ylabel("Admin Base Pay$")
mpl.show()

mpl.scatter(x2,x4)
mpl.xlabel("Research Overhead")
mpl.ylabel("Tution(out-state)$")
mpl.show()

mpl.scatter(x3,x4)
mpl.xlabel("Admin Base Pay$")
mpl.ylabel("Tution(out-state)$")
mpl.show()

print("Most correlated pair: CS Score (USNews) and Research Overhead %")
print("Least correlated pair: CS Score (USNews) and Admin Base Pay$"+"\n")

#calculating univariate logLikelihood
uv1=sum([math.log(sps.norm.pdf(x,loc=mu1,scale=sigma1)) for x in x1])
uv2=sum([math.log(sps.norm.pdf(x,loc=mu2,scale=sigma2)) for x in x2])
uv3=sum([math.log(sps.norm.pdf(x,loc=mu3,scale=sigma3)) for x in x3])
uv4=sum([math.log(sps.norm.pdf(x,loc=mu4,scale=sigma4)) for x in x4])
logLikelihood_univariate=uv1+uv2+uv3+uv4
print("logLikelihood = " + str(round(logLikelihood_univariate,2)))

#calculating multivariate logLikelihood
meanarray=np.array([mu1,mu2,mu3,mu4])
mpdf=sps.multivariate_normal(mean=meanarray,cov=covarianceMat,allow_singular=True)
logLikelihood=0
for i in range (0,49):
    x=np.array([df.iloc[i,2:6]])
    xx=np.asmatrix(x)
    mpdf1=mpdf.pdf(x)
    logLikelihood+=math.log(mpdf1)    
print("logLikelihood_multivariate= " + str(round(logLikelihood,2)))


    
