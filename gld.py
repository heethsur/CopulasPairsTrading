from gldpy import GLD
import numpy as np
import pandas as pd
from scipy.special import gammaln
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns

# getting the data 
training = pd.read_csv(r'C:\Users\erick\Documents\IAQF-Competition\TrainingData.csv')
testing = pd.read_csv(r'C:\Users\erick\Documents\IAQF-Competition\TestingData.csv')

# creating dataframe
training_df = pd.DataFrame(training)
testing_df = pd.DataFrame(testing)

# Generalized lambda
# mu is the location paramter
# sigma is the scale parameter
# xi is the skewness parameter
# nu is the shape parameter

def pdf(mu,sigma,xi,nu,x):
    z = (x-mu)/sigma
    k = np.where(xi == 0, -np.log(nu) - z - np.exp(-z), np.where(xi>0, -np.log(nu)-z-np.power(1+xi*z, -1/xi), -np.log(nu) - z - np.power(1-xi*z, 1/xi)))
    p = np.exp(k-gammaln(nu) - gammaln(1/xi) - gammaln(nu + 1/xi))
    return p

def ppf(mu,sigma,xi,nu,q):
    k = np.power(-np.log(q), 1/nu)
    z = np.where(xi==0, -np.log(k), np.where(xi>0, (np.power(k,xi)-1)/xi, (1-np.power(k,-xi))/xi))
    return mu+sigma*z

def cdf(mu,sigma,xi,nu,x):
    z = (x-mu)/sigma
    k = np.where(xi == 0, np.exp(-np.exp(-z)), np.where(xi > 0, np.power(1 + xi*z, -1/xi), np.power(1 - xi*z, 1/xi)))
    p = np.exp(-np.power(k,nu))
    return p

# creating function to check parameters for gld implementation
def parameters(x):
    mean = np.mean(x)
    std = np.std(x)
    skew = ss.skew(x,nan_policy = 'omit')
    kurtosis = ss.kurtosis(x,nan_policy = 'omit')
    
    df = dict()
    df['Mean'] = mean
    df['STD'] = std
    df['Skew'] =  skew
    df['Kurtosis'] = kurtosis
    df = pd.DataFrame(df, index = ['parameters:'])
    return df


# creating data frames for different volatility
train_low = training_df[training_df['Cluster'] == 0]
train_medium = training_df[training_df['Cluster'] == 1]
train_high = training_df[training_df['Cluster'] == 2]

# function to calculate the cdf of NDX depending on volatility parameter
def ndx_vol_cdf(vol, x):
    if vol == 0:
        mean = np.mean(train_low['Return_NDX'])
        std = np.std(train_low['Return_NDX'])
        skew = ss.skew(train_low['Return_NDX'], nan_policy='omit')
        kurtosis = ss.kurtosis(train_low['Return_NDX'], nan_policy='omit')
        cdf_low = cdf(mean,std,skew,kurtosis,x)
        return cdf_low
    elif vol == 1:
        mean = np.mean(train_medium['Return_NDX'])
        std = np.std(train_medium['Return_NDX'])
        skew = ss.skew(train_medium['Return_NDX'], nan_policy='omit')
        kurtosis = ss.kurtosis(train_medium['Return_NDX'], nan_policy='omit')
        cdf_medium = cdf(mean,std,skew,kurtosis,x)
        return cdf_medium
    else:
        mean = np.mean(train_high['Return_NDX'])
        std = np.std(train_high['Return_NDX'])
        skew = ss.skew(train_high['Return_NDX'], nan_policy='omit')
        kurtosis = ss.kurtosis(train_high['Return_NDX'], nan_policy='omit')
        cdf_high = cdf(mean,std,skew,kurtosis,x)
        return cdf_high

# functions to get the cdf of RGUSTL depending on volatility parameter
def rgustl_vol_cdf(vol, x):
    if vol == 0:
        mean = np.mean(train_low['Return_RGUSTL'])
        std = np.std(train_low['Return_RGUSTL'])
        skew = ss.skew(train_low['Return_RGUSTL'], nan_policy='omit')
        kurtosis = ss.kurtosis(train_low['Return_RGUSTL'], nan_policy='omit')
        cdf_low = cdf(mean,std,skew,kurtosis,x)
        return cdf_low
    elif vol == 1:
        mean = np.mean(train_medium['Return_RGUSTL'])
        std = np.std(train_medium['Return_RGUSTL'])
        skew = ss.skew(train_medium['Return_RGUSTL'], nan_policy='omit')
        kurtosis = ss.kurtosis(train_medium['Return_RGUSTL'], nan_policy='omit')
        cdf_medium = cdf(mean,std,skew,kurtosis,x)
        return cdf_medium
    else:
        mean = np.mean(train_high['Return_RGUSTL'])
        std = np.std(train_high['Return_RGUSTL'])
        skew = ss.skew(train_high['Return_RGUSTL'], nan_policy='omit')
        kurtosis = ss.kurtosis(train_high['Return_RGUSTL'], nan_policy='omit')
        cdf_high = cdf(mean,std,skew,kurtosis,x)
        return cdf_high

# getting the cdf for both the NDX and RGUSTL depending on volatility parameter
ndx_train_cdf_vol0 = ndx_vol_cdf(0, training_df['Return_NDX'])
rgustl_train_cdf_vol0 = rgustl_vol_cdf(0, training_df['Return_RGUSTL'])
ndx_train_cdf_vol1 = ndx_vol_cdf(1, training_df['Return_NDX'])
rgustl_train_cdf_vol1 = rgustl_vol_cdf(1, training_df['Return_RGUSTL'])
ndx_train_cdf_vol2 = ndx_vol_cdf(2, training_df['Return_NDX'])
rgustl_train_cdf_vol2 = rgustl_vol_cdf(2, training_df['Return_RGUSTL'])

# creating a dataframe to store the joint cdf
joint_cdf = pd.DataFrame()
joint_cdf["NASDAQ CDF_0"] = ndx_train_cdf_vol0
joint_cdf["RUSSELL CDF_0"] = rgustl_train_cdf_vol0
joint_cdf["NASDAQ CDF_1"] = ndx_train_cdf_vol1
joint_cdf["RUSSELL CDF_1"] = rgustl_train_cdf_vol1
joint_cdf["NASDAQ CDF_2"] = ndx_train_cdf_vol2
joint_cdf["RUSSELL CDF_2"] = rgustl_train_cdf_vol2


# plottiing the empirical distributions for NASDAQ
fig, axs = plt.subplots(nrows=3,ncols=2,figsize=(7,7))
axs[0,0].hist(training_df['Return_NDX'], bins=50, alpha = .3)
axs[0,0].axis(xmin=-.03)
axs[0,0].set_title("Returns")
axs[0,0].grid()

axs[0,1].scatter(training_df["Return_NDX"],ndx_train_cdf_vol0, alpha = .3, label = "Vol = 0")
axs[0,1].set_title("CDF")
axs[0,1].legend()
axs[0,1].grid()

axs[1,0].hist(training_df['Return_NDX'], bins=50, alpha = .3)
axs[1,0].axis(xmin=-.03)
axs[1,0].grid()

axs[1,1].scatter(training_df["Return_NDX"],ndx_train_cdf_vol1, alpha = .3, label = "Vol = 1")
axs[1,1].legend()
axs[1,1].grid()

axs[2,0].hist(training_df['Return_NDX'], bins=50, alpha = .3)
axs[2,0].axis(xmin=-.03)
axs[2,0].grid()

axs[2,1].scatter(training_df["Return_NDX"],ndx_train_cdf_vol2, alpha = .3, label = "Vol = 2")
axs[2,1].legend()
axs[2,1].grid()

fig.suptitle("NASDAQ Index")
plt.show()

# plotting the empirical distributions for RUSSEL
fig, axs = plt.subplots(nrows=3,ncols=2,figsize=(7,7))
axs[0,0].hist(training_df['Return_RGUSTL'], bins=50, alpha = .3)
axs[0,0].axis(xmin=-.03)
axs[0,0].set_title("Returns")
axs[0,0].grid()

axs[0,1].scatter(training_df["Return_RGUSTL"],rgustl_train_cdf_vol0, alpha = .3, label = "Vol = 0")
axs[0,1].set_title("CDF")
axs[0,1].legend()
axs[0,1].grid()

axs[1,0].hist(training_df['Return_RGUSTL'], bins=50, alpha = .3)
axs[1,0].axis(xmin=-.03)
axs[1,0].grid()

axs[1,1].scatter(training_df["Return_RGUSTL"],rgustl_train_cdf_vol1, alpha = .3, label = "Vol = 1")
axs[1,1].legend()
axs[1,1].grid()

axs[2,0].hist(training_df['Return_RGUSTL'], bins=50, alpha = .3)
axs[2,0].axis(xmin=-.03)
axs[2,0].grid()

axs[2,1].scatter(training_df["Return_RGUSTL"],rgustl_train_cdf_vol2, alpha = .3, label = "Vol = 2")
axs[2,1].legend()
axs[2,1].grid()

fig.suptitle("Russell Index")
plt.show()

# plotting the joint cdf of both NASDAQ and RUSSELL
plot = sns.jointplot(data = joint_cdf,x = joint_cdf['NASDAQ CDF_0'], y = joint_cdf['RUSSELL CDF_0'], dropna = True)
plot.fig.suptitle("Joint CDF of Indexes for Volatilty = 0")
plot.fig.subplots_adjust(top=0.95)
plt.show()

plot = sns.jointplot(data = joint_cdf,x = joint_cdf['NASDAQ CDF_1'], y = joint_cdf['RUSSELL CDF_1'], dropna = True)
plot.fig.suptitle("Joint CDF of Indexes for Volatilty = 1")
plot.fig.subplots_adjust(top=0.95)
plt.show()

plot = sns.jointplot(data = joint_cdf,x = joint_cdf['NASDAQ CDF_2'], y = joint_cdf['RUSSELL CDF_2'], dropna = True)
plot.fig.suptitle("Joint CDF of Indexes for Volatilty = 2")
plot.fig.subplots_adjust(top=0.95)
plt.show()