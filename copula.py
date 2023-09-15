import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
#import sympy as sp
#from sympy import symbols, diff

#import test data with GLD CDF values calculated in the GLD section    
test_df = pd.read_excel(r'C:\Users\heeth\Github\FM5252\Testing.xlsx')

def theta(rho): #calculates theta given Kendall's Tau
    return 1/(1 - rho)

def generator(t,theta): #Generator for Gumbel
    return np.exp(-t**(1/theta))

def inverse_generator(t,theta): #Inverse generator for Gumbel
    return (-np.log(t))**(theta)

def gumbel_copula(u,v,theta): #Gumbel copula function
    return generator(inverse_generator(u,theta) + inverse_generator(v,theta), theta)

def flipped_gumbel(u,v,theta): #flipped Gumbel function
    return u + v - 1 + gumbel_copula(1-u,1-v,theta)

#sympy differentiation. Commented out to not interfere with numpy package
#u, v, t = symbols('u v t', real=True)
#f = u + v - 1 + sp.exp(-(((-sp.ln(1-u))**t + (-sp.ln(1-v))**t)**(1/t)))
#print(diff(f, v))

#compute probabilities using partial derivatives for both gumbel and flipped gumbel
def u_given_v(u,v,theta):
    return (-np.log(1 - v))**theta*((-np.log(1 - u))**theta + (-np.log(1 - v))**theta)**(1/theta)*np.exp(-((-np.log(1 - u))**theta + (-np.log(1 - v))**theta)**(1/theta))/((1 - v)*((-np.log(1 - u))**theta + (-np.log(1 - v))**theta)*np.log(1 - v)) + 1

def v_given_u(u,v,theta):
    return (-np.log(1 - u))**theta*((-np.log(1 - u))**theta + (-np.log(1 - v))**theta)**(1/theta)*np.exp(-((-np.log(1 - u))**theta + (-np.log(1 - v))**theta)**(1/theta))/((1 - u)*((-np.log(1 - u))**theta + (-np.log(1 - v))**theta)*np.log(1 - u)) + 1

def u_given_v_gumbel(u,v,theta):
    return -(-np.log(v))**theta*((-np.log(u))**theta + (-np.log(v))**theta)**(1/theta)*np.exp(-((-np.log(u))**theta + (-np.log(v))**theta)**(1/theta))/(v*((-np.log(u))**theta + (-np.log(v))**theta)*np.log(v))


def v_given_u_gumbel(u,v,theta):
    return -(-np.log(u))**theta*((-np.log(u))**theta + (-np.log(v))**theta)**(1/theta)*np.exp(-((-np.log(u))**theta + (-np.log(v))**theta)**(1/theta))/(u*((-np.log(u))**theta + (-np.log(v))**theta)*np.log(u))

#produce trade signals
def trade_signal(u,v, theta, c):
    if (u_given_v(u,v,theta) >= c) & (v_given_u(u,v,theta) <= (1 - c)):
        return "SHORT"
    elif (u_given_v(u,v,theta) <= (1 - c)) & (v_given_u(u,v,theta) >= c):
        return "LONG"
    elif (u_given_v(u,v,theta) < 0.5) & (v_given_u(u,v,theta) > 0.5):
        return "SHORT EXIT"
    elif (u_given_v(u,v,theta) > 0.5) & (v_given_u(u,v,theta) < 0.5):
        return "LONG EXIT"
    else:
        return "NEUTRAL"
    

############################################################################################
# Unit tests
print(theta(0.5)==2)
print(round(generator(0.7,2),2)==0.43)        
print(round(inverse_generator(0.7,2),2)==0.13) 
print(round(gumbel_copula(0.4,0.7,2),2) == 0.37)
print(round(flipped_gumbel(0.9,0.7,2),2) == 0.67)
print(trade_signal(0.4,0.7,2,0.95))
############################################################################################

#Implement trade signals on test dataset

thetA = theta(0.80679381)

#populate conditionals
p_u = pd.DataFrame(columns = ['U|V'])
for i in range(len(test_df)):
    p_u.loc[i] = [u_given_v(test_df.iloc[i,7],test_df.iloc[i,8],thetA)]
test_df = pd.concat([test_df, p_u], axis=1)

p_v = pd.DataFrame(columns = ['V|U'])
for i in range(len(test_df)):
    p_v.loc[i] = [v_given_u(test_df.iloc[i,7],test_df.iloc[i,8],thetA)]
test_df = pd.concat([test_df, p_v], axis=1)


#Compute trading signals for each day
s = pd.DataFrame(columns = ['Signal'])
for i in range(len(test_df)):
    s.loc[i] = [trade_signal(test_df.iloc[i,7],test_df.iloc[i,8],thetA,0.95)]
test_df = pd.concat([test_df, s], axis=1)

#filter non entry/exit days
test_df = test_df[test_df['Signal'] != "NEUTRAL"]
test_df = test_df.reset_index(drop = True)

#flter out repeated duplicate signals
transactions = pd.DataFrame(columns = ['Date','Last Price_RGUSTL', 'Return_RGUSTL','Last Price_NDX','Return_NDX', 'VIX', 'Cluster', 'RGUSTL_CDF', 'NDX_CDF','U|V','V|U', 'Signal'])
for i in range(1,len(test_df)):
    if test_df['Signal'][i] != test_df['Signal'][i-1]:
        transactions.loc[i] = test_df.loc[i]

transactions.loc[0] = test_df.loc[0]  

transactions = transactions.reset_index(drop=True)

transactions.to_excel("Flipped_Gumbel_Signals.xlsx")

#Entry-Exit transactions matched in Excel

transactions = pd.read_csv(r'C:\Users\heeth\Github\FM5252\Flipped_Gumbel_Transactions.csv')


# calculate returns for every transaction (long or short)

transactions['r'] = 'abc'

for i in range(0,len(transactions)-1):
   if transactions.iloc[i,11]== 'LONG':
        r1 = (transactions.iloc[i+1,1] - transactions.iloc[i,1])/transactions.iloc[i,1]
        r2 = (transactions.iloc[i,3] - transactions.iloc[i+1,3])/transactions.iloc[i,3]
        transactions['r'][i] = 0.5*r1 + 0.5*r2
   elif transactions.iloc[i,11] == 'SHORT':
        r1 = (transactions.iloc[i,1] - transactions.iloc[i+1,1])/transactions.iloc[i,1]
        r2 = (transactions.iloc[i+1,3] - transactions.iloc[i,3])/transactions.iloc[i,3]
        transactions['r'][i] = 0.5*r1 + 0.5*r2


transactions = transactions.loc[(transactions['Signal'] != 'LONG EXIT') & (transactions['Signal'] != 'SHORT EXIT')]
transactions['cum_r'] = np.cumprod(1+transactions['r'])
transactions.to_excel("Flipped_Gumbel_Returns.xlsx")


#repeat for Gumbel

