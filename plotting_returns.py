import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv("/Users/brissseidavillarreal/Desktop/IAQF/MasterDataset.csv")
print(df)

# Cumulative Returns 
ndx_cumulative_ret = (1 + df["Return_NDX"]).cumprod() - 1
rgustl_cumulative_ret = (1 + df["Return_RGUSTL"]).cumprod() - 1

x = np.linspace(2008, 2023, 3777)

# Cumulative Returns Plot
plt.figure(figsize=(10, 8))
plt.plot(x, ndx_cumulative_ret, label = 'Nasdaq 100 Index')
plt.plot(x, rgustl_cumulative_ret, label = 'Russell 1000 Technology Index')
plt.title("Nasdaq 100 and Russell 1000 Tech Indexes Cumulative Returns")
plt.legend(fontsize="x-large")
plt.xlabel('Year')
plt.ylabel('Percentage Return')
plt.show()

# Train/Test Split Cumulative Returns
plt.figure(figsize=(10, 8))
plt.plot(x, ndx_cumulative_ret, label = 'Nasdaq 100 Index')
plt.plot(x, rgustl_cumulative_ret, label = 'Russell 1000 Technology Index')
plt.axvline(x=2018.166, color = 'red', label = "Train/Test Split")
plt.title("Nasdaq 100 and Russell 1000 Tech Indexes Cumulative Returns")
plt.legend(fontsize="x-large")
plt.xlabel('Year')
plt.ylabel('Percentage Return')
plt.show()
