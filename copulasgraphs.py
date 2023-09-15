#produce copula illustrations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

sns.set_style("darkgrid")
sns.mpl.rc("figure", figsize=(8, 8))

from statsmodels.distributions.copula.api import (
    CopulaDistribution, GumbelCopula, IndependenceCopula, )

copula = GumbelCopula(theta=2)
_ = copula.plot_pdf()  # returns a matplotlib figure

sample = copula.rvs(10000)
h = sns.jointplot(x=sample[:, 0], y=sample[:, 1], kind="hex")
_ = h.set_axis_labels("X1", "X2", fontsize=16)