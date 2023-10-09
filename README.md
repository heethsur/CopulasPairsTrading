# CopulasPairsTrading

Implementation of a pairs trading strategy between two closely correlated equity indices using a flipped Gumbel copula to model joint probabilities of the pair's returns. 
Short term divergence in spread is probabilistically determined using the copula. A simulataneous long-short position is taken in the respective undervalued and overvalued security in the pair to produce alpha, until the spread converges back to a long run average prompting exit. 

IAQF_paper.pdf is a detailed white-paper covering the implementation and testing of this strategy using two highly correlated equity indices. 
Code implementation of the copula construction and trade signals can be found in copulas.py



