import numpy as np
from Chain import Chain

class Chains:
    
    '''
    Ensemble of Chains

    Parameters
    ----------
    n_chains
        Number of chains in the ensemble. Set to 100 by default.
    depth
        Number of feature splits to be performed. Set to 25 by default.
    chains
        Array grouping all the chains.

    '''
    
    def __init__(self, deltamax, n_chains=100, depth=25):

        self.n_chains = n_chains
        self.depth = depth
        self.chains = []

        for i in range(self.n_chains):
          c = Chain(deltamax, depth=self.depth)
          self.chains.append(c)

    def fit(self, X, update=False):
              
      for i, c in enumerate(self.chains):
          # print('Fitting chain: ', i)
          c.fit(X, update)

    def score(self, X, id=None):
        scores = np.zeros(X.shape[0])
        for i, c in enumerate(self.chains):
            c_score = c.score(X)
            # print('Chain ', i, ' score: , ', c_score)
            scores += c_score

        scores /= float(self.n_chains)

        
        return scores
    
    def next_window(self):
        for c in self.chains:
          c.next_window()
    
    def set_deltamax(self, deltamax):
        for c in self.chains:
            c.deltamax = deltamax
            c.shift = c.rand_arr * deltamax