import numpy as np

from StreamhashProjection import StreamhashProjection
from Chains import Chains



class xStream:

    '''
    XStream Algorithm

    Parameters
    ----------
    streamhash
        StreamhashProjection class object.
    deltamax
        List of bin-widths corresponding to half the range of the projected data.
    window_size
        Number of points to observe before replacing the counts in the reference window by those of the current window.
    chains
        Chains class object.
    step
        Counter for the number of points observed.
    cur_window
        Bin-counts for the current window.
    ref_window
        Bin-counts for the reference window.
    '''

    def __init__(
            self,
            cache,
            num_components=100,
            n_chains=100,
            depth=25,
            window_size=25,
            ):
      
        self.streamhash = StreamhashProjection(n_components=num_components, density=1/3.0, random_state=42)
        
        deltamax = np.ones(num_components) * 0.5

        deltamax[np.abs(deltamax) <= 0.0001] = 1.0
        self.window_size = window_size

        self.chains = Chains(
            deltamax=deltamax,
            n_chains=n_chains,
            depth=depth)

        self.step = 0     #num points
        self.cur_window = []
        self.ref_window = None
        self.ready_to_score = False

        self.cache = cache

      

    def fit_partial(self, X, y=None):
        """Fits the model to next instance.
        """

        self.is_update = False

        #X is incoming update
        id = int(X[0])

        if id not in list(self.cache.keys()):

          if self.step == self.window_size :
            self.step = 0
            self.ref_window = self.cur_window
            self.cur_window = []
            deltamax = self._compute_deltamax()
            self.chains.set_deltamax(deltamax)
            self.chains.next_window()
            self.ready_to_score = True
          
          self.step += 1
        else:
          #Remove (id, yid) from cache
          yid = self.cache[id]
          del self.cache[id]
            
          #Remove yid from current counters of C - Proceeds exactly as addition, with the bin-count being decremented instead
          self.is_update = True
          self.chains.fit(yid, update=True)

        #Process income: get feature names and feature values
        names, values = self.get_value_labels(X)

        #Get projection
        X = self.streamhash.fit_transform_partial(np.array(values), feature_names=names)
        X = X.reshape(1, -1)

        if self.is_update:
          X = np.add(X, yid)

        self.cache[id] = X
        
        self.cur_window = list(self.cache.values())[-self.window_size:]

        #Add yid to the current counter of C
        self.chains.fit(X)

        return self
    
    def print_ids_cache(self, cache):
      return [i[0] for i in cache]

    
    def get_value_labels(self, X):
 
      names = []
      values = []
      for i, update in enumerate(X[1:]):
          feature_value = update.split(":")
          names.append(feature_value[0])
          values.append(int(feature_value[1]))
        
      return names, values

    def score_partial(self, X):
        """Scores the anomalousness of the next instance.
        """
        #Process income: get feature names and feature values
        # names, values = self.get_value_labels(X)
        # X = self.streamhash.fit_transform_partial(np.array(values), feature_names=names)
        
        id = int(X[0])

        X = self.cache[id]
        X = X.reshape(1, -1)

        score = self.chains.score(X).flatten()

        return score

    def _compute_deltamax(self):

        mx = np.max(np.concatenate(self.ref_window, axis=0), axis=0)
        mn = np.min(np.concatenate(self.ref_window, axis=0), axis=0)

        deltamax = (mx - mn) / 2.0
        deltamax[np.abs(deltamax) <= 0.0001] = 1.0
        return deltamax