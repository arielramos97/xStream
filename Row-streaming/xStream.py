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
            num_components=100,
            n_chains=100,
            depth=25,
            window_size=25):
      
        self.streamhash = StreamhashProjection(n_components=num_components,
                                              density=1/3.0,
                                              random_state=42)
        
        deltamax = np.ones(num_components) * 0.5

        deltamax[np.abs(deltamax) <= 0.0001] = 1.0
        self.window_size = window_size

        self.chains = Chains(
            deltamax=deltamax,
            n_chains=n_chains,
            depth=depth)

        self.step = 0
        self.cur_window = []
        self.ref_window = None

    def fit_partial(self, X, y=None):
        """Fits the model to next instance.
        """
        self.step += 1

        X = self.streamhash.fit_transform_partial(X)

        X = X.reshape(1, -1)
        self.cur_window.append(X)
        self.chains.fit(X)

        if self.step % self.window_size == 0:
            self.ref_window = self.cur_window
            self.cur_window = []
            deltamax = self._compute_deltamax()
            self.chains.set_deltamax(deltamax)
            self.chains.next_window()

        return self

    def score_partial(self, X):
        """Scores the anomalousness of the next instance.
        """
        X = self.streamhash.fit_transform_partial(X)
        X = X.reshape(1, -1)
        score = self.chains.score(X).flatten()

        return score

    def _compute_deltamax(self):

        mx = np.max(np.concatenate(self.ref_window, axis=0), axis=0)
        mn = np.min(np.concatenate(self.ref_window, axis=0), axis=0)

        deltamax = (mx - mn) / 2.0

        deltamax[np.abs(deltamax) <= 0.0001] = 1.0

        return deltamax