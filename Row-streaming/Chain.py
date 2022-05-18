import numpy as np

class Chain:

    '''
    Individual Chain

    Method to estimate density at multiple scales 
    The chain approximates the density of a point by counting its nearby neighbors at multiple scales. 
    For every scale or level, a count-min-sketch approximates the bin-counts at that level.
    Non-stationarity of data is handled by maintaining separate bin-counts for an alternating pair of windows
    containing ψ points each, termed as current and reference windows. 
    
    Parameters
    ----------
    k
        Number of components or random projections.
    deltamax
        List of bin-widths corresponding to half the range of the projected data.
    depth
        Number of feature splits to be performed. Set to 25 by default.
    fs
        List containing the randomly selected split features or dimensions.
    cmsketches_ref
        Reference count-min-sketches corresponding to the reference window.
    cmsketches_cur
        Current count-min-sketches corresponding to the current window.
    rand_arr
        List of uniform random numbers used to compute the shift values.
    shift
        List containing the uniform shift value for every component.
    is_first_window
        Boolean value indicating whether the window under consideration is the first one or not.
    '''

    def __init__(self, deltamax, depth=25):
        k = len(deltamax)
        self.deltamax = deltamax # feature ranges
        self.depth = depth
        self.fs = [np.random.randint(0, k) for d in range(depth)]
        self.cmsketches_ref = [{} for i in range(depth)] * depth
        self.cmsketches_cur = [{} for i in range(depth)] * depth
        self.rand_arr = np.random.rand(k)
        self.shift = self.rand_arr * deltamax
        self.is_first_window = True

    def fit(self, X, verbose=False):#, update=False):
        prebins = np.zeros(X.shape, dtype=float)
        depthcount = np.zeros(len(self.deltamax), dtype=int)
        for depth in range(self.depth):
            f = self.fs[depth]
            depthcount[f] += 1

            if depthcount[f] == 1:
                prebins[:,f] = (X[:,f] + self.shift[f])/self.deltamax[f]
            else:
                prebins[:,f] = 2.0*prebins[:,f] - self.shift[f]/self.deltamax[f]

            if self.is_first_window:
              cmsketch = self.cmsketches_ref[depth] 
              for prebin in prebins:
                  l = tuple(np.floor(prebin).astype(int))
                  if not l in cmsketch:
                      cmsketch[l] = 0
                  cmsketch[l] += 1

              self.cmsketches_ref[depth] = cmsketch
              self.cmsketches_cur[depth] = cmsketch

            else:
              cmsketch = self.cmsketches_cur[depth] 
              for prebin in prebins:
                  l = tuple(np.floor(prebin).astype(int))
                  if not l in cmsketch:
                      cmsketch[l] = 0
                  cmsketch[l] += 1
              
              self.cmsketches_cur[depth] = cmsketch

        return self

    def bincount(self, X):
        scores = np.zeros((X.shape[0], self.depth))
        prebins = np.zeros(X.shape, dtype=float)
        depthcount = np.zeros(len(self.deltamax), dtype=int)
        for depth in range(self.depth):
            f = self.fs[depth] 
            depthcount[f] += 1

            if depthcount[f] == 1:
                prebins[:,f] = (X[:,f] + self.shift[f])/self.deltamax[f]
            else:
                prebins[:,f] = 2.0*prebins[:,f] - self.shift[f]/self.deltamax[f]

            cmsketch = self.cmsketches_ref[depth]
            for i, prebin in enumerate(prebins):
                l = tuple(np.floor(prebin).astype(int))
                if not l in cmsketch:
                    scores[i,depth] = 0.0
                else:
                    scores[i,depth] = cmsketch[l]

        return scores

    def score(self, X, adjusted=False):
        # scale score logarithmically to avoid overflow:
        #    score = min_d [ log2(bincount x 2^d) = log2(bincount) + d ]
        scores = self.bincount(X)
        depths = np.array([d for d in range(1, self.depth+1)])
        scores = np.log2(1.0 + scores) + depths # add 1 to avoid log(0)
        return np.min(scores, axis=1)

    def next_window(self):
        self.is_first_window = False
        self.cmsketches_ref = self.cmsketches_cur
        self.cmsketches_cur = [{} for _ in range(self.depth)] * self.depth