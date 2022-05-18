import pandas as pd
import numpy as np
import math
import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score
from itertools import chain
import statistics


from xStream import xStream




   
    
df = pd.read_excel (r'Mammography.xlsx', sheet_name='Sheet1')
y = ((df.iloc[:,-1:]).values.tolist())
y = list(chain(*y))
x = (df.iloc[:,0:6]).values.tolist()
X = np.array(x)
X = np.squeeze(X)
y= np.array(y)
y = np.squeeze(y)


''' 
window_size_percentage: is a percentage of the rows which will be used as a window size. 
Actual window size is equal to window_size
     '''

# window_size_percentage = 0.01
# window_size_percentage = 0.05
#window_size_percentage = 0.1
window_size_percentage = 0.25
window_size = math.floor(window_size_percentage*X.shape[0])
    
k = 10
n_chains = 10
depth = 3

cf = xStream(num_components=k, n_chains=n_chains, depth=depth, window_size=window_size) 


all_scores = []
average_precisions_per_window = []
scores_per_window = []
window_size_counter = 0
counter_window = 0

for i, sample in enumerate(tqdm.notebook.tqdm(X)): 
    cf.fit_partial(sample)
    if i>=window_size:
        anomalyscore = -cf.score_partial(sample)
        all_scores.append(anomalyscore)

        window_size_counter= window_size_counter + 1
        scores_per_window.append(anomalyscore)

        if window_size_counter == window_size:
            counter_window = counter_window + 1
            y_per_window = y[counter_window*window_size:((counter_window*window_size)+window_size+1)]
            len_y = len(y_per_window)
            len_scores = len(scores_per_window)
            if(len_y>len_scores):
              y_per_window=y_per_window[0:len_scores]
            elif(len_scores>len_y):
              scores_per_window = scores_per_window[0:len_y]
            average_precisions_per_window.append(average_precision_score(y_per_window,scores_per_window))
            window_size_counter = 0
            scores_per_window = []

y = y[window_size:]
std = statistics.stdev(average_precisions_per_window)
oap = average_precision_score(y, all_scores) 
auc = roc_auc_score(y, all_scores)
map = sum(average_precisions_per_window) / len(average_precisions_per_window)

print("XStream results for window size as a percentage of rows equal to",window_size_percentage*100,"% :\n",
  "OAP = ", oap, "\n AUC = ", auc, "\n MAP = ", map, "with a standard deviation equal to: ",std )