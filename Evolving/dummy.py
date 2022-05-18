import numpy as np
import tqdm 
print(tqdm.__version__)
from sklearn.metrics import average_precision_score, roc_auc_score
from collections import OrderedDict

from xStream import xStream


with open('Data/Evolving-data/X-spam-sms-evolving') as f:
    lines = f.readlines()
stream_data = []
for line in lines:
  stream_data.append(line.split())

with open('Data/Evolving-data/y-spam-sms-evolving') as f:
    lines = f.readlines()
y = []
for line in lines:
  y.append(int(line))


k = 10
n_chains = 10
depth = 3
window_size = int(0.01 * len(y))
# window_size = 25

LRU_DIC = OrderedDict()
cf = xStream(LRU_DIC, num_components=k, n_chains=n_chains, depth=depth, window_size=window_size)  

y_scores = np.zeros((len(y)))

for i, sample in enumerate(tqdm.tqdm_notebook(stream_data)):

    print(i)
    cf.fit_partial(sample)
    if cf.ready_to_score:
      anomalyscore = -cf.score_partial(sample)

      #Uncomment depending on what dataset you are working on
      y_scores[int(sample[0])] = anomalyscore    
      # y_scores[i] = anomalyscore


ap = average_precision_score(y[window_size:], y_scores[window_size:]) 
auc = roc_auc_score(y[window_size:], y_scores[window_size:])
print("xstream: AP =", ap, "AUC =", auc)