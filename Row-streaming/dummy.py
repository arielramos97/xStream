import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.datasets import load_svmlight_file

from xStream import xStream
   
data = load_svmlight_file("Data/streaming-data/spam-sms")
X = data[0]
y = data[1]
X= X.todense()



window_size = int(0.01 * len(y))
k = 10
n_chains = 10
depth = 3

cf = xStream(num_components=k, n_chains=n_chains, depth=depth, window_size=window_size) 

all_scores = []

for i, sample in enumerate(tqdm.tqdm_notebook(X)):
  cf.fit_partial(sample.A1)
  if i>=window_size:
    anomalyscore = -cf.score_partial(sample.A1)
    all_scores.append(anomalyscore)

ap = average_precision_score(y[window_size:window_size+len(all_scores)], all_scores) 
auc = roc_auc_score(y[window_size:window_size+len(all_scores)], all_scores)
print("xstream: AP =", ap, "AUC =", auc)
