import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import scipy.stats as stats
import pandas as pd
import scipy

trch = 1
trrep = 0
tstrep = 0
trintensity = 0
tstintensity = 0
scores2plot = []
column_names = []
label = []
for trch in [0, 1]:
    for trintensity in [1,2]:
        for tstintensity in [1, 2]:
            label.append(f"trch_{trch}-trintensity_{trintensity}")
            filepath = Path(f'../results/ravness_results_new/trch_{trch}_trrep_{trrep}_'
                       f'trintensity_{trintensity}_tstrep_{tstrep}_tstintensity_{tstintensity}')
            with open(filepath / 'data.pkl', 'rb') as ofile:
                data = pickle.load(ofile)

            scorestot = [data["scorestot"]["test_score"], data["scorestot"]["train_score"]]
            test_scores = data["test_score"]
            preds = data["preds"]
            scorestot = np.mean(scorestot, axis=1)
            test_scores = np.array(test_scores)
            preds = np.array(preds)
            scores2plot.append(test_scores)
            column_names.append(f'tr_channel_{trch}-tr_intensity_{trintensity}')

scores2plot =  np.array(scores2plot).transpose()
from scipy.stats import pearsonr
df = pd.DataFrame(scores2plot)
rho = df.corr()
pval = df.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)
p = pval.applymap(lambda x: ''.join(['*' for t in [.05, .01, .001] if x<=t]))
print(rho.round(2).astype(str) + p)
for j in range(8):
    for i in range(8):
        print(stats.f_oneway(scores2plot[:, j], scores2plot[:, i]))
plt.scatter(scores2plot[:, 2], scores2plot[:, 0] )
plt.show()
plt.figure()
plt.violinplot(scores2plot, showextrema=False, showmeans=True)
[plt.plot(np.ones((24, 1))+j, scores2plot[:, j], 'o', label=label[j]) for j in range(scores2plot.shape[1])]
plt.legend()
plt.show()
