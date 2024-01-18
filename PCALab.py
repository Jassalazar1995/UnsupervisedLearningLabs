import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Replace USArrests with your dataset
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df.columns = ['setosa', 'versicolor', 'virginica','pedal length']  # Rename according to your dataset

# Mean and Variance
print(df.mean())
print(df.var())

# PCA
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
pca = PCA()
pca.fit(df_scaled)

print(pca.mean_)
print(pca.explained_variance_ratio_)
print(pca.components_)

# Biplot
def biplot(score, coeff, labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex, ys * scaley)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1], color='r', alpha=0.5)
        if labels is None:
            plt.text(coeff[i,0]*1.15, coeff[i,1]*1.15, "Var"+str(i+1), color='g', ha='center', va='center')
        else:
            plt.text(coeff[i,0]*1.15, coeff[i,1]*1.15, labels[i], color='g', ha='center', va='center')
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()

biplot(pca.transform(df_scaled), np.transpose(pca.components_[0:2, :]), list(df.columns))
plt.show()

# Variance
pr_var = pca.explained_variance_
print(pr_var)

# Proportion of Variance Explained
pve = pr_var / sum(pr_var)
print(pve)

plt.plot(pve, 'o-')
plt.title('Proportion of Variance Explained')
plt.show()

# Cumulative Proportion of Variance Explained
plt.plot(np.cumsum(pve), 'o-')
plt.title('Cumulative Proportion of Variance Explained')
plt.show()
