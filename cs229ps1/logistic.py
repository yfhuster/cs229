import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_X = pd.read_csv('data/logistic_x.txt', delim_whitespace=True, header=None, engine='python')
ys = pd.read_csv('data/logistic_y.txt', delim_whitespace=True, header=None, engine='python')
ys = ys.astype(int)

df_X['label'] = ys[0]

# print(df_X)

Xs = df_X[[0, 1]].to_numpy()

Xs = np.hstack([np.ones((Xs.shape[0], 1)), Xs])
ys = df_X['label'].to_numpy()

# begin to train
theta = np.zeros(Xs.shape[1])
diff = 1e8

# for demonstration, not necessary
all_thetas = []
num_iter = 0

# train to the specific accuracy such as 1e-6
while diff > 1e-6:
    zs = ys*Xs.dot(theta)
    gzs = 1 / (1 + np.exp(-zs))
    # print(gzs.shape, ys.shape, Xs.shape)
    derivative = ((gzs - 1) * ys).reshape((-1, 1)) * Xs
    derivative = np.mean(derivative, axis=0)
    # print(derivative.shape)

    # Hessian
    # more efficient way to calculate Hessian
    Hessian = np.zeros((theta.shape[0], theta.shape[0]))
    # print(Hessian.shape)
    for i in range(Hessian.shape[0]):
        for j in range(Hessian.shape[1]):
            if i <= j:
                Hessian[i, j] = np.mean(gzs * (1 - gzs) * Xs[:, i] * Xs[:, j])
            else:
                Hessian[i, j] = Hessian[j, i]
    # print(Hessian)
    delta = np.linalg.inv(Hessian).dot(derivative)
    old_theta = theta.copy()
    theta -= delta

    # demonstration part
    all_thetas.append(theta.copy())
    num_iter += 1
    diff = np.sum(np.abs(theta - old_theta))

print("Converge after {} iterations".format(num_iter))

# plot
ax = plt.axes()
df_X.query("label == -1").plot.scatter(x=0, y=1, ax=ax, color='blue')
df_X.query("label == 1").plot.scatter(x=0, y=1, ax=ax, color='red')

_xs = np.array([np.min(Xs[:, 1]), np.max(Xs[:, 1])])
for k, his_theta in enumerate(all_thetas):
    _ys = (his_theta[0] + his_theta[1] * _xs) / (-his_theta[2])
    plt.plot(_xs, _ys, label='iter{}'.format(k + 1), lw=0.5)
plt.legend(bbox_to_anchor=(0.94, 1.17))
plt.show()
