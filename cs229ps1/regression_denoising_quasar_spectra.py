import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def lwr(x, wave, y_sample, tau=5):
    pre_sample = []
    for j, xj in enumerate(wave):
        # 5, (1, 10, 100, 1000)
        w = np.diag(np.exp(-1 / (2 * tau ** 2) * (wave - xj) ** 2))
        theta_point = np.linalg.inv(x.T.dot(w).dot(x)).dot(x.T).dot(w).dot(y_sample)
        pre_sample.append(theta_point.dot(x[j, :]))
    return pre_sample


def ker(t):
    return np.max(1 - t, 0)


df_train = pd.read_csv("data/quasar_train.txt")
df_test = pd.read_csv("data/quasar_test.txt")

# obtain the labels of columns
cols_train = df_train.columns.values.astype(float).astype(int)
# print(cols_train.shape)
cols_test = df_test.columns.values.astype(float).astype(int)
# print(cols_test.shape)

assert (cols_train == cols_test).all()
df_train.columns = cols_train
df_test.columns = cols_test
wave_lens = cols_train

print(df_test.head())

plt.plot(wave_lens, df_train.loc[0])
# plt.show()

x0 = np.ones(wave_lens.shape[0])
# 有点奇怪为什么(450,)和(450,)用vs堆叠成了(2, 450)
X = np.vstack([x0, wave_lens]).T
y = df_train.loc[0].values
print("shape of X: {}\n".format(X.shape), "shape of y: {}".format(y.shape))

theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# plt.plot(wave_lens, X.dot(theta))
# plt.show()

predicts = []
for i, xi in enumerate(wave_lens):
    # 5, (1, 10, 100, 1000)
    Wi = np.diag(np.exp(-1/(2*5**2) * (wave_lens-xi)**2))
    theta_i = np.linalg.inv(X.T.dot(Wi).dot(X)).dot(X.T).dot(Wi).dot(y)
    predicts.append(theta_i.dot(X[i, :]))

plt.plot(wave_lens, predicts)
# plt.show()


pres_train = []
for index, row in df_train.iterrows():
    print(index)
    y = row.values
    predict_sample = lwr(X, wave_lens, y, tau=5)
    pres_train.append(predict_sample)

df_smoothed_train = pd.DataFrame(pres_train, columns=df_train.columns)
print(df_smoothed_train.shape)

pres_test = []
for index, row in df_test.iterrows():
    print(index)
    y = row.values
    predict_sample = lwr(X, wave_lens, y, tau=5)
    pres_test.append(predict_sample)

df_smoothed_test = pd.DataFrame(pres_test, columns=df_test.columns)
print(df_smoothed_test.shape)

num_neighbor = 3
wave_left = wave_lens[wave_lens < 1200]
wave_right = wave_lens[wave_lens >= 1300]
df_smoothed_right = df_smoothed_train[wave_right]
df_smoothed_left = df_smoothed_train[wave_left]
f_left_pre = []
errors = []
for k, row in df_smoothed_right.iterrows():
    dist = ((df_smoothed_right - row) ** 2).sum(axis=1)
    dis_max = dist.max()
    dis_nei = dist.sort_values()[:num_neighbor]
    # confused
    a = np.sum([ker(d/dis_max) * df_smoothed_left.loc[idx] for (idx, d) in dis_nei.iteritems()], axis=0)
    b = np.sum([ker(d/dis_max) for (idx, d) in dis_nei.iteritems()], axis=0)
    f_left_hat = a / b
    f_left_pre.append(f_left_hat)
    error = np.sum((f_left_hat - df_smoothed_left.loc[k]) ** 2)
    errors.append(error)

avg_error = np.mean(errors)
print(avg_error)

# visualize
fig, axes = plt.subplots(3, 3, figsize=(14, 12))
axes = axes.ravel()

for k, idx in enumerate([0, 5, 10, 15, 20, 25, 30, 35, 40]):
    ax = axes[k]
    ax.plot(wave_lens, df_smoothed_train.loc[k], label="smoothed")
    ax.plot(wave_left, f_left_pre[k], label="predicted")
    ax.legend()
    ax.set_title('Example {}'.format(idx))

plt.show()


