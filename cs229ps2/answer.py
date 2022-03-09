import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def add_intercept(X_):
    m, n = X_.shape
    X = np.zeros((m, n + 1))
    X[:, 0] = 1
    X[:, 1:] = X_
    return X


def load_data(filename):
    D = np.loadtxt(filename)
    Y = D[:, 0]
    X = D[:, 1:]
    return add_intercept(X), Y


def calc_grad(X, Y, theta):
    m, n = X.shape
    grad = np.zeros(theta.shape)

    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))
    grad = -(1. / m) * (X.T.dot(probs * Y))

    return grad


def logistic_regression(X, Y):
    m, n = X.shape
    theta = np.zeros(n)
    learning_rate = 10
    thetas = []
    errors = []

    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta - learning_rate * grad
        norm = np.linalg.norm(prev_theta - theta)
        errors.append(norm)
        if i % 10000 == 0:
            print('Finished {0} iterations; Diff theta: {1}; theta: {2}; Grad: {3}'.format(
                i, norm, theta, grad))
            thetas.append(theta)
            errors.append(norm)
        if norm < 1e-15:
            print('Converged in %d iterations' % i)
            break
        if i == 60000:
            break
    return thetas, errors


# data_A
df_a = pd.read_csv("data_a.txt", header=None, sep=' ', names=['label', 'x1', 'x2'])
print(df_a)

# ax = plt.axes()
# df_a.query('label == 1').plot.scatter(x='x1', y='x2', ax=ax, color='blue')
# df_a.query('label == -1').plot.scatter(x='x1', y='x2', ax=ax, color='red')
# plt.title("data_A")
# plt.show()


# data_b
df_b = pd.read_csv("data_b.txt", header=None, sep=' ', names=['label', 'x1', 'x2'])
print(df_b)

# plt.figure()
# ax = plt.axes()
# df_b.query('label == 1').plot.scatter(x='x1', y='x2', ax=ax, color='blue')
# df_b.query('label == -1').plot.scatter(x='x1', y='x2', ax=ax, color='red')
# plt.title("data_B")
# plt.show()

# A
Xa, Ya = load_data("data_a.txt")
print(Xa.shape, Ya.shape)
thetas, errors = logistic_regression(Xa, Ya)
print(len(thetas))

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes = axes.ravel()

for k, theta in enumerate(thetas[:3]):
    ax = axes[k]
    print(k, theta, errors[k])
    line_x = np.arange(0, 1, 0.1)
    line_y = -(theta[0] + theta[1]*line_x) / theta[2]
    df_a.query('label == 1').plot.scatter(x='x1', y='x2', ax=ax, color='blue')
    df_a.query('label == -1').plot.scatter(x='x1', y='x2', ax=ax, color='red')
    ax.plot(line_x, line_y)

# plt.show()

# B
# A
Xb, Yb = load_data("data_b.txt")
print(Xb.shape, Yb.shape)
thetas, errors = logistic_regression(Xb, Yb)
print(len(thetas))

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.ravel()

for k, theta in enumerate(thetas[:6]):
    ax = axes[k]
    print(k, theta, errors[k])
    line_x = np.arange(0, 1, 0.1)
    line_y = -(theta[0] + theta[1]*line_x) / theta[2]
    df_b.query('label == 1').plot.scatter(x='x1', y='x2', ax=ax, color='blue')
    df_b.query('label == -1').plot.scatter(x='x1', y='x2', ax=ax, color='red')
    ax.plot(line_x, line_y)

plt.show()
