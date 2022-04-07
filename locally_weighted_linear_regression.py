import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.stats
import json
from data_one import train_df, test_df


def mean_squared_error(x, y):
    y_p = np.asarray(y).reshape(-1)
    return np.mean((x - y_p) ** 2)


def r2_score(x, y):
    """Return R^2 where x and y are array-like."""
    y_p = np.asarray(y).reshape(-1)
    slope, intercept, r_value, p_value, std_err = sp.stats.linregress(x, y_p)
    return r_value ** 2


Data = []
# # Locally Weighted Linear Regression
#
# 2. Another way to address nonlinear functions with a lower likelihood of overfitting is the use of locally weighted linear regression where the neighborhood function addresses non-linearity and the feature vector stays simple. In this case we assume that we will use only the raw feature, x, as well as the bias (i.e.
# a constant feature 1).
# Thus the locally applied regression function is $$y = Θ_{0} + Θ_{1} ∗ x.$$
#
# As discussed in class, locally weighted linear regression solves a linear regression problem for each query
# point, deriving a local approximation for the shape of the function at that point (as well as for its value).
# To achieve this, it uses a modified error function that applies a weight to each data point’s error that is
# related to its distance from the query point. Here we will assume that the weight function for the ith data
# point and query point x is: $$w^{i} (x) = e^{-\frac{(x^{i}-x)^{2}}{(2γ^{2})}}$$
#
# where γ is a measure of the ”locality” of the weight function, indicating how fast the influence of a data
# point changes with its distance from the query point.
# Your value for γ is provided during data generation.
# a) Implement a locally weighted linear regression learner to solve the best fit problem for 1 dimen-
# sional data.
#
# b) Apply your locally weighted linear regression learner to the data set that was generated for Ques-
# tion 1b) and plot the resulting function together with the data points (using your favorite plotting
# program, e.g. Matlab, Octave, ...)
#
# c) Evaluate the locally weighted linear regression on the Test data from Question 1 c). How does the
# performance compare to the one for the results from Question 1 c) ?
#
# d) Repeat the experiment and evaluation of part b) and c) using only the first 20 elements of the training
# data set. How does the performance compare to the one for the results from Question 1 d) ? Why
# might this be the case ?
#
# e) Given the results form parts c) and d), do you believe the data set you used was actually derived
# from a function that is consistent with the function format in Question 1 ? Justify your answer.
#

# ## Question 2 weight scaling factor γ : 0.1

# In[40]:


def kernel(pt, X, tau):
    m, n = np.shape(X)
    weights = np.mat(np.eye((m)))
    for j in range(m):
        diff = pt - X[j]
        weights[j, j] = np.exp(diff * diff.T / (-2 * (tau ** 2)))
    return weights


def lweight(pt, X, Y, tau):
    wt = kernel(pt, X, tau)
    w_x = (X.T * (wt * X)).I * (X.T * wt * Y.T)
    return w_x


def lweightedreg(X, Y, tau=0.1):
    m, n = np.shape(X)
    X = np.mat(X)
    Y = np.mat(Y)
    y_pred = np.zeros(m)
    for i in range(m):
        y_pred[i] = X[i] * lweight(X[i], X, Y, tau)

    return y_pred


def logregW(X, k=4):
    """Return y of x based."""
    data = []

    for i, x in enumerate(X):
        data.append([x])
    return np.asarray(data)


d0aw = logregW(train_df.X)
d1aw = logregW(train_df.X)
d2aw = logregW(train_df.X)
d3aw = logregW(train_df.X)
d4aw = logregW(train_df.X)
d5aw = logregW(train_df.X)

d0amw = lweightedreg(d0aw, train_df.y)
# d0amw.fit(d0aw,train_df.y)

d1amw = lweightedreg(d1aw, train_df.y)
# d1amw.fit(d1aw,train_df.y)

d2amw = lweightedreg(d2aw, train_df.y)
# d2amw.fit(d2aw,train_df.y)

d3amw = lweightedreg(d3aw, train_df.y)
# d3amw.fit(d3aw,train_df.y)

d4amw = lweightedreg(d4aw, train_df.y)
# d4amw.fit(d4aw,train_df.y)

d5amw = lweightedreg(d5aw, train_df.y)
# d5amw.fit(d5aw,train_df.y)
# print(d0amw)

err0aw = mean_squared_error(train_df.y, d0amw)
err1aw = mean_squared_error(train_df.y, d1amw)
err2aw = mean_squared_error(train_df.y, d2amw)
err3aw = mean_squared_error(train_df.y, d3amw)
err4aw = mean_squared_error(train_df.y, d4amw)
err5aw = mean_squared_error(train_df.y, d5amw)


acc0aw = r2_score(train_df.y, d0amw)
acc1aw = r2_score(train_df.y, d1amw)
acc2aw = r2_score(train_df.y, d2amw)
acc3aw = r2_score(train_df.y, d3amw)
acc4aw = r2_score(train_df.y, d4amw)
acc5aw = r2_score(train_df.y, d5amw)

accsw = [acc0aw, acc1aw, acc2aw, acc3aw, acc4aw, acc5aw]
errsw = [err0aw, err1aw, err2aw, acc3aw, err4aw, err5aw]

print(err0aw, sep="\n\n")
print("\n")

print(f"{acc0aw*100}%", sep="\n\n")
Data.append({"mse": errsw[0], "r2_score": accsw[0]})
models = [d0amw]
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
axes = [ax]
xsort = train_df.X.copy()
xsort.sort_values(axis=0, inplace=True)
for i, mod in enumerate(models):
    axes[i].scatter(train_df.X, train_df.y)
    axes[i].set_title(
        f"Question 2a: Locally Weighted Linear Regression using all elements(tau=0.1)"
    )
    axes[i].plot(xsort, mod[train_df.X.argsort(0)], color="y")
plt.savefig("2a.png")


# In[41]:


d0aw = logregW(train_df.X.iloc[0:20])
d1aw = logregW(train_df.X.iloc[0:20])
d2aw = logregW(train_df.X.iloc[0:20])
d3aw = logregW(train_df.X.iloc[0:20])
d4aw = logregW(train_df.X.iloc[0:20])
d5aw = logregW(train_df.X.iloc[0:20])

d0amw = lweightedreg(d0aw, train_df.y.iloc[0:20])
# d0amw.fit(d0aw,train_df.y)

d1amw = lweightedreg(d1aw, train_df.y.iloc[0:20])
# d1amw.fit(d1aw,train_df.y)

d2amw = lweightedreg(d2aw, train_df.y.iloc[0:20])
# d2amw.fit(d2aw,train_df.y)

d3amw = lweightedreg(d3aw, train_df.y.iloc[0:20])
# d3amw.fit(d3aw,train_df.y)

d4amw = lweightedreg(d4aw, train_df.y.iloc[0:20])
# d4amw.fit(d4aw,train_df.y)

d5amw = lweightedreg(d5aw, train_df.y.iloc[0:20])
# d5amw.fit(d5aw,train_df.y)
# print(d0amw)

err0aw = mean_squared_error(train_df.y.iloc[0:20], d0amw)
err1aw = mean_squared_error(train_df.y.iloc[0:20], d1amw)
err2aw = mean_squared_error(train_df.y.iloc[0:20], d2amw)
err3aw = mean_squared_error(train_df.y.iloc[0:20], d3amw)
err4aw = mean_squared_error(train_df.y.iloc[0:20], d4amw)
err5aw = mean_squared_error(train_df.y.iloc[0:20], d5amw)


acc0aw = r2_score(train_df.y.iloc[0:20], d0amw)
acc1aw = r2_score(train_df.y.iloc[0:20], d1amw)
acc2aw = r2_score(train_df.y.iloc[0:20], d2amw)
acc3aw = r2_score(train_df.y.iloc[0:20], d3amw)
acc4aw = r2_score(train_df.y.iloc[0:20], d4amw)
acc5aw = r2_score(train_df.y.iloc[0:20], d5amw)

accsaw = [acc0aw, acc1aw, acc2aw, acc3aw, acc4aw, acc5aw]
erraws = [err0aw, err1aw, err2aw, acc3aw, err4aw, err5aw]

print(err0aw, sep="\n\n")
print("\n")

print(f"{acc0aw*100}%", sep="\n\n")
Data.append({"mse": erraws[0], "r2_score": accsaw[0]})
models = [d0amw]
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
axes = [ax]
xsort = train_df.X.iloc[0:20].copy()
xsort.sort_values(axis=0, inplace=True)
for i, mod in enumerate(models):
    axes[i].scatter(train_df.X, train_df.y)
    axes[i].set_title(
        f"Question 2b:Locally Weighted Linear Regression: first 20 elements(tau=0.1)"
    )
    axes[i].plot(xsort, mod[train_df.X.iloc[0:20].argsort(0)], color="y")
plt.savefig("2b.png")


# In[42]:


[Data[i].update({"name": f"part_{i}"}) for i in range(len(Data))]
print(Data)

with open("q2.json", "w") as wf:
    json.dump(Data, wf)


# In[43]:


# In[44]:


d2f = pd.DataFrame.from_dict(Data)
d2f


# In[ ]:


# In[ ]:


# In[ ]:
