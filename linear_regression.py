#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import warnings
import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt
from data_one import train_df, test_df

# get_ipython().run_line_magic('matplotlib', 'inline')


warnings.filterwarnings(action="ignore", category=FutureWarning)


def mean_squared_error(x, y):
    y_p = np.asarray(y).reshape(-1)
    return np.mean((x - y_p) ** 2)


# In[28]:


train_df.head()


# In[29]:


test_df.head()


# In[30]:


train_df.shape


# In[31]:


test_df.shape


# In[32]:


Data = []


# ## Frequency k increment = 4

# # Linear Regression
#
# 1. Consider a simplified fitting problem in the frequency domain where we are looking to find the best fit
# of data with a set of periodic (trigonometric) basis functions of the form 1, x, sin(x), cos(x), sin(k ∗
# x), cos(k ∗ x), sin(2 ∗ k ∗ x), cos(2 ∗ k ∗ x), ..., where k is effectively the frequency increment. The
# resulting function for a given ”frequency increment”, k, and ”function depth”, d, and parameter vector Θ
# is then:
#
# $$y = Θ_{0} ∗ 1 + Θ_{1} ∗ x + \sum_{i=1}^{d}(Θ_{2*i} ∗ sin(i ∗ k ∗ x) + Θ_{2∗i+1} ∗ cos(i ∗ k ∗ x))$$
#
#
# For example, if k = 1 and d = 2, your basis (feature) functions are 1, x, sin(x), cos(x), sin(2x), cos(2x),
# and we are looking for the best matching parameters Θ for the function $$Θ_{0} + Θ_{1} ∗ x + Θ_{2} ∗ sin(x) +
# Θ_{3} ∗ cos(x) + Θ_{4}∗ sin(2x) + Θ_{5} ∗ cos(2x)$$.
#
# This means that this problem can be solved using linear
# regression as the function is linear in terms of the parameters Θ.
# You obtain your value for the ”frequency increment” k and thus your basis functions as part of the data
# generation process described above.
#
# a). Implement a linear regression learner to solve this best fit problem for 1 dimensional data. Make
# sure your implementation can handle fits for different ”function depths” (at least to ”depth” 6).
#
# b). Apply your regression learner to the data set that was generated for Question 1b) and plot the
# resulting function for ”function depth” 0, 1, 2, 3, 4, 5, and 6. Plot the resulting function together
# with the data points (using your favorite plotting program, e.g. Matlab, Octave, ...)
#
# c) Evaluate your regression functions by computing the error on the test data points that were generated
# for Question 1c). Compare the error results and try to determine for what polynomials overfitting
# might be a problem. Which order polynomial would you consider the best prediction function and
# why.
#
#
# d) Repeat the experiment and evaluation of part b) and c) using only the first 20 elements of the training
# data set part b) and the Test set of part c). What differences do you see and why might they occur ?
#
#

# In[35]:


# Defining the class
class LinearRegression:
    def __init__(self, learning_rate, iterations):

        self.learning_rate = learning_rate

        self.iterations = iterations

    # Function for model training

    def fit(self, X, Y):

        # no_of_training_examples, no_of_features

        self.m, self.n = X.shape

        # weight initialization
        # print(self.m,self.n)

        self.W = np.zeros(self.n)

        self.b = 0

        self.X = X

        self.Y = Y

        # gradient descent learning

        for i in range(self.iterations):

            self.update_weights()

        return self

    # Helper function to update weights in gradient descent

    def update_weights(self):

        Y_pred = np.asarray(self.predict(self.X)).reshape(-1)

        dW = -(2 * (self.X.T).dot(self.Y - Y_pred)) / self.m

        db = -2 * np.sum(self.Y - Y_pred) / self.m
        # update weights

        self.W = self.W - self.learning_rate * dW

        self.b = self.b - self.learning_rate * db

        return self

    # Hypothetical function  h( x )

    def predict(self, X):

        return X.dot(self.W.T) + self.b


def logreg(X, d, k=4):
    """Return y of x based on theta vector, frequency increment k, and function depth d."""
    data = []

    for i, x in enumerate(X):
        data.append([1, x])
        for p in range(1, d + 1):
            data[i].append(np.sin(i * k * x))
            data[i].append(np.cos(i * k * x))
    return np.matrix(data)


def r2_score(x, y):
    """Return R^2 where x and y are array-like."""
    y_p = np.asarray(y).reshape(-1)
    slope, intercept, r_value, p_value, std_err = sp.stats.linregress(x, y_p)
    return r_value ** 2


d0 = logreg(train_df.X, 0)
d1 = logreg(train_df.X, 1)
d2 = logreg(train_df.X, 2)
d3 = logreg(train_df.X, 3)
d4 = logreg(train_df.X, 4)
d5 = logreg(train_df.X, 5)

d0m = LinearRegression(0.0001, 1000)
d0m.fit(d0, train_df.y)

d1m = LinearRegression(0.0001, 1000)
d1m.fit(d1, train_df.y)

d2m = LinearRegression(0.0001, 1000)
d2m.fit(d2, train_df.y)

d3m = LinearRegression(0.0001, 1000)
d3m.fit(d3, train_df.y)

d4m = LinearRegression(0.0001, 1000)
d4m.fit(d4, train_df.y)

d5m = LinearRegression(0.0001, 1000)
d5m.fit(d5, train_df.y)


# print(d0m.coef_,d1m.coef_,d2m.coef_,d3m.coef_,d4m.coef_,d5m.coef_,sep="\n\n")


err0 = mean_squared_error(test_df.y, d0m.predict(logreg(test_df.X, 0)))
err1 = mean_squared_error(test_df.y, d1m.predict(logreg(test_df.X, 1)))
err2 = mean_squared_error(test_df.y, d2m.predict(logreg(test_df.X, 2)))
err3 = mean_squared_error(test_df.y, d3m.predict(logreg(test_df.X, 3)))
err4 = mean_squared_error(test_df.y, d4m.predict(logreg(test_df.X, 4)))
err5 = mean_squared_error(test_df.y, d5m.predict(logreg(test_df.X, 5)))


acc0 = r2_score(test_df.y, d0m.predict(logreg(test_df.X, 0)))
acc1 = r2_score(test_df.y, d1m.predict(logreg(test_df.X, 1)))
acc2 = r2_score(test_df.y, d2m.predict(logreg(test_df.X, 2)))
acc3 = r2_score(test_df.y, d3m.predict(logreg(test_df.X, 3)))
acc4 = r2_score(test_df.y, d4m.predict(logreg(test_df.X, 4)))
acc5 = r2_score(test_df.y, d5m.predict(logreg(test_df.X, 5)))

accs = [acc0, acc1, acc2, acc3, acc4, acc5]
errs = [err0, err1, err2, err3, err4, err5]
Data.append({"mse": errs, "r2_score": accs})

# print(err0,err1,err2,err3,err4,err5,sep='\n\n')

# print(f"{acc0*100}%",f"{acc1*100}%",f"{acc2*100}%",f"{acc3*100}%",f"{acc4*100}%",f"{acc5*100}%",sep='\n\n')


models = [d0m, d1m, d2m, d3m, d4m, d5m]
fig, ((ax0, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(2, 3, figsize=(15, 8))
axes = [ax0, ax1, ax2, ax3, ax4, ax5]

xsort = test_df.X.copy()
xsort.sort_values(axis=0, inplace=True)
fig.suptitle("Question 1(a): Linear Regression using all elements(tau=0.1).")
for i, mod in enumerate(models):
    axes[i].scatter(train_df.X, train_df.y)
    axes[i].set_title(f"Function depth={i}")
    axes[i].plot(
        xsort, mod.predict(logreg(test_df.X, i))[
            test_df.X.argsort(0)], color="g"
    )
plt.savefig("1a.png")


# In[ ]:


# In[36]:


# Repeat evaluation using first 20 elements of trainset
d0a = logreg(train_df.X.iloc[0:20], 0)
d1a = logreg(train_df.X.iloc[0:20], 1)
d2a = logreg(train_df.X.iloc[0:20], 2)
d3a = logreg(train_df.X.iloc[0:20], 3)
d4a = logreg(train_df.X.iloc[0:20], 4)
d5a = logreg(train_df.X.iloc[0:20], 5)

d0am = LinearRegression(0.0001, 1000)
d0am.fit(d0a, train_df.iloc[0:20].y)

d1am = LinearRegression(0.0001, 1000)
d1am.fit(d1a, train_df.iloc[0:20].y)

d2am = LinearRegression(0.0001, 1000)
d2am.fit(d2a, train_df.iloc[0:20].y)

d3am = LinearRegression(0.0001, 1000)
d3am.fit(d3a, train_df.iloc[0:20].y)

d4am = LinearRegression(0.0001, 1000)
d4am.fit(d4a, train_df.iloc[0:20].y)

d5am = LinearRegression(0.0001, 1000)
d5am.fit(d5a, train_df.iloc[0:20].y)

err0a = mean_squared_error(test_df.y, d0am.predict(logreg(test_df.X, 0)))
err1a = mean_squared_error(test_df.y, d1am.predict(logreg(test_df.X, 1)))
err2a = mean_squared_error(test_df.y, d2am.predict(logreg(test_df.X, 2)))
err3a = mean_squared_error(test_df.y, d3am.predict(logreg(test_df.X, 3)))
err4a = mean_squared_error(test_df.y, d4am.predict(logreg(test_df.X, 4)))
err5a = mean_squared_error(test_df.y, d5am.predict(logreg(test_df.X, 5)))


acc0a = r2_score(test_df.y, d0am.predict(logreg(test_df.X, 0)))
acc1a = r2_score(test_df.y, d1am.predict(logreg(test_df.X, 1)))
acc2a = r2_score(test_df.y, d2am.predict(logreg(test_df.X, 2)))
acc3a = r2_score(test_df.y, d3am.predict(logreg(test_df.X, 3)))
acc4a = r2_score(test_df.y, d4am.predict(logreg(test_df.X, 4)))
acc5a = r2_score(test_df.y, d5am.predict(logreg(test_df.X, 5)))

accs = [acc0a, acc1a, acc2a, acc3a, acc4a, acc5a]
erras = [err0a, err1a, err2a, err3a, err4a, err5a]

Data.append({"mse": erras, "r2_score": accs})
print(err0a, err1a, err2a, err3a, err4a, err5a, sep="\n\n")
print("\n")

print(
    f"{acc0a*100}%",
    f"{acc1a*100}%",
    f"{acc2a*100}%",
    f"{acc3a*100}%",
    f"{acc4a*100}%",
    f"{acc5a*100}%",
    sep="\n\n",
)

models = [d0am, d1am, d2am, d3am, d4am, d5am]
fig, ((ax0, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(2, 3, figsize=(15, 8))
axes = [ax0, ax1, ax2, ax3, ax4, ax5]
xsort = test_df.X.copy()
xsort.sort_values(axis=0, inplace=True)
fig.suptitle(
    "Question 1(b): Linear Regression using First 20 elements(tau=0.1).")
for i, mod in enumerate(models):
    axes[i].scatter(train_df.X, train_df.y)
    axes[i].set_title(f"Function depth={i}")
    axes[i].plot(
        xsort, mod.predict(logreg(test_df.X, i))[
            test_df.X.argsort(0)], color="r"
    )
plt.savefig("1b.png")


# In[37]:


[Data[i].update({"name": f"part_{i}"}) for i in range(len(Data))]
Data

with open("q1.json", "w") as wf:
    json.dump(Data, wf)


# In[38]:


d1f = pd.DataFrame.from_dict(Data)
df = pd.DataFrame()
df["mse_all"] = d1f.mse[0]
df["mse_20"] = d1f.mse[1]
df["r2_all"] = d1f.r2_score[0]
df["r2_20"] = d1f.r2_score[1]
df


# In[ ]:


# In[39]:
