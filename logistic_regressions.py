import pandas as pd
import numpy as np
import scipy as sp
import warnings
from data import (
    heights,
    weights,
    age,
    gender,
    samples,
    k_values,
    program_height,
    program_age,
    program_gender,
    program_weight,
    df,
)
import matplotlib.pyplot as plt
import math


def mean_squared_error(x, y):
    y_p = np.asarray(y).reshape(-1)
    return np.mean((x - y_p) ** 2)


def r2_score(x, y):
    """Return R^2 where x and y are array-like."""
    y_p = np.asarray(y).reshape(-1)
    slope, intercept, r_value, p_value, std_err = sp.stats.linregress(x, y_p)
    return r_value ** 2


# # Logistic Regression
# 3. Consider again the problem from Questions 2 and 3 in the first assignment where we want to predict the gender of a person from a set of input parameters, namely height, weight, and age. Assume the same datasets you generated for the first assignment.
#
# a) Implement logistic regression to classify this data (use the individual data elements, i.e. height,
# weight, and age, as features). Your implementation should take different data sets as input for
# learning.
#
#
# b) Plot the resulting function together with the data points (using your favorite plotting program, e.g.
# Matlab, Octave, ...). To generate the function for locally weighted linear regression for this data you
# need to compute its value for finely spaced test data points, x, between −3 and 3.
#
#
# c) Evaluate the performance of your logistic regression classifier in the same way as for Homework
# 1 using leve-one-out validation and compare the results with the ones for KNN and Naı̈ve Bayes
# (either from your first assignment or, if you did not implement these, using an existing implemen-
# tation). Discuss what differences exist and why one method might outperform the others for this
# problem.
#
#
# d) Repeat the evaluation and comparison from part c) with the age feature removed. Again, discuss
# what differences exist and why one method might outperform the others in this case.
#
#

# In[18]:


# In[46]:


df.shape
df.head()


# In[53]:


class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.verbose = verbose
        self.fit_intercept = fit_intercept

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        # weights initialization
        self.theta = np.zeros(X.shape[1])

        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient

            if self.verbose == True and i % 10000 == 0:
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                print(f"loss: {self.__loss(h, y)} \t")

    def predict_proba(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        return self.__sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold=0.5):
        return self.predict_proba(X) >= threshold


# In[60]:


# define cross-validation method to use

lr = LogisticRegression(lr=0.1, num_iter=300000)


def lencode(x):
    if x == "M":
        return 1
    else:
        return 2


X = df[["height", "weight"]]
df["gender_code"] = df["gender"].apply(lencode)
y = df["gender_code"]
# df.info()
df.head()
df.shape


# In[61]:


X_train, X_test, y_train, y_test = X.iloc[0:
                                          80], X.iloc[80:], y.iloc[0:80], y.iloc[80:]
warnings.filterwarnings(action="ignore", category=UserWarning)
warnings.filterwarnings(action="ignore", category=Warning)
lr.fit(X_train, y_train)


# In[51]:


def calculate_cartesian_distance_withOutage(sample, df_dataset):
    inputs = df_dataset.drop(["age", "gender"], axis=1).values
    diffs = sample - inputs
    sum_pow = np.sum(np.power(diffs, 2), axis=1)

    return np.power(sum_pow, 0.5)


def calculate_cartesian_distance_withage(sample, df_dataset):
    inputs = df_dataset.drop(["gender"], axis=1).values
    diffs = sample - inputs
    sum_pow = np.sum(np.power(diffs, 2), axis=1)

    return np.power(sum_pow, 0.5)


def gender_prediction(k, sorted_labels):
    # the method predicts the gender with the help of k-nearest nighbour and sorted_labels
    k_nearest_neighbors = sorted_labels[:k]
    men_occurencies = np.count_nonzero(k_nearest_neighbors == "M")
    women_occurencies = np.count_nonzero(k_nearest_neighbors == "W")

    return "M" if men_occurencies > women_occurencies else "W"


def kNN_classifier(sample, k, df_dataset, drop_age):
    if drop_age:
        cart_distance = calculate_cartesian_distance_withOutage(
            sample, df_dataset)
    else:
        cart_distance = calculate_cartesian_distance_withage(
            sample, df_dataset)

    labels = df_dataset["gender"].values

    # get the cartesian distance from each data point
    # cart_distance = cartesian_distance(sample, inputs)

    # create a 2D array with the 1st column being the above distances and the second corresponding label
    labeled_cart = np.vstack((cart_distance, labels))

    # sort in an ascending manner the above 2D array based on the distances
    sorted_cart = labeled_cart.T[labeled_cart.T[:, 0].argsort()]
    sorted_labels = sorted_cart.T[1]

    return gender_prediction(k, sorted_labels)


def samplesClassification(samples, training_dataset):
    for sample in samples:
        print("For samples :{} - the predictions are ".format(sample))
        # selecting the neiighbors for each classification for each samples.
        for k in k_values:
            # print("\tK:{}".format(k))
            prediction_1 = kNN_classifier(
                sample, k, training_dataset, drop_age=False)
            print(
                "\tFor k:{} number of neighbors prediction is {} ".format(
                    k, prediction_1
                )
            )
            # prediction_2 = kNN_classifier(sample[:2], k, df_dataset,
            #                                   drop_age=True)  # assumption: gender is is the 3rd element of the sample
            # print("\tPrediction is {} for k:{} number of neighbors without using age feature".format(prediction_2, k))
            print()
    print()


df_dataset = pd.DataFrame(
    {"heights": heights, "weights": weights, "age": age, "gender": gender}
)

program_dataset = pd.DataFrame(
    {
        "heights": program_height,
        "weights": program_weight,
        "age": program_age,
        "gender": program_gender,
    }
)

samplesClassification(samples, df_dataset)

for k in k_values:
    valid_predictions_all_features, valid_predictions_exclude_age = 0, 0

    # test with leave-1-out training method
    for index, test_sample in program_dataset.iterrows():
        sample = test_sample.values[:3]  # leave the target out
        target = test_sample.values[3]
        prediction = kNN_classifier(
            sample, k, program_dataset.drop(index), drop_age=False
        )
        valid_predictions_all_features += 1 if target == prediction else 0

        prediction = kNN_classifier(
            sample[:2], k, program_dataset.drop(index), drop_age=True
        )
        valid_predictions_exclude_age += 1 if target == prediction else 0
    print("KNN Performance using k:{}".format(k))
    print(
        "{}/{} correct predictions using all features".format(
            valid_predictions_all_features, program_dataset.shape[0]
        )
    )
    print(
        "{}/{} correct predictions excluding age".format(
            valid_predictions_exclude_age, program_dataset.shape[0]
        )
    )
    print()


# In[62]:


l = np.linspace(-3, 3, 300)
# l
correct = 0
hy = lr.predict(X_test)
for v1, v2 in zip(hy, y_test):
    if v1 == v2:
        correct += 1
print(f"Without age feature: {correct} out of {len(y_test)}")


# In[64]:


# With Age
X = df[["height", "weight", "age"]]
X_train, X_test, y_train, y_test = X.iloc[0:
                                          80], X.iloc[80:], y.iloc[0:80], y.iloc[80:]
lr1 = LogisticRegression(lr=0.1, num_iter=300000)
lr1.fit(X_train, y_train)


# In[65]:


correct = 0
hy = lr1.predict(X_test)
for v1, v2 in zip(hy, y_test):
    if v1 == v2:
        correct += 1
print(f"Using all features: {correct} out of {len(y_test)}")


# In[70]:


def accuracy_score(y, y_pred):
    c = 0
    for v1, v2 in zip(y, y_pred):
        if v1 == v2:
            c += 1
    return c / len(y)


# In[ ]:


# In[67]:


def calculate_gaussian_probability(sample, mu, sigma):
    return (
        1 / (math.sqrt(sigma ** math.pi)) *
        np.exp(-sigma * np.power((sample - mu), 2))
    )


def pdf_calculate(sample, feature, df_dataset):
    """
    Calculates the Probability Density Function (PDF) of 2 classes; 'M' and 'W' here.
    :param feature: feature to calulate PDF for
    :return: probability for each class
    """
    p_feature_men_mean = np.mean(
        df_dataset.loc[df_dataset["gender"] == "M"][feature].values
    )
    p_feature_men_std = np.std(
        df_dataset.loc[df_dataset["gender"] == "M"][feature].values
    )
    pdf_feature_men = calculate_gaussian_probability(
        sample, p_feature_men_mean, p_feature_men_std
    )

    p_feature_women_mean = np.mean(
        df_dataset.loc[df_dataset["gender"] == "W"][feature].values
    )
    p_feature_women_std = np.std(
        df_dataset.loc[df_dataset["gender"] == "W"][feature].values
    )
    pdf_feature_women = calculate_gaussian_probability(
        sample, p_feature_women_mean, p_feature_women_std
    )

    return pdf_feature_men, pdf_feature_women


def gaussian_naive_bayes_classification(sample, df_dataset, drop_age):
    """
    Naive Assumption -> every feature is independent from each other
    Thus, P(height, weight, age | class_i) = P(height| class_i)*P(weight| class_i)*P(age| class_i)
    Two classes: "M" & "W"
    :param sample:
    :param df_dataset:
    :return: predicted class
    """
    # Calculate PDFs for each feature
    pdf_height_men, pdf_height_women = pdf_calculate(
        sample[0], "heights", df_dataset)
    pdf_weight_men, pdf_weight_women = pdf_calculate(
        sample[1], "weights", df_dataset)

    # calculate prior probabilities of the classes
    num_of_men, num_of_women = np.count_nonzero(
        np.asarray(gender) == "M"
    ), np.count_nonzero(np.asarray(gender) == "W")
    total_num_of_classes = num_of_women + num_of_men
    prior_men, prior_women = (
        num_of_men / total_num_of_classes,
        num_of_women / total_num_of_classes,
    )

    if drop_age:
        # P(Class|Data) = P(Data|Class) * P(Class)
        # "Naive" -> P(feature_1, feature_2| class) = P(feature_1|Class)*P(feature_2|Class)
        p_man = pdf_height_men * pdf_weight_men * prior_men
        p_woman = pdf_height_women * pdf_weight_women * prior_women
    else:
        pdf_age_men, pdf_age_women = pdf_calculate(
            sample[2], "age", df_dataset)

        # P(Class|Data) = P(Data|Class) * P(Class)
        # "Naive" -> P(feature_1, feature_2, feature_3| class) = P(feature_1|Class)*P(feature_2|Class)*P(feature_3|Class)
        p_man = pdf_height_men * pdf_weight_men * pdf_age_men * prior_men
        p_woman = pdf_height_women * pdf_weight_women * pdf_age_women * prior_women

    return "M" if p_man > p_woman else "W"


if __name__ == "__main__":

    df_dataset = pd.DataFrame(
        {"heights": heights, "weights": weights, "age": age, "gender": gender}
    )

    for sample in samples:
        print("sample:{}".format(sample))
        prediction_1 = gaussian_naive_bayes_classification(
            sample, df_dataset, drop_age=False
        )
        print("\tPrediction is {}".format(prediction_1))
        # prediction_2 = KNN_classification(sample[:2], k, df_dataset,
        #                                   drop_age=True)  # assumption: gender is is the 3rd element of the sample
        # print("\tPrediction is {} for k:{} number of neighbors without using age feature".format(prediction_2, k))
        print()

    valid_predictions_all_features, valid_predictions_exclude_age = 0, 0

    # test with leave-1-out training method
    for index, test_sample in df_dataset.iterrows():
        sample = test_sample.values[:3]  # leave the target out
        target = test_sample.values[3]
        prediction = gaussian_naive_bayes_classification(
            sample, df_dataset.drop(index), drop_age=False
        )
        valid_predictions_all_features += 1 if target == prediction else 0
        # print("Prediction:{} - Target: {} for k: {} number of neighbors".format(prediction_1, target, k))

        prediction = gaussian_naive_bayes_classification(
            sample[:2], df_dataset.drop(index), drop_age=True
        )  # assumption: gender is is the 3rd element of the sample
        valid_predictions_exclude_age += 1 if target == prediction else 0
        # print("Prediction: {} - Target: {} for k:{} number of neighbors without using age feature".format(prediction_2, target, k))

        # prediction = KNN_classification(sample[:2], k, df_dataset.drop(index),
        #                                   drop_age=True)  # assumption: gender is is the 3rd element of the sample
        # valid_predictions_all_features += 1 if target == prediction else 0
    print("Gaussian Naive Performance")
    print(
        "{}/{} correct predictions using all features".format(
            valid_predictions_all_features, df_dataset.shape[0]
        )
    )
    print(
        "{}/{} correct predictions excluding age".format(
            valid_predictions_exclude_age, df_dataset.shape[0]
        )
    )
    print()


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[73]:


# plt.scatter(X['height'],y)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 8))
cols = ["height", "weight"]  # ,"age"]
axs = [ax1, ax2]  # ,ax3]
xx = np.linspace(3, 9, 100)
yy = np.linspace(1, 5, 100).T
zz = np.linspace(1, 5, 100).T
xx, yy = np.meshgrid(xx, yy)
Xfull = np.c_[xx.ravel(), yy.ravel()]


classifiers = {"LogisticRegression All features": lr1}
n_classifiers = 3
for index, (name, classifier) in enumerate(classifiers.items()):
    y_pred = classifier.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print("Accuracy for %s: %0.1f%% " % (name, accuracy * 100))


# ax = plt.axes([0.15, 0.04, 0.7, 0.05])
# plt.title("Probability")
# plt.colorbar(imshow_handle, cax=ax, orientation="horizontal")

# plt.show()
for i in range(2):
    axs[i].set_title(f"Gender against {cols[i]} 1=M, 2=W")
    axs[i].scatter(X[cols[i]], y)
    axs[i].set_xlabel(f"{cols[i]}")
    axs[i].set_ylabel("Gender")

# plt.savefig(".png")


# In[74]:


82 / 120


# In[75]:


29 / 40


# In[76]:


11 / 14


# In[ ]:
