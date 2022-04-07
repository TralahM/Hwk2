import numpy as np
import pandas as pd

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
)


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


if __name__ == "__main__":

    # ColumnsToUse = ["Height","weights","age","gender"]
    # samples = pd.read_csv("TestingDataSet.csv", names = ColumnsToUse[:3], header=0 )
    # training_dataset = pd.read_csv("TrainingDataSet.csv", names = ColumnsToUse, header=0 )
    # df_dataset = pd.read_csv("Program_Data.csv", names = ColumnsToUse, header=0 )

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
