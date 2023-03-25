import numpy as np
import pandas as pd
import math

""" Calculate the prior probability of all the class labels for the naive bayes
    model. The argument of the function is the train dataframe and the output
    is a dictionary containing the prior probabilities of each label """
def calculate_prior_prob(train_df : pd.DataFrame):
    prior_prob = {}
    label_instances = train_df["label"]
    labels = np.unique(label_instances)
    total_instances = label_instances.count()
    
    # iterate over all the labels and calculate the prior probabilities
    for label in labels:
        label_count = (label_instances == label).sum()
        prior_prob[label] = label_count / total_instances

    return prior_prob

""" Calculate the parameters of a Gaussian distribution for likelihood 
    probabilities of all the features in the dataset conditional on the class 
    label for the model. The argument of the function is the train dataframe 
    and the output is a 2D dictionary containing the means and 
    standard deviations
    """
def calculate_likelihood_prob(train_df : pd.DataFrame):
    gaussian_likelihood = {}
    features = train_df.columns[:-1]
    label_instances = train_df["label"]
    labels = np.unique(label_instances)

    # iterate over each feature and then each label
    for feature in features:
        gaussian_likelihood[feature] = {}
        feature_label_instances = train_df[[feature, "label"]]

        for label in labels:
            label_count = (label_instances == label).sum()

            # get all the feature values that have the label and compute the
            # mean and standard deviation
            values_in_label = feature_label_instances[(label_instances == label)]
            likelihood_mean = values_in_label[feature].sum() / label_count
            likelihood_sd = math.sqrt(values_in_label[feature].apply(lambda x: (x - likelihood_mean)**2).sum() / (label_count - 1))
            gaussian_likelihood[feature][label] = (likelihood_mean, likelihood_sd)

    return gaussian_likelihood
