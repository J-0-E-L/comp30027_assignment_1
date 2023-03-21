import numpy as np
import pandas as pd

""" Calculate the prior probability of all the class labels for the naive bayes
    model. The argument of the function is the train dataframe and the output
    is a dictionary containing the prior probabilities of each label """
def calculate_prior_prob(train_df):
    prior_prob = {}
    label_instances = train_df["label"]
    labels = np.unique(label_instances)
    total_instances = label_instances.count()
    
    # iterate over all the labels and calculate the prior probabilities
    for label in labels:
        label_count = (label_instances == label).count()
        prior_prob[label] = label_count / total_instances

    return prior_prob