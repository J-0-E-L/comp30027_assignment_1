import pandas as pd
import itertools

"""
    The first argument should be a set which contains all possible class labels.
    The second argument should be a series which contains the true labels of a test set.
    The third argument should be a series which contains predicted labels of the same test set."""
def evaluate(true_labels, predicted_labels, labels):
    
    # Count the number of occurrences for each combination of true and predicted labels
    combination_counts = pd.concat([true_labels, predicted_labels], axis="columns", ignore_index=True).value_counts()
    # print(combination_counts)
    
    # Set the count of any combination that wasn't seen to 0
    index = pd.MultiIndex.from_tuples(itertools.product(labels, labels), names=["true_labels", "predicted_labels"])
    confusion_matrix = combination_counts.reindex(index=index, fill_value=0)
        
    # print(confusion_matrix)
    
    # TODO: choose metrics and implement them... special case for when there are 2 labels?
