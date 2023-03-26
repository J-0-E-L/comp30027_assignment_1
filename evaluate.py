import pandas as pd
import itertools

"""
    The first argument should be a series which contains the true labels of a test set.
    The second argument should be a series which contains predicted labels of the same test set."""
def evaluate(true_labels, predicted_labels, positive_label, negative_label):
    labels = [positive_label, negative_label]
    
    # Count the number of occurrences for each combination of true and predicted labels
    combination_counts = pd.concat([true_labels, predicted_labels], axis="columns", ignore_index=True).value_counts()
    
    # Set the count of any combination that wasn't seen to 0
    index = pd.MultiIndex.from_tuples(itertools.product(labels, labels), names=["true_labels", "predicted_labels"])
    confusion_matrix = combination_counts.reindex(index=index, fill_value=0)
    
    evaluation_metrics = dict()
    
    # At this point there is no ambiguity regarding the meaning of these variables... give them aliases
    m, positive, negative = confusion_matrix, positive_label, negative_label
    print(confusion_matrix)
    evaluation_metrics["accuracy"] = (m[positive, positive] + m[negative, negative]) / m.sum()
    evaluation_metrics["precision"] = m[positive, positive] / (m[positive, positive] + m[negative, positive])
    evaluation_metrics["recall"] = m[positive, positive] / (m[positive, positive] + m[positive, negative])
    evaluation_metrics["f_1"] = 2 * evaluation_metrics["precision"] * evaluation_metrics["recall"] / (evaluation_metrics["precision"] + evaluation_metrics["recall"])
    
    # TODO: more than 2 classes:
    #   - total accuracy
    #   - macro/micro/weighted averaging:
    #        - precision
    #        - recall
    #        - f1 (beta? -> what does it even do?)
    #   - per class:
    #        - precision
    #        - recall
    #        - f1 (beta? -> what does it even do?)    
    
    return evaluation_metrics