# Naive Bayes Model Application

# Predicts class for a test instance
def predict_instance(instance, priors, likelihood_pdf, likelihood_pdf_parameters):    
    labels = likelihood_pdf_parameters[likelihood_pdf_parameters.keys()[0]].keys()
    best, best_score = [], 0
    for label in labels:
        # Calculate the best score for the current class label
        score = 0
        for feature in instance[instance.notnull()].index:
            score += likelihood_pdf(instance[feature], likelihood_pdf, likelihood_pdf_parameters)
        # Keep track of the best class labels
        if score > best_score:
            best = [label]
            best_score = score
        elif score == best_score:
            best.append(label)
    
# Predicts classes for each instance in the test dataset
def predict(test_df, priors, likelihood_pdf, likelihood_pdf_parameters):    
    return test_df.apply(lambda instance: predict_instance(instance, priors, likelihood_pdf, likelihood_pdf_parameters))