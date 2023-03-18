# comp30027_assignment_1

Task 2: Qs 3, 5, 6

 TODO List:
1. implement preprocess(train_file_name, test_file_name) -> train_df, test_df
1. implement train(train_df) -> model (argmax over list of scoring functions... the whole bayes thing)
    - compute (log) priors
    - compute (log) likelihood functions
    - bundle them all into a scoring function for each class
1. implement predict(model, test_df (should the labels be separated?)) -> data frame predictions
1. implement evaluate(test_df) -> performance metrics: accuracy, precision, recall
    - make the confusion matrix
    - ??
