""" Classifies each instance using the given model.
    The first argument should be the model (a function: Series -> label).
    The second argument should be a DataFrame, where each row is an instance, with
    column names that are consistent with what the model is expecting)."""
def predict(model, instances):
    return instances.apply(model, axis="columns")