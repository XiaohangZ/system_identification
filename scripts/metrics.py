from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def accuracy(predictions,labels):
    print("mean_absolute_error:", mean_absolute_error(predictions, labels))
    print("mean_squared_error:", mean_squared_error(predictions, labels))
    print("rmse:", sqrt(mean_squared_error(predictions, labels)))
    print("r2 score:", r2_score(predictions, labels))

