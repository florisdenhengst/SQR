from pmlb import fetch_data
from pysr import PySRRegressor
import numpy as np
import random
import sklearn


N_ITERS = 900
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# get ne
dataset_name = "1199_BNG_echoMonths"
X, y = fetch_data(dataset_name, return_X_y=True)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=SEED)

binary_operators = ["+", "*", "/", "-"]
unary_operators = ["exp", "sin", "cos", "log", "square"]
complexity_of_operators = {
    "+": 1,
    "-": 1,
    "*": 1,
    "/": 2,
    "exp": 4,
    "sin": 3,
    "cos": 3,
    "log": 3,
    "square": 2,
}

target_quantile = 0.5
regressor = PySRRegressor(
        niterations=N_ITERS,  # improve for better results50
        binary_operators=binary_operators,
        unary_operators=unary_operators,
        complexity_of_operators=complexity_of_operators,
        elementwise_loss=f"QuantileLoss({target_quantile})",
)
regressor.fit(X_train, y_train)
print(regressor)