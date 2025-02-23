import joblib
from utils import get_device

DEVICE = get_device()


class RegressionModelLoader:
    def __init__(self):
        print("Regression Model init")
        self.model = self.load("./logistic_regression_model.pkl")

    def load(self, regression_model_file_name):
        return joblib.load(regression_model_file_name)


regression_model = RegressionModelLoader()
