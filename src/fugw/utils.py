from sklearn.base import BaseEstimator, TransformerMixin


class BaseModel(BaseEstimator, TransformerMixin):
    def fit(self, source_data, target_data):
        return None

    def predict(self, source_map):
        return None
