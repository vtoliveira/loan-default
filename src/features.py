import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklego.preprocessing import ColumnCapper

class NewFeatureTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, quantile_range=None):
        super().__init__()
        if not quantile_range:
            self.cc = ColumnCapper()
        else:
            self.cc = ColumnCapper(quantile_range=quantile_range)
    
    def fit(self, X, y=None):
        self.cc.fit(X["annual_inc"].values.reshape(-1, 1))

        return self

    def transform(self, X):
        
        func_tot_loan_amnt = lambda df: df["installment"]*((1+(df["int_rate"]/100))**(df["term"] / 12))
        func_int_to_inc = lambda df: ((df["installment"]*12) / (df["annual_inc"]))*100
        func_tot_loan_to_inc = lambda df: (df["total_loan_amount"] / (df["annual_inc"]*(df["term"] / 12)))*100

        X = (
            X.assign(
                annual_inc=lambda df: self.cc.transform(df["annual_inc"].values.reshape(-1, 1)),
                total_loan_amount=func_tot_loan_amnt,
                installment_to_income=func_int_to_inc,
                total_loan_to_income=func_tot_loan_to_inc
            )
        )

        return X
    
