import logging

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklego.pandas_utils import log_step

logging.basicConfig(level=logging.DEBUG)



def label_encoder(train, test, columns):
    train, test = train.copy(), test.copy()
    
    for col in columns:
        train_array = list(train[col].values.astype('str'))
        test_array = list(test[col].values.astype('str'))

        lb = LabelEncoder()
        lb.fit(train_array + test_array)
        train[col] = lb.transform(train_array)
        test[col] = lb.transform(test_array)

    return train, test

@log_step(level=logging.DEBUG)
def initialize_pipeline(df):
    return df.copy()

def convert_to_datetime(df):
    df = (
        df.assign(issue_d=lambda x: pd.to_datetime(x['issue_d']),
                  earliest_cr_line=lambda x: pd.to_datetime(x['earliest_cr_line']))
    )

    return df

def correct_dtypes(df, cols, types):
    for col, type_ in zip(cols,  types):
        df[col] = df[col].astype(type_)
    
    return df

@log_step(level=logging.DEBUG)
def remove_trailing_spaces(df, column):
    df[column] = (
        df[column]
            .str.strip()
            .str.extract(r'(\d+)').astype(int)
    )

    return df

@log_step
def filter_not_default_or_paid_loans(df):
    return (
        df[~(df["loan_status"].isin(["Current", "In Grace Period", "Late (16-30 days)"]))]
    )

@log_step
def binarize_loan_status(df):
    df["loan_status_general"] = (
        df["loan_status"].apply(lambda x: "fully_paid" 
                                          if "Fully Paid" in x
                                          else "default")
    )

    return df

def filter_columns(df, columns):
    return df.drop(columns, axis=1)

@log_step
def calculate_total_loan_amount(df):    
    df["total_loan_amount"] = (
        df["installment"]*((1+(df["int_rate"]/100))**(df["term"] / 12))
    )

    return df

@log_step
def calculate_credit_time_years(df):
    df["credit_time_in_years"] = (
        np.round((df['issue_d'] - pd.to_datetime(df["earliest_cr_line"])) / np.timedelta64(1, 'Y'), 2)
    )

    return df

def calculate_inst_to_income(df):
    df["installment_to_income"] = (
        (df["installment"]*12) / (df["annual_inc"]+1)
        )

    return df

def calculate_tot_loan_to_inc(df):
    df["total_loan_to_income"] = (
        (df["total_loan_amount"] / (df["annual_inc_cap"]*(df["term"] / 12)))*100
    )

    return df


