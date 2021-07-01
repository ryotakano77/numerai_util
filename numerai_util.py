import csv
import os
import pickle
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# Submissions are scored by spearman correlation
def correlation(predictions, targets):
    ranked_preds = predictions.rank(pct=True, method="first")
    return np.corrcoef(ranked_preds, targets)[0, 1]

def max_drawdown(correlations):
    max_so_far = (correlations + 1).cumprod().cummax()
    daily_value = (correlations + 1).cumprod()
    return -(max_so_far - daily_value).max()
    
def feature_exposures(df):
    """
    Compute feature exposure
    
    :INPUT:
    - valid_df : pd.DataFrame where at least 2 columns ('prediction' & 'target') exist
    """
    feature_names = [f for f in df.columns
                     if f.startswith("feature")]
    exposures = []
    for f in feature_names:
        fe = spearmanr(df['prediction'], df[f])[0]
        exposures.append(fe)
    return np.array(exposures)

def max_feature_exposure(fe : np.ndarray):
    return np.max(np.abs(fe))

def feature_exposure(fe : np.ndarray):
    return np.sqrt(np.mean(np.square(fe)))

# convenience method for scoring
def score(df, prediction_name: str="pred", target_name: str="target"):
    return correlation(df[prediction_name], df[target_name])

# Payout is just the score cliped at +/-25%
def payout(scores):
    return scores.clip(lower=-0.25, upper=0.25)

# Read the csv file into a pandas Dataframe as float16 to save space
def read_csv(file_path):
    with open(file_path, 'r') as f:
        column_names = next(csv.reader(f))

    dtypes = {x: np.float16 for x in column_names if x.startswith(('feature', 'target'))}
    df = pd.read_csv(file_path, dtype=dtypes, index_col=0)
    return df

def read_csv_memory_friendrily(file_path):
    with open(file_path, 'r') as f:
        column_names = next(csv.reader(f))
    dtypes = {f"target": np.float16}
    to_uint8 = lambda x: np.uint8(float(x) * 4)
    converters = {x: to_uint8 for x in column_names if x.startswith('feature')}
    df = pd.read_csv(file_path, dtype=dtypes, converters=converters)
    return df

# to neutralize a column in a df by many other columns on a per-era basis
def neutralize(df,
               columns,
               extra_neutralizers=None,
               proportion=1.0,
               normalize=True,
               era_col="era"):
    # need to do this for lint to be happy bc [] is a "dangerous argument"
    if extra_neutralizers is None:
        extra_neutralizers = []
    unique_eras = df[era_col].unique()
    computed = []
    for u in unique_eras:
        print(u, end="\r")
        df_era = df[df[era_col] == u]
        scores = df_era[columns].values
        if normalize:
            scores2 = []
            for x in scores.T:
                x = (pd.Series(x).rank(method="first").values - .5) / len(x)
                scores2.append(x)
            scores = np.array(scores2).T
            extra = df_era[extra_neutralizers].values
            exposures = np.concatenate([extra], axis=1)
        else:
            exposures = df_era[extra_neutralizers].values

        scores -= proportion * exposures.dot(
            np.linalg.pinv(exposures.astype(np.float32)).dot(scores.astype(np.float32)))

        scores /= scores.std(ddof=0)

        computed.append(scores)

    return pd.DataFrame(np.concatenate(computed),
                        columns=columns,
                        index=df.index)

# to neutralize any series by any other series
def neutralize_series(series, by, proportion=1.0):
    scores = series.values.reshape(-1, 1)
    exposures = by.values.reshape(-1, 1)

    # this line makes series neutral to a constant column so that it's centered and for sure gets corr 0 with exposures
    exposures = np.hstack(
        (exposures,
         np.array([np.mean(series)] * len(exposures)).reshape(-1, 1)))

    correction = proportion * (exposures.dot(
        np.linalg.lstsq(exposures, scores, rcond=None)[0]))
    corrected_scores = scores - correction
    neutralized = pd.Series(corrected_scores.ravel(), index=series.index)
    return neutralized

def unif(df):
    x = (df.rank(method="first") - 0.5) / len(df)
    return pd.Series(x, index=df.index)

def get_feature_neutral_mean(df, prediction_name: str="pred", target_name: str="target"):
    feature_cols = [c for c in df.columns if c.startswith("feature")]
    df.loc[:, "neutral_sub"] = neutralize(df, [prediction_name],
                                          feature_cols)[prediction_name]
    scores = df.groupby("era").apply(
        lambda x: correlation(x["neutral_sub"], x[target_name])).mean()
    return np.mean(scores)

def read_data(data):
    # get data 
    if data == 'train':
        df = pd.read_csv('https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_training_data.csv.xz')
    elif data == 'test':
        df = pd.read_csv('https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_tournament_data.csv.xz')
    return df

def get_era(era: str) -> int:
    return int(era[3:])

def load_or_predict(pred_from: str, model_name: str, base_path: str="/Users/ryo/Documents/numerai/numerai_datasets") -> pd.DataFrame:
    pred_dir_path = os.path.join(base_path, "predictions")
    pred_dir_path = os.path.join(pred_dir_path, pred_from)
    file_name = model_name + ".csv"
    file_path = os.path.join(pred_dir_path, file_name)

    if os.path.exists(file_path):
        print(f"{model_name}'s prediction exists. Now loading...")
        return pd.read_csv(file_path, index_col=0)

    else:
        print(f"{model_name}'s prediction does not exist. Now loading model and predicting...")
        model_path = os.path.join(base_path, "models")
        model_path = os.path.join(model_path, model_name+".pkl")
        features_path = os.path.join(base_path, "data")
        features_path = os.path.join(features_path, pred_from+".pkl")
        with open(features_path, "rb") as f:
            df_features = pickle.load(f)
        feature_names = [f for f in df_features.columns if f.startswith("feature")]
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print("predicting...")
        df_features["prediction"] = model.predict(df_features[feature_names])
        print("predicted!")
        df_features.to_csv(file_path, header=True)
        return df_features