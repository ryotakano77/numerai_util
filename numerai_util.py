import csv
import os
import pickle
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from PIL import Image
import matplotlib.pyplot as plt

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

def calc_stats(df: pd.DataFrame):
    stats = {}
    target = df["target"]
    prediction = df["prediction"]

    stats["peason_corr"] = np.corrcoef(target, prediction)[0, 1]
    stats["corr"] = correlation(target, prediction)
    corr_by_era = df.groupby("era")[["target", "prediction"]].corr().iloc[0::2, -1].reset_index()["prediction"]
    stats["max_dd"] = max_drawdown(corr_by_era)
    stats["corr_mean"] = corr_by_era.mean()
    stats["corr_std"] = corr_by_era.std()
    stats["sharp"] = stats["corr_mean"] / stats["corr_std"]
    fes = feature_exposures(df)
    stats["feature_exposure"] = feature_exposure(fes)
    stats["max_feature_exposure"] = max_feature_exposure(fes)
    return pd.Series(stats)

def calc_corr_by_era(df: pd.DataFrame):
    corr_by_era = df.groupby("era")[["target", "prediction"]].corr().iloc[0::2, -1].reset_index()[["era", "prediction"]]
    era_name = ["era" + str(e) for e in corr_by_era["era"]]
    corr_s = pd.Series(corr_by_era["prediction"])
    corr_s.index = era_name
    return corr_s

def calc_all_stats(df: pd.DataFrame):
    stats_val_all = calc_stats(df)
    stats_val1 = calc_stats(df.query("era < 150"))
    stats_val2 = calc_stats(df.query("era > 150"))
    corr_by_era = calc_corr_by_era(df)
    stats_val_all.rename(index=lambda x: x+"_val_all", inplace=True)
    stats_val1.rename(index=lambda x: x+"_val_1", inplace=True)
    stats_val2.rename(index=lambda x: x+"_val_2", inplace=True)
    stats = pd.concat([stats_val_all, stats_val1, stats_val2, corr_by_era])
    df_stats = pd.DataFrame([stats])
    return df_stats

def load_data_and_calc_stats(model_name: str):
    prediction_dir = "/Users/ryo/Documents/numerai/numerai_datasets/predictions/validation_all"
    prediction_file = model_name + ".csv"
    prediction_path = os.path.join(prediction_dir, prediction_file)
    df_pred = pd.read_csv(prediction_path, index_col=0)
    df_pred["era"] = df_pred["era"].map(get_era)
    df_stats = calc_all_stats(df_pred)
    return df_stats.rename(index={0: model_name})

def get_im_from_df(df: pd.DataFrame, feature_idx: int, era_idx: int):
    counts_by_feature = 5
    col_start_idx = counts_by_feature * feature_idx
    col_end_idx   = counts_by_feature * (feature_idx + 1)

    counts = df.iloc[era_idx, col_start_idx : col_end_idx]

    fig, ax = plt.subplots()
    ax.bar([0, 0.25, 0.5, 0.75, 1], counts, width=0.1)
    ax.set_title(counts.index[0][0])
    fig.canvas.draw()
    im = np.array(fig.canvas.renderer.buffer_rgba())

    return Image.fromarray(im)

def gif_from_im_list(im_list, file_path, duration=5):
    return im_list[0].save(file_path, save_all=True, append_images=im_list[1:], optimize=False, duration=duration)