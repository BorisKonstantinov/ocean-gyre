import xarray
import pandas
import numpy

from sklearn.preprocessing import StandardScaler


# --- Load Data ---
def load_data():
    # Load data individually per variable
    ds_t2m = xarray.open_dataset(
        "2425Ex.grib", engine="cfgrib", filter_by_keys={"shortName": "2t"}
    )  # Its actually 23' and 24'
    ds_d2m = xarray.open_dataset(
        "2425Ex.grib", engine="cfgrib", filter_by_keys={"shortName": "2d"}
    )
    ds_sp = xarray.open_dataset(
        "2425Ex.grib", engine="cfgrib", filter_by_keys={"shortName": "sp"}
    )
    ds_tcc = xarray.open_dataset(
        "2425Ex.grib", engine="cfgrib", filter_by_keys={"shortName": "tcc"}
    )
    ds_ssr = xarray.open_dataset(
        "2425Ex.grib", engine="cfgrib", filter_by_keys={"shortName": "ssr"}
    )
    # Convert to Pandas DataFrame
    df_t2m = ds_t2m.to_dataframe()
    df_d2m = ds_d2m.to_dataframe()
    df_sp = ds_sp.to_dataframe()
    df_tcc = ds_tcc.to_dataframe()
    df_ssr = ds_ssr.to_dataframe()
    # Drop clutter
    df_t2m = df_t2m.drop(columns=["number", "step", "surface", "valid_time"])
    df_d2m = df_d2m.drop(columns=["number", "step", "surface", "valid_time"])
    df_sp = df_sp.drop(columns=["number", "step", "surface", "valid_time"])
    df_tcc = df_tcc.drop(columns=["number", "step", "surface", "valid_time"])
    # Drop lat/lon from index
    df_t2m.index = df_t2m.index.droplevel(["latitude", "longitude"])
    df_d2m.index = df_d2m.index.droplevel(["latitude", "longitude"])
    df_sp.index = df_sp.index.droplevel(["latitude", "longitude"])
    df_tcc.index = df_tcc.index.droplevel(["latitude", "longitude"])
    # Solar Radiation needs special handling
    df_ssr = df_ssr[["valid_time", "ssr"]].reset_index(drop=True)
    df_ssr = df_ssr.set_index("valid_time")
    df_ssr.index.name = "time"
    df_ssr = df_ssr[(df_ssr.index >= "2023-01-01") & (df_ssr.index < "2024-12-30")]
    # Merge all dataframes into a master
    df = pandas.concat([df_t2m, df_d2m, df_sp, df_tcc, df_ssr], axis=1)
    df = df[(df.index >= "2023-01-01") & (df.index < "2024-12-29")]
    # Preview data information
    print("Description \n", df.describe(), "\n", "-" * 70)
    print("Correlation matrix \n", df.corr(numeric_only=True), "\n", "-" * 70)
    print("Data head \n", df.head(), "\n", "-" * 70)
    return df


def structure_data(df):
    # Convert T[K] -> T[C]
    df["t2m"] = df["t2m"] - 273.15
    df["d2m"] = df["d2m"] - 273.15
    # Make time cyclical
    df["hour_sin"] = numpy.sin(2 * numpy.pi * df.index.hour / 24.0)
    df["hour_cos"] = numpy.cos(2 * numpy.pi * df.index.hour / 24.0)
    df["day_sin"] = numpy.sin(2 * numpy.pi * df.index.dayofyear / 365.25)
    df["day_cos"] = numpy.cos(2 * numpy.pi * df.index.dayofyear / 365.25)
    # Define features X and prediction y(X)
    X = df[
        ["t2m", "d2m", "sp", "tcc", "ssr"]
    ]  # , "hour_sin", "hour_cos", "day_sin", "day_cos"]]
    y = df[["t2m", "d2m", "sp", "tcc", "ssr"]].shift(-6)
    # Reshape X and y to match
    X = X.iloc[:-6]
    y = y.iloc[:-6]
    return X, y


def prep_data(X, y):
    # Create a cronological testing & training split
    split = int(0.79 * len(X))
    X_train = X.iloc[:split]
    y_train = y.iloc[:split]
    X_test = X.iloc[split:]
    y_test = y.iloc[split:]
    # Scale the data with mean 0, std 1
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    # Test data is scaled with training data fit
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)
    # Convert y to a NumPy array from a DataFrame
    y_train = y_train.values
    y_test = y_test.values
    return (
        X_train,
        y_train,
        X_test,
        y_test,
        X_scaler,
        y_scaler,
        X_train_scaled,
        X_test_scaled,
        y_train_scaled,
        y_test_scaled,
    )
