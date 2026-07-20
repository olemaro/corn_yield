"""
Corn Yield Prediction Pipeline
================================
Dataset: https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset

Скачай ВСЕ файлы из архива и положи рядом с этим скриптом.
Обычно там: yield_df.csv, rainfall.csv, pesticides.csv, temp.csv

Install deps:
    pip install pandas scikit-learn xgboost matplotlib seaborn

Run:
    python pipeline.py
"""

import pandas
import numpy
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost
import os

# ---------------------------------------------------------------------------
# 1. Load all files and show what we have
# ---------------------------------------------------------------------------

print("=== Files in current directory ===")
for filename in os.listdir("."):
    if filename.endswith(".csv"):
        temp_df = pandas.read_csv(filename, nrows=2)
        print(f"\n{filename}: {temp_df.columns.tolist()}")

print("\n" + "="*50)

# Load yield data
yield_data = pandas.read_csv("yield_df.csv")
print(f"\nyield_df.csv shape: {yield_data.shape}")
print(yield_data.head(3))

# ---------------------------------------------------------------------------
# 2. Load and merge auxiliary files if they exist
# ---------------------------------------------------------------------------

rainfall_file = "rainfall.csv"
pesticides_file = "pesticides.csv"
temperature_file = "temp.csv"

auxiliary_dataframes = {}

if os.path.exists(rainfall_file):
    rainfall_data = pandas.read_csv(rainfall_file)
    print(f"\nrainfall.csv columns: {rainfall_data.columns.tolist()}")
    auxiliary_dataframes["rainfall"] = rainfall_data

if os.path.exists(pesticides_file):
    pesticides_data = pandas.read_csv(pesticides_file)
    print(f"\npesticides.csv columns: {pesticides_data.columns.tolist()}")
    auxiliary_dataframes["pesticides"] = pesticides_data

if os.path.exists(temperature_file):
    temperature_data = pandas.read_csv(temperature_file)
    print(f"\ntemp.csv columns: {temperature_data.columns.tolist()}")
    auxiliary_dataframes["temperature"] = temperature_data

# ---------------------------------------------------------------------------
# 3. Merge — join on Area + Year if auxiliary files exist
# ---------------------------------------------------------------------------

merged_data = yield_data.copy()

# Rename columns to standard names for merging
# The patelris dataset uses: Area, Item, Year, hg/ha_yield
# rainfall: Area, Year, average_rain_fall_mm_per_year
# pesticides: Area, Year, pesticides_tonnes
# temp: Area, Year, avg_temp

for dataset_name, auxiliary_dataframe in auxiliary_dataframes.items():
    # Standardize merge keys — drop duplicates on Area+Year
    merge_keys = [col for col in ["Area", "Year"] if col in auxiliary_dataframe.columns]
    if merge_keys:
        # Keep only numeric value columns + merge keys
        value_columns = [col for col in auxiliary_dataframe.columns if col not in ["Item"]]
        auxiliary_clean = auxiliary_dataframe[value_columns].drop_duplicates(subset=merge_keys)
        merged_data = merged_data.merge(auxiliary_clean, on=merge_keys, how="left")
        print(f"\nAfter merging {dataset_name}: {merged_data.shape}")

print(f"\nFinal columns: {merged_data.columns.tolist()}")
print(merged_data.head(3))

# ---------------------------------------------------------------------------
# 4. Filter for corn (Maize)
# ---------------------------------------------------------------------------

item_column = "Item" if "Item" in merged_data.columns else merged_data.columns[1]
unique_crops = merged_data[item_column].unique()
print(f"\nAvailable crops: {unique_crops[:15]}")

# Look for maize/corn
corn_names = [name for name in unique_crops if "maize" in name.lower() or "corn" in name.lower()]
print(f"Corn/Maize entries: {corn_names}")

if corn_names:
    corn_data = merged_data[merged_data[item_column].isin(corn_names)].copy()
else:
    print("No corn found — using all crops")
    corn_data = merged_data.copy()

print(f"\nCorn rows: {len(corn_data)}")

# ---------------------------------------------------------------------------
# 5. Convert units: hg/ha → bushels/acre
#    1 bu/acre corn = 62.77 hg/ha  (56 lbs/bu, 1 ha = 2.47105 acres)
# ---------------------------------------------------------------------------

yield_column = "hg/ha_yield"
if yield_column not in corn_data.columns:
    # Find the yield column by guessing
    yield_column = [col for col in corn_data.columns if "yield" in col.lower() or "hg" in col.lower()][0]
    print(f"Using yield column: {yield_column}")

BUSHELS_PER_HG_HA = 1 / 62.77
corn_data["yield_bushels_per_acre"] = corn_data[yield_column] * BUSHELS_PER_HG_HA

print("\nYield stats (bu/acre):")
print(corn_data["yield_bushels_per_acre"].describe())

# ---------------------------------------------------------------------------
# 6. Select features — use whatever columns are available
# ---------------------------------------------------------------------------

label_encoder = LabelEncoder()
corn_data["area_encoded"] = label_encoder.fit_transform(corn_data["Area"])

candidate_features = {
    "Year": "Year",
    "area_encoded": "area_encoded",
    "average_rain_fall_mm_per_year": "average_rain_fall_mm_per_year",
    "pesticides_tonnes": "pesticides_tonnes",
    "avg_temp": "avg_temp",
}

# Also check for alternate column names
alternate_names = {
    "average_rain_fall_mm_per_year": ["rainfall", "rain_fall", "precipitation", " average_rain_fall_mm_per_year"],
    "pesticides_tonnes": ["pesticides", "Value"],
    "avg_temp": ["temperature", "temp", "avg_temp"],
}

available_features = ["Year", "area_encoded"]
for standard_name, alternatives in alternate_names.items():
    if standard_name in corn_data.columns:
        available_features.append(standard_name)
    else:
        for alt_name in alternatives:
            if alt_name in corn_data.columns:
                corn_data[standard_name] = corn_data[alt_name]
                available_features.append(standard_name)
                print(f"Mapped {alt_name} → {standard_name}")
                break

print(f"\nFeatures used: {available_features}")
target_column = "yield_bushels_per_acre"

clean_data = corn_data[available_features + [target_column]].dropna()
print(f"Clean rows after dropna: {len(clean_data)}")

if len(clean_data) < 50:
    print("WARNING: very few rows — check your data files!")

feature_matrix = clean_data[available_features].values
target_vector = clean_data[target_column].values

# ---------------------------------------------------------------------------
# 7. Train / test split  (80 / 20)
# ---------------------------------------------------------------------------

features_train, features_test, target_train, target_test = train_test_split(
    feature_matrix, target_vector, test_size=0.2, random_state=42
)

print(f"\nTrain: {len(features_train)} rows, Test: {len(features_test)} rows")

# ---------------------------------------------------------------------------
# 8. XGBoost model
# ---------------------------------------------------------------------------

model = xgboost.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)

model.fit(features_train, target_train, verbose=False)
target_predicted = model.predict(features_test)

mean_absolute_error_value = mean_absolute_error(target_test, target_predicted)
root_mean_squared_error_value = numpy.sqrt(mean_squared_error(target_test, target_predicted))
r2_value = r2_score(target_test, target_predicted)

print("\n=== Test set metrics ===")
print(f"  MAE  : {mean_absolute_error_value:.2f} bu/acre")
print(f"  RMSE : {root_mean_squared_error_value:.2f} bu/acre")
print(f"  R²   : {r2_value:.4f}")

cross_val_scores = cross_val_score(model, feature_matrix, target_vector, cv=5, scoring="r2")
print(f"  5-fold CV R²: {cross_val_scores.mean():.4f} ± {cross_val_scores.std():.4f}")

# ---------------------------------------------------------------------------
# 9. Quantile regression (вероятностный прогноз)
# ---------------------------------------------------------------------------

quantile_models = {}
for quantile_value in [0.10, 0.50, 0.90]:
    quantile_model = xgboost.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        objective="reg:quantileerror",
        quantile_alpha=quantile_value,
        random_state=42,
        n_jobs=-1,
    )
    quantile_model.fit(features_train, target_train, verbose=False)
    quantile_models[quantile_value] = quantile_model

prediction_q10 = quantile_models[0.10].predict(features_test)
prediction_q50 = quantile_models[0.50].predict(features_test)
prediction_q90 = quantile_models[0.90].predict(features_test)

print("\n=== Probabilistic forecast (first 5 test samples) ===")
print(f"{'Actual':>10}  {'Q10':>10}  {'Q50':>10}  {'Q90':>10}")
for index in range(min(5, len(target_test))):
    print(
        f"{target_test[index]:>10.1f}  "
        f"{prediction_q10[index]:>10.1f}  "
        f"{prediction_q50[index]:>10.1f}  "
        f"{prediction_q90[index]:>10.1f}"
    )

# ---------------------------------------------------------------------------
# 10. Plots
# ---------------------------------------------------------------------------

figure, axes_array = pyplot.subplots(1, 3, figsize=(15, 5))

axes_array[0].scatter(target_test, target_predicted, alpha=0.4, s=15, color="steelblue")
min_value = min(target_test.min(), target_predicted.min())
max_value = max(target_test.max(), target_predicted.max())
axes_array[0].plot([min_value, max_value], [min_value, max_value], "r--", linewidth=1.5)
axes_array[0].set_xlabel("Actual yield (bu/acre)")
axes_array[0].set_ylabel("Predicted yield (bu/acre)")
axes_array[0].set_title(f"Predicted vs Actual  (R²={r2_value:.3f})")

feature_importance_series = pandas.Series(
    model.feature_importances_, index=available_features
).sort_values(ascending=True)
feature_importance_series.plot(kind="barh", ax=axes_array[1], color="teal")
axes_array[1].set_title("Feature Importance")
axes_array[1].set_xlabel("Importance score")

sample_count = min(50, len(target_test))
sample_indices = numpy.arange(sample_count)
axes_array[2].fill_between(
    sample_indices,
    prediction_q10[:sample_count],
    prediction_q90[:sample_count],
    alpha=0.3, color="orange", label="Q10–Q90 interval",
)
axes_array[2].plot(sample_indices, prediction_q50[:sample_count], color="darkorange", label="Q50", linewidth=1.5)
axes_array[2].scatter(sample_indices, target_test[:sample_count], color="steelblue", s=15, label="Actual", zorder=3)
axes_array[2].set_xlabel("Sample index")
axes_array[2].set_ylabel("Yield (bu/acre)")
axes_array[2].set_title("Prediction Interval (Q10/Q50/Q90)")
axes_array[2].legend(fontsize=8)

pyplot.tight_layout()
pyplot.savefig("corn_yield_results.png", dpi=150)
pyplot.show()
print("\nPlot saved → corn_yield_results.png")
