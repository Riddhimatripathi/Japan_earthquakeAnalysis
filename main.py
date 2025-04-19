from data_cleaning import load_and_clean, save_clean
from feature_engineering import add_seismic_features
from kalman_filtering import apply_kalman_filter
from risk_classification import add_risk_zones
from anomaly_detection import detect_anomalies

def main():
    df = load_and_clean("Japan earthquakes 2001 - 2018.csv")
    save_clean(df, "cleaned.csv")
    df = add_seismic_features(df)
    df['Magnitude_Denoised'] = apply_kalman_filter(df['Magnitude'].values)
    add_risk_zones(df)
    df = detect_anomalies(df)
    df.to_csv("final_output.csv", index=False)

if __name__ == "__main__":
    main()
