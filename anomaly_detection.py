from sklearn.ensemble import IsolationForest

def detect_anomalies(df, contamination=0.05):
    numeric = df.select_dtypes(include='number')
    iso = IsolationForest(contamination=contamination, random_state=42)
    df['Anomaly'] = iso.fit_predict(numeric)
    df['Anomaly'] = df['Anomaly'].map({1: 'Normal', -1: 'Anomaly'})
    return df
