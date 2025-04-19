import numpy as np

def add_seismic_features(df):
    df['Amplitude']      = 10 ** (df['Magnitude'] + 3)
    df['Frequency']      = 10 ** (3 - df['Magnitude'])
    df['Velocity']       = 3.5 * (10 ** (0.5 * df['Magnitude']))
    df['Acceleration']   = df['Velocity'] * (2 * np.pi * df['Frequency'])
    df['Seismic_Energy'] = 10 ** (1.5 * df['Magnitude'] + 4.8)
    df['Seismic_Moment'] = 10 ** (1.5 * df['Magnitude'] + 9.1)
    cols = ['Amplitude','Frequency','Velocity','Seismic_Energy','Seismic_Moment']
    df[cols] = df[cols].round(2)
    return df
