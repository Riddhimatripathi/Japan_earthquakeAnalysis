import pandas as pd

def convert_to_mw(mag, magType):
    if pd.isna(mag) or pd.isna(magType):
        return mag
    magType = magType.lower()
    if magType == 'mww':
        return mag
    elif magType == 'mwr':
        return mag + 0.1
    elif magType == 'mb':
        return (mag * 0.85) + 1.03
    return mag

def load_and_clean(infile: str) -> pd.DataFrame:
    df = pd.read_csv(infile)
    df['Magnitude'] = df.apply(lambda row: convert_to_mw(row['mag'], row['magType']), axis=1)
    df['time'] = pd.to_datetime(df['time'])
    df['Date'] = df['time'].dt.date
    df['Time'] = df['time'].dt.strftime('%H:%M:%S')
    drop_cols = ['time','mag','magType','nst','id','type','status','magSource','locationSource','net','updated']
    df = df.drop(columns=drop_cols, errors='ignore')
    numeric_cols = ['latitude','longitude','depth','Magnitude','gap','dmin','rms',
                    'horizontalError','depthError','magError','magNst']
    df[numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.mean())).round(2)
    return df

def save_clean(df: pd.DataFrame, outfile: str):
    df.to_csv(outfile, index=False)
