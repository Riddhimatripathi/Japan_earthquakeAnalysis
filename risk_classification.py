def assign_risk_zone(mag: float) -> str:
    if mag < 5.0:
        return "Low"
    elif mag < 6.0:
        return "Medium"
    return "High"

def add_risk_zones(df) -> None:
    df['Risk_Zone'] = df['Magnitude'].apply(assign_risk_zone)
