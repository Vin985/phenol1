def format_date_short(dates):
    if (dates[len(dates) - 1] - dates[0]).days > 1:
        return [x.strftime("%d") for x in dates]
    else:
        return [x.strftime("%H:%M") for x in dates]


def read_trends(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    return df
