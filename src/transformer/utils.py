from datetime import datetime, timezone, timedelta

import pandas as pd


def get_timestamp():
    return pd.Timestamp.today(
        tz=datetime.now(timezone(timedelta(0))).astimezone().tzinfo
    ).strftime('%Y-%m-%d_%H.%M')
