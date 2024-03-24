from datetime import datetime, timedelta


def summer(dt: datetime):
    return 6 <= dt.month <= 9


def weekend(dt: datetime):
    return dt.weekday() >= 5


def peak(dt: datetime):
    return 16 <= dt.hour <= 20


def electricity_price(dt: datetime):
    if summer(dt):
        if peak(dt):
            if weekend(dt):
                return 0.4
            else:
                return 0.54
        else:
            return 0.22
    else:
        if peak(dt):
            return 0.5
        else:
            return 0.21


def generate_24_hours(start_date: datetime):
    return [start_date + timedelta(hours=i) for i in range(1, 25)]


def generate_24_price(start_date: datetime):
    return [electricity_price(d) for d in generate_24_hours(start_date)]
