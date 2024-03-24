from io import StringIO
from csv import writer


def generate_sequence(data, lookback, horizon, feature):
    output = StringIO()
    csv_writer = writer(output)

    fields = ['Month', 'DayType', 'Hour']
    for i in range(lookback + 1):
        if i == lookback:
            fields.append('past_t')
        else:
            fields.append('past_t-' + str(lookback - i))
    for i in range(horizon):
        fields.append('pred_t+' + str(i + 1))

    meta = [len(data), len(fields) - horizon, horizon]

    csv_writer.writerow(meta)
    csv_writer.writerow(fields)

    for i in range(lookback + 1, len(data) - horizon):
        row = list(data.loc[i, ["month", "day_type", "hour"]]) \
              + list(data[feature][(i - lookback - 1):(i + horizon)])
        csv_writer.writerow(row)
    output.seek(0)
    return output
