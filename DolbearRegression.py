from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# remove all data but temp and chirp rate
def preprocess(df):
    data = df.drop("Day", axis=1)
    data.drop("Location", axis=1, inplace=True)
    data.drop("Type", axis=1, inplace=True)
    data.drop("Time", axis=1, inplace=True)
    data.drop("CricketNumber", axis=1, inplace=True)
    data.drop("RecNum", axis=1, inplace=True)

    return data


# fit linear regression only on filtered samples, and plot results
def fit_and_plot(data, filter, filter_value, purpose):
    lr = LinearRegression()
    if filter:
        data = data.loc[data[filter] == filter_value]
    data = preprocess(data)

    if purpose == "Regression":
        lr.fit(data["ChirpPer15Sec"].to_numpy().reshape(-1, 1), data["Temp"].to_numpy().reshape(-1, 1))
        mae = metrics.mean_absolute_error(
            lr.coef_[0][0] * data["ChirpPer15Sec"].to_numpy().reshape(-1, 1) + lr.intercept_[0],
            data["Temp"].to_numpy().reshape(-1, 1))
        print("slope is", lr.coef_[0][0], " intercept is", lr.intercept_, "MAE is", mae)

        line = px.line(x=np.arange(100), y=(lr.coef_[0][0] * np.arange(100) + lr.intercept_[0]))

    else:
        mae = metrics.mean_absolute_error((5 / 9) * data["ChirpPer15Sec"].to_numpy().reshape(-1, 1) + 40 / 9,
                                          data["Temp"].to_numpy().reshape(-1, 1))
        print("MAE is", mae)

        line = px.line(x=np.arange(100), y=(5 / 9 * np.arange(100) + 40 / 9))

    print_val = dict()
    print_val[1] = "Unidentified"
    print_val[2] = "Field: spectrogram 1"
    print_val[3] = "Field: spectrogram 2"

    scatter = px.scatter(data, x="ChirpPer15Sec", y="Temp", color_discrete_sequence=["Blue"])
    line.update_traces(line_color="red")
    title = ["Unidentified Chirp", "Gryllus Bimaculatus type I", "Gryllus Bimaculatus type II"]
    layout = go.Layout(
        title=title[filter_value - 1] + ", MAE: " + str(np.round(mae, 2)),
        xaxis=dict(title="ChirpPer15Sec"),
        yaxis=dict(title="Temp"),
        yaxis_range=[18, 28]
    )
    fig = go.Figure(data=scatter.data + line.data, layout=layout)
    fig.show()


# test unfitted samples and print mae
def test(train, test, spec):
    # test unidentified:
    lr = LinearRegression()
    data1 = train.loc[train["Spectrogram"] == spec]
    data1 = preprocess(data1)
    lr.fit(data1["ChirpPer15Sec"].to_numpy().reshape(-1, 1), data1["Temp"].to_numpy().reshape(-1, 1))

    test1 = test.loc[test["Spectrogram"] == spec]
    test1 = preprocess(test1)
    pred_temp = lr.predict(test1["ChirpPer15Sec"].to_numpy().reshape(-1, 1))

    mae = metrics.mean_absolute_error(test1["Temp"].to_numpy().reshape(-1, 1), pred_temp)

    st = ["Unidentified", "GB Type I"]
    print("Test MAE for " + st[spec - 1] + " is " + str(mae))


if __name__ == "__main__":
    # load and preprocess
    crickets_data = pd.read_csv("crickets_chirp.csv")

    # fit_and_plot(crickets_data, "Spectrogram", 1, "Dolbear")
    # fit_and_plot(crickets_data, "Spectrogram",2, "Dolbear")
    # fit_and_plot(crickets_data, "Spectrogram", 3, "Dolbear")
    #
    # fit_and_plot(crickets_data, "Spectrogram", 1, "Regression")
    # fit_and_plot(crickets_data, "Spectrogram", 2, "Regression")
    # fit_and_plot(crickets_data, "Spectrogram", 3, "Regression")

    # test model
    test_data = pd.read_csv("crickets_chirp_test.csv")
    test(crickets_data, test_data, 1)
    test(crickets_data, test_data, 2)
