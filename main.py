from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

unconventional_data = {}
feature_names = []

# Converter of column values into classes for use in
# the algorithm model of the linear regression programme
def classifier(col, data):
    column_data = set(data[col].tolist())
    column_data = [(a, list(column_data).index(a)) for a in column_data]
    unconventional_data[col] = column_data
    data[col] = data[col].replace(list(map(lambda x: x[0], column_data)), list(map(lambda x: x[1], column_data)))
    return data[col]


def xtractor_num(col, val):
    return list(filter(lambda x: x[0] == val, unconventional_data[col]))[0][1]


def xtractor_str(col, val):
    return list(filter(lambda x: x[1] == val, unconventional_data[col]))[0][0]

# Column-based dataset filter by status
def list_by_state(end_index, state, list_data, index, unconv):
    return list(map(lambda x: xtractor_str(feature_names[i], x[index]) if unconv else x[index],
                    list(filter(lambda x: x[end_index] == state, list_data))))


# Function used to generate graphs and
# graphic illustrations for the programme
def graph(data, i, bins, unconv):
    ei = -5
    print("based on : ", feature_names[i])

    my_list = data.values.tolist()

    # Assume we have two lists: pledge amounts for successful and failed campaigns
    success_list = list_by_state(ei, 'successful', my_list, i, unconv)
    fail_list = list_by_state(ei, 'failed', my_list, i, unconv)
    cancel_list = list_by_state(ei, 'canceled', my_list, i, unconv)
    undefined_list = list_by_state(ei, 'undefined', my_list, i, unconv)
    suspend_list = list_by_state(ei, 'suspended', my_list, i, unconv)

    # Create histograms
    plt.hist(success_list, bins=bins, alpha=0.5, label='Successful Campaigns')
    plt.hist(fail_list, bins=bins, alpha=0.5, label='Failed Campaigns')
    plt.hist(cancel_list, bins=bins, alpha=0.5, label='Canceled Campaigns')
    plt.hist(undefined_list, bins=bins, alpha=0.5, label='Undefined Campaigns')
    plt.hist(suspend_list, bins=bins, alpha=0.5, label='Suspended Campaigns')

    # Add a title
    plt.title("{0} graph".format(feature_names[i]))

    # Add a legend
    plt.legend(loc='upper right')

    # Show the plot
    plt.show()

# Function used to normalise and standardise the dates
# in the datatset by converting them to the same format
def date_normalizer(data, col):
    dates = []
    for date in data[col]:
        try:
            dates.append(datetime.strptime(date, '%Y-%m-%d %H:%M:%S'))
        except:
            dates.append(datetime.strptime(date, '%d/%m/%Y %H:%M'))
    return dates


if __name__ == '__main__':
    # Data loading form the dataset source located in ./ks_dataset.csv
    # with encoding specification at [latin-1]
    # and parameter low_memory=False to avoid computer to limit the size of memory use for file fetching
    print("Starting data loading -", datetime.now().strftime('%d, %b %Y %H:%M:%S'))
    data = pd.read_csv("ks_dataset.csv", encoding='latin-1', low_memory=False)
    data.head()
    data = data.dropna()

    print("End of data loading process -", datetime.now().strftime('%d, %b %Y %H:%M:%S'))
    print("===================================")

    # normalization of columns launched format
    # that are using string value to classes values
    data['country'] = classifier('country', data)
    data['currency'] = classifier('currency', data)
    data['category'] = classifier('category', data)
    data['main_category'] = classifier('main_category', data)

    # normalization of date launched format
    # to a singular and unique date format and store it in a list
    data['launched'] = date_normalizer(data, 'launched')
    data['deadline'] = date_normalizer(data, 'deadline')

    # creation of another column that will store the difference number betweeen date
    data['time'] = (data['deadline'] - data['launched']).dt.components['days']

    # initialization of columns list of dataset
    feature_names = data.columns.tolist()

    # selection of dependants variables
    deps_columns = ['goal', 'backers', 'pledged', 'usd_pledged', 'country', 'currency', 'category', 'main_category',
                    'time']

    # Preprocessing data stated
    scale = StandardScaler()
    print("Preprocessing data in progress -", datetime.now().strftime('%d, %b %Y %H:%M:%S'))
    X_data = scale.fit_transform(data[deps_columns])
    Y_data = data['state']
    print("End of data preprocessing -", datetime.now().strftime('%d, %b %Y %H:%M:%S'))
    print("===================================")
    # Preprocessing data finished

    # Separation of data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.2)

    # Model drive or training
    model = LogisticRegression(max_iter=20000, C=1000, solver='newton-cg')
    model.fit(X_train, y_train)

    # Predictions on the test data set
    y_pred = model.predict(X_test)

    # test with new data not included in differents sets
    new_data = pd.DataFrame(
        data={
            'goal': [5000],
            'backers': [2],
            'pledged': [579],
            'usd_pledged': [579],
            'country': [xtractor_num('country', 'US')],
            'currency': [xtractor_num('currency', 'USD')],
            'category': [xtractor_num('category', "Children's Books")],
            'main_category': [xtractor_num('main_category', 'Publishing')],
            'time': [30]
        })

    # standardization of the new preprocessed
    # value that we want to predict
    std_data = scale.transform(new_data)
    result = model.predict(std_data)

    print("Prediction for this model is :", result[0])

    # Evaluation of the model with percentage of precision
    accuracy_score = model.score(X_test, y_test)
    print("Précision :", accuracy_score)
    print("\n")

    # initialization of selected columns for graph and plots generations
    # selected_cols = [2, 3, 4, 6, 11, 13]
    selected_cols = [2, 3, 4, 6, 8, 10, 11, 12, 13]
    for i in selected_cols:
        # determination of length of x-axis for the plot based on the length of the category
        try:
            unconventional = True
            bins = len(unconventional_data[feature_names[i]])
        except:
            unconventional = False
            bins = 100

        graph(data, i, bins, unconventional)
