from DataPreprocessfile import *
import tensorflow as tf
from tensorflow import keras
from DataVisualizaton import *

train_file = 'train.csv'
test_file = 'test.csv'
desired_width = 320

pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 15)


def main():
    feature_columns = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
    target_column = ['Survived']
    input_df = load_file(train_file)

    # Data Viz (Visualize Data to explore data and do analysis
    # boxplot_single_attributes(df, 'Age')
    # boxplot_groupby_single_category(df,'Embarked','Fare')
    # boxplot_groupby_two_category(df,'Embarked','Age','Pclass')
    # boxplot_catplot_three_categories(df, 'Embarked', 'Age', 'Pclass', 'Parch')
    # count_plot(df, 'Parch')
    # count_plot_two_attributes(df,'Pclass','Sex')
    # boxplot_groupby_single_category(df,'Sex','Age')
    # boxplot_groupby_two_category(df, 'Pclass', 'Age', 'Sex')

    df = data_preprocessing(input_df)
    features_data, label_data = features_label_split(df, target_column, False)

    # Split in Train and validation Set
    x_features, validation_features, x_labels, validation_labels = test_train_split(features_data, label_data, 0.2)

    # build model with training set
    # model = model_build_train(x_features, x_labels)

    # model test with validation set
    # evaluate_model(model, validation_features, validation_labels)

    # if satisfied with accuracy train with entire set
    model = model_build_train(features_data, label_data)

    # Generate results
    get_predictions(model, test_file, feature_columns, target_column)


def get_predictions(model, test_file, feature_columns, target_column):
    test_input_data = load_file(test_file)
    test_data = data_preprocessing(test_input_data)
    p_survived = model.predict_classes(test_data.values)
    final_op = pd.DataFrame()
    final_op['PassengerId'] = test_input_data['PassengerId']
    final_op['Survived'] = p_survived
    final_op.to_csv('submission.csv', index=False)

    # model building


def model_build_train(X_features, X_labels):
    model = keras.Sequential()
    model.add(keras.layers.Dense(30, input_dim=X_features.shape[1]))
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(2))
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(X_features, X_labels, epochs=100)
    return model


# Testing the model

def evaluate_model(model, validation_features, validation_labels):
    test_loss, test_acc = model.evaluate(validation_features, validation_labels, verbose=2)
    print('Test_loss :', test_loss)
    print('Test_acc :', test_acc)


# preprocessing steps
# remove unwanted columns
# replace null values by mean
# change categorical to numeric
# Scale down
# return dataframe
def data_preprocessing(df):
    df = drop_columns(df, ['PassengerId', 'Name', 'Ticket', 'Cabin'])
    df = replace_null_by_mean(df, 'Age')
    df = replace_null_by_mean(df, 'Fare')
    df = replace_null_value(df, 'Embarked', 'S')
    # Change categorical to numeric
    df = pd.get_dummies(df, columns=['Pclass', 'Embarked'])
    # change male and female to 0 and 1
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}).astype(int)
    return df


def features_label_split(df, label_col, is_test_data):
    if not is_test_data:
        features_data = get_all_columns_except(df, label_col)
        label_col = get_subset_columns(df, label_col)
    else:
        features_data = df
        label_col = pd.DataFrame()
    return features_data, label_col


if __name__ == '__main__':
    main()
