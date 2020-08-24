import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style(style="whitegrid")


# boxplot for all numeric values
def boxplot_numeric_value(dataframe):
    sns.boxplot(data=dataframe, orient="h", palette="Set2")


# boxplot for single attribute single horizontal boxplot
def boxplot_single_attributes(dataframe, attribute):
    sns.boxplot(x=dataframe[attribute])
    show_plot()


# Draw a vertical boxplot grouped by a categorical variable:
def boxplot_groupby_single_category(dataframe, category_attribute, numeric_attribute):
    sns.boxplot(x=category_attribute, y=numeric_attribute, data=dataframe)
    show_plot()


# boxplot with nested grouping by two categorical variables
def boxplot_groupby_two_category(dataframe, categpry_attribute, numeric_attribute, hue):
    sns.boxplot(x=categpry_attribute, y=numeric_attribute, hue=hue, data=dataframe)
    show_plot()


# plot 3 categories for a single numeric attribute
def boxplot_catplot_three_categories(dataframe, category_attribute_one, numeric_attribute, hue, category_attribute_two):
    sns.catplot(x=category_attribute_one, y=numeric_attribute, hue=hue, col=category_attribute_two, data=dataframe)
    show_plot()


# Scatter plots

# Simple scatterplot between two variables
def scatter_plot(dataframe, x_attribute, y_attribute):
    sns.scatterplot(x=x_attribute, y=y_attribute, data=dataframe)
    show_plot()


# Scatterplot Group by another variable and show the groups with different colors:
def scatter_plot_groupby_single_category(dataframe, x_attribute, y_attribute, category):
    sns.scatterplot(x=x_attribute, y=y_attribute, hue=category, data=dataframe)
    show_plot()


# Scatterplot Vary the size with a categorical variable, and use a different palette
def scatter_plot_groupby_two_category(dataframe, x_attribute, y_attribute, category_one, category_two):
    sns.scatterplot(x=x_attribute, y=y_attribute, hue=category_one, size=category_two, data=dataframe)
    show_plot()


# S Scatterplot Groupby three attributes
def scatter_plot_groupby_three_attributes(dataframe, x_attribute, y_attribute, category_one, category_two,
                                          category_three):
    sns.relplot(x=x_attribute, y=y_attribute, hue=category_one, size=category_two, col=category_three, data=dataframe)
    show_plot()


# Simple countplot
def count_plot(dataframe, category_attribute):
    sns.countplot(x=category_attribute, data=dataframe)
    show_plot()


# Countplot Show value counts for two categorical variables
def count_plot_two_attributes(dataframe, category_attribute_one, category_attribute_two):
    sns.countplot(x=category_attribute_one, hue=category_attribute_two, data=dataframe)
    show_plot()


# render plot and return false
def show_plot():
    plt.show()
    return False
