#script for exploratory data analysis (EDA)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from preprocess_data import load_data

database_numeric, info_dict = load_data()

# count and print the nans in each column
print('NANS IN EACH COLUMN')
print(database_numeric.isnull().sum().to_string())


nongenetic_df = database_numeric.copy()
nongenetic_df = nongenetic_df.drop(info_dict['genetic_vars'], axis=1)




sns.heatmap(nongenetic_df.isnull(),
            yticklabels=False,
            cbar=False,
            cmap='magma')
plt.tight_layout()
plt.show()

# now with genetic variables
genetic_df = database_numeric.copy()
genetic_df = genetic_df.loc[:, info_dict['genetic_vars']]

fig = plt.figure(figsize=(20, 8))
sns.heatmap(genetic_df.isnull(),
            yticklabels=False,
            cbar=False,
            cmap='magma')
plt.tight_layout()
plt.show()


#CORRELATION MATRIX
fig = plt.figure(figsize=(65,65))
# Plot the correlation matrix
correlation_matrix = database_numeric.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()

#CORRELATION MATRIX FOR TREATMENTS
fig = plt.figure(figsize=(65,10))
sns.heatmap(correlation_matrix.loc[info_dict['treatments'],:], annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Bar plots of binary variables
#first non genetic
fig, axs = plt.subplots(2, 3, figsize=(20, 10))
cont_i = 0
for col in info_dict['binary_nongenetic_vars']:
    sns.countplot(x=col, data=database_numeric, ax=axs[cont_i // 3, cont_i % 3])
    cont_i += 1
fig.suptitle('Non genetic BINARY variables')
plt.tight_layout()
plt.show()


fig, axs = plt.subplots(info_dict['number_gens'] // 3, 3, figsize=(20, 65))
cont_i = 0
for col in info_dict['genetic_vars']:
    sns.countplot(x=col, data=database_numeric, ax=axs[cont_i // 3, cont_i % 3])
    cont_i += 1
fig.suptitle('Genetic BINARY variables')
plt.tight_layout()
plt.show()


fig, axs = plt.subplots(info_dict['number_cat'] // 3 + 1, 3, figsize=(20, 10))
cont_i = 0
for col in info_dict['all_cat']:
    sns.countplot(x=col, data=database_numeric, ax=axs[cont_i // 3, cont_i % 3])
    cont_i += 1
fig.suptitle('Categorical variables')
plt.tight_layout()
plt.show()


def plot_3chart(df, feature):
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import MaxNLocator
    from scipy.stats import norm
    from scipy import stats

    # Creating a customized chart. and giving in figsize and everything.

    fig = plt.figure(constrained_layout=True, figsize=(12, 8))

    # Creating a grid of 3 cols and 3 rows.

    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)

    # Customizing the histogram grid.

    ax1 = fig.add_subplot(grid[0, :2])

    # Set the title.

    ax1.set_title('Histogram')

    # Plot the histogram.

    sns.distplot(df.loc[:, feature],
                 hist=True,
                 kde=True,
                 fit=norm,
                 ax=ax1,
                 color='#e74c3c')
    ax1.legend(labels=['Normal', 'Actual'])

    # Customizing the QQ_plot.

    ax2 = fig.add_subplot(grid[1, :2])

    # Set the title.

    ax2.set_title('Probability Plot')

    # Plotting the QQ_Plot.

    stats.probplot(df.loc[:, feature].fillna(np.mean(df.loc[:, feature])),
                   plot=ax2)
    ax2.get_lines()[0].set_markerfacecolor('#e74c3c')
    ax2.get_lines()[0].set_markersize(12.0)

    # Customizing the Box Plot.

    ax3 = fig.add_subplot(grid[:, 2])

    # Set title.

    ax3.set_title('Box Plot')

    # Plotting the box plot.

    sns.boxplot(df.loc[:, feature], orient='v', ax=ax3, color='#e74c3c')
    ax3.yaxis.set_major_locator(MaxNLocator(nbins=24))

    plt.suptitle(f'{feature}', fontsize=24)


for col in info_dict['continuous_vars']:
    plot_3chart(database_numeric, col)
    plt.tight_layout()
    plt.show()


# Check overlaping of treatments with other variables
#for categorical variables
for treatment in info_dict['treatments']:
    fig, axs = plt.subplots(3, 3, figsize=(20, 10))
    cont_i = 0
    for col in info_dict['all_cat']:
        sns.countplot(x=col, hue=treatment, data=database_numeric, ax=axs[cont_i // 3, cont_i % 3])
        cont_i += 1
    fig.suptitle(treatment)
    plt.tight_layout()
    plt.show()

# for continuous variables
number_cont = len(info_dict['continuous_vars'])
for treatment in info_dict['treatments']:
    fig, axs = plt.subplots(number_cont // 3, 3, figsize=(20, 45))
    cont_i = 0
    for col in info_dict['continuous_vars']:
        sns.boxplot(x=treatment, y=col, data=database_numeric, ax=axs[cont_i // 3, cont_i % 3])
        cont_i += 1
    fig.suptitle(treatment)
    plt.tight_layout()
    plt.show()

# are treatment excluyent?
# return the count of each combination of treatments
treatments_df = database_numeric.loc[:, info_dict['treatments']]
treatments_df['count'] = 1
treatments_df = treatments_df.groupby(info_dict['treatments']).count()
print(treatments_df)

# for binary variables
# nongenetic
for treatment in info_dict['treatments']:
    fig, axs = plt.subplots(info_dict['number_binary_nongenetic']//3, 3, figsize=(20, 10))
    cont_i = 0
    for col in info_dict['binary_nongenetic_vars']:
        sns.countplot(x=col, hue=treatment, data=database_numeric, ax=axs[cont_i // 3, cont_i % 3])
        cont_i += 1
    fig.suptitle(treatment)
    plt.tight_layout()
    plt.show()

# genetic
for treatment in info_dict['treatments']:
    fig, axs = plt.subplots(info_dict['number_gens']//3, 3, figsize=(20, 60))
    cont_i = 0
    for col in info_dict['genetic_vars']:
        sns.countplot(x=col, hue=treatment, data=database_numeric, ax=axs[cont_i // 3, cont_i % 3])
        cont_i += 1
    fig.suptitle(treatment)
    plt.tight_layout()
    plt.show()


# focus con 'ECOG' covariate
# ECOG is a covariate that is used to measure the general well-being of a patient and is used to determine the
# treatment to be applied. It is a categorical variable with 6 possible values: 0, 1, 2, 3, 4, 5.

# check the distribution of ECOG
sns.countplot(x='ECOG', data=database_numeric)
plt.show()

# check the distribution of ECOG for each treatment
for treatment in info_dict['treatments']:
    sns.countplot(x='ECOG', hue=treatment, data=database_numeric)
    plt.show()
















