#script for exploratory data analysis (EDA)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

def load_data():


    database = pd.read_excel('AML_Synthema.xlsx', sheet_name='Synthema')
    info_cols = database.columns
    cols = database.iloc[0, :]
    database = pd.DataFrame(database.values[1:, :], columns=cols)

    # create a numeric database, removing string columns
    database_numeric = database.copy()

    categorical_sure = []
    #first, remove strings
    for info_col, col in zip(info_cols, cols):
        if 'string' in info_col or 'patient id' in info_col or 'formula' in info_col:
            database_numeric = database_numeric.drop(col, axis=1)
        elif 'categorical' in info_col:
            categorical_sure.append(col)



    # labeled variables (ordered)
    labeled_vars = ['ELN 2017', 'ELN 2022'] #, 'RESPONSE TO INDUCTION', 'STATUS AT HSCT', 'EFS EVENT']

    # categorical variables (unordered)
    categorical_doubtful_vars = ['STATUS AT HSCT', 'RESPONSE TO INDUCTION'] # same info:
                                                                            # "CR=complete response
                                                                            # PR=partial response
                                                                            # RD=refractory disease
                                                                            # Rez=Relapse
    categorical_doubtful_vars.append('EFS EVENT') # "RD=refractory disease
                                                  # Rez=relapse"

    # binary variables
    #are those that have '1=' in the name
    binary_vars = []
    for info_col, col in zip(info_cols, cols):
        if '1=' in info_col:
            binary_vars.append(col)

    binary_nongenetic_vars = ['FAMILY DONOR', 'STATUS_OS', 'STATUS_EFS', 'RFS STATUS', 'SEX', 'SPLENOMEGALY']
    treatments = ['Allogeneic HSCT', 'Autologous HSCT']
    # 'Intensive chemotherapy induction' column only has 1s, so it is useless


    # convert labeled and categorical variables to numeric using LabelEncoder
    for labeled_var in labeled_vars:
        database_numeric.loc[:, labeled_var] = LabelEncoder().fit_transform(database_numeric.loc[:, labeled_var].values)

    for categorical_doubtful_var in categorical_doubtful_vars:
        database_numeric.loc[:, categorical_doubtful_var] = LabelEncoder().fit_transform(database_numeric.loc[:, categorical_doubtful_var].values)

    # AT THIS POINT, ALL VARIABLES SHOULD BE NUMERIC, transform dataframe to_numeric
    #column by column
    for col in database_numeric.columns:
        database_numeric.loc[:, col] = pd.to_numeric(database_numeric.loc[:, col].values, errors='coerce')

    continuous_vars = [col for col in database_numeric.columns if col not in binary_vars + labeled_vars + categorical_doubtful_vars +
                       binary_nongenetic_vars + treatments + categorical_sure]

    # REMOVE variables that are the same for all individuals
    # remove variables that have only one value
    for col in database_numeric.columns:
        if len(database_numeric.loc[:, col].unique()) == 1:
            database_numeric = database_numeric.drop(col, axis=1)

            if col in binary_vars:
                binary_vars.remove(col)
            elif col in labeled_vars:
                labeled_vars.remove(col)
            elif col in categorical_doubtful_vars:
                categorical_doubtful_vars.remove(col)
            elif col in binary_nongenetic_vars:
                binary_nongenetic_vars.remove(col)
            elif col in treatments:
                treatments.remove(col)

    genetic_vars = [var for var in binary_vars if (var not in binary_nongenetic_vars and var not in treatments)]#binary vars - binary nongenetic vars

    #now genetic
    number_gens = len(genetic_vars)
    number_cont = len(continuous_vars)

    # Bar plots of categorical variables
    all_cat = categorical_doubtful_vars + labeled_vars + categorical_sure
    number_cat = len(all_cat)
    features = all_cat + binary_nongenetic_vars + genetic_vars + continuous_vars

    info_dict = {'number_gens': number_gens, 'number_cont': number_cont, 'number_binary_nongenetic': len(binary_nongenetic_vars),
                 'number_binary': len(binary_vars), 'number_labeled': len(labeled_vars),
                 'number_categorical_doubtful': len(categorical_doubtful_vars), 'number_treatments': len(treatments),
                 'number_categorical_sure': len(categorical_sure),
                 'number_patients': len(database_numeric),
                 'number_variables': len(database_numeric.columns),
                 'labeled_vars': labeled_vars, 'categorical_doubtful_vars': categorical_doubtful_vars,
                 'binary_vars': binary_vars, 'binary_nongenetic_vars': binary_nongenetic_vars,
                 'treatments': treatments, 'categorical_sure': categorical_sure,
                 'genetic_vars': genetic_vars, 'continuous_vars': continuous_vars,
                 'features': features, 'all_cat': all_cat, 'number_cat': number_cat}

    #database numeric to float64
    for col in database_numeric.columns:
        database_numeric.loc[:, col] = database_numeric.loc[:, col].astype(np.float64)



    return database_numeric, info_dict
















