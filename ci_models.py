# SCRIPTS OF WELL-KNOWN CAUSAL INFERENCE MODELS
import numpy as np
import sklearn

from sklearn.metrics import  mean_squared_error
import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Concatenate
from tensorflow.keras import regularizers
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN

#TARNET
#CODE FROM
# Koch, B., Sainburg, T., Geraldo, P., JIANG, S., Sun, Y., & Foster, J. G. (2021, October 10). Deep Learning for Causal Inference. https://doi.org/10.31235/osf.io/aeszf

def make_tarnet(input_dim, reg_l2=0.01, nl_phi=3, nl_y=2, nn_phi=200, nn_y=100):
    '''
    The first argument is the column dimension of our data.
    It needs to be specified because the functional API creates a static computational graph
    The second argument is the strength of regularization we'll apply to the output layers
    '''
    x = Input(shape=(input_dim,), name='input')

    # REPRESENTATION
    # in TF2/Keras it is idiomatic to instantiate a layer and pass its inputs on the same line unless the layer will be reused
    # Note that we apply no regularization to the representation layers
    phi = Dense(units=nn_phi, activation='elu', kernel_initializer='RandomNormal', name='phi_1')(x)
    for i_l in range(nl_phi - 1):
        phi = Dense(units=nn_phi, activation='elu', kernel_initializer='RandomNormal', name='phi_' + str(i_l + 2))(phi)


    # HYPOTHESIS

    y0_hidden = Dense(units=nn_y, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name='y0_hidden_1')(phi)
    y1_hidden = Dense(units=nn_y, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name='y1_hidden_1')(phi)

    for i_l in range(nl_y - 1):
        y0_hidden = Dense(units=nn_y, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name='y0_hidden_' + str(i_l + 2))(y0_hidden)
        y1_hidden = Dense(units=nn_y, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name='y1_hidden_' + str(i_l + 2))(y1_hidden)

    # third
    y0_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y0_predictions')(y0_hidden)
    y1_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y1_predictions')(y1_hidden)

    # a convenience "layer" that concatenates arrays as columns in a matrix
    # this time we'll return Phi as well to calculate cate_nn_err
    concat_pred = Concatenate(1)([y0_predictions, y1_predictions, phi])
    # the declarations above have specified the computational graph of our network, now we instantiate it
    model = Model(inputs=x, outputs=concat_pred)

    return model

# every loss function in TF2 takes 2 arguments, a vector of true values and a vector predictions
def regression_loss(concat_true, concat_pred):
    # computes a standard MSE loss for TARNet
    y_true = concat_true[:, 0]  # get individual vectors
    t_true = concat_true[:, 1]

    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]

    # Each head outputs a prediction for both potential outcomes
    # We use t_true as a switch to only calculate the factual loss
    loss0 = tf.reduce_sum((1. - t_true) * tf.square(y_true - y0_pred))
    loss1 = tf.reduce_sum(t_true * tf.square(y_true - y1_pred))
    # note Shi uses tf.reduce_sum for her losses even though mathematically we should be using the mean
    # tf.reduce_mean and tf.reduce_sum should be equivalent, but maybe having larger error gradients makes training easier?
    return loss0 + loss1

def predict_causal_effects(models, data, learner = 'slearner', config = None):
    # t learner trains a model for each treatment value
    # Define machine learning models for outcome prediction
    # Initialize dictionaries to store causal effects and predicted outcomes
    predicted_y = {}

    x_train, t_train, y_train= data['data_train']
    x_test, t_test, y_test = data['data_test']

    if learner=='tarnet':
        sgd_callbacks = [
            TerminateOnNaN(),
            EarlyStopping(monitor='val_loss', patience=config['tarnet']['patience'], min_delta=0),
            # 40 is Shi's recommendation patience for this dataset, but you should tune for your data
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=0, mode='auto',
                              min_delta=0, cooldown=0, min_lr=0)]

        model = make_tarnet(x_train.shape[1], reg_l2=0.01,
                            nl_phi=config['tarnet']['NL_phi'], nl_y=config['tarnet']['NL_y'],
                            nn_phi=config['tarnet']['NN_phi'], nn_y=config['tarnet']['NN_y'])
        model.compile(optimizer='adam', loss=regression_loss)
        model.fit(x=x_train, y=np.column_stack((y_train, t_train)),
                  callbacks=sgd_callbacks, validation_split=0.1,
                  epochs=config['tarnet']['NE'], batch_size=config['tarnet']['batch_size'], verbose=0)

        prediction_train = model.predict(x_train)
        prediction_test = model.predict(x_test)
        y0_train, y1_train =prediction_train[:,0], prediction_train[:,1]
        y0_test, y1_test = prediction_test[:,0], prediction_test[:,1]

        y_pred_train = np.zeros_like(y_train)
        y_pred_train[t_train == 0] = y0_train[t_train == 0]
        y_pred_train[t_train == 1] = y1_train[t_train == 1]

        y_pred_test = np.zeros_like(y_test)
        y_pred_test[t_test == 0] = y0_test[t_test == 0]
        y_pred_test[t_test == 1] = y1_test[t_test == 1]

        # Calculate predicted outcomes for t=0, t=1, and observed t values
        predicted_y['TARNET'] = {
            "y0_train": y0_train,
            "y1_train": y1_train,
            "y_train": y_pred_train,
            "y0_test": y0_test,
            "y1_test": y1_test,
            "y_test": y_pred_test,
        }

        return predicted_y

    # Iterate over models
    for model_name, model in models.items():

        if learner == 'slearner':
            # Fit the model to observed data
            model.fit(np.column_stack((x_train, t_train)), y_train)

            # Counterfactual prediction in train
            y0_train = model.predict(np.column_stack((x_train, np.zeros_like(t_train))))
            y1_train = model.predict(np.column_stack((x_train, np.ones_like(t_train))))

            # Counterfactual predictions for t=0 and t=1 in test
            y0_test = model.predict(np.column_stack((x_test, np.zeros_like(t_test))))
            y1_test = model.predict(np.column_stack((x_test, np.ones_like(t_test))))

            # Calculate causal effect as y1 - y0
            causal_effect_train = y1_train - y0_train
            causal_effect_test = y1_test - y0_test

            y_pred_train = model.predict(np.column_stack((x_train, t_train)))
            y_pred_test = model.predict(np.column_stack((x_test, t_test)))



        elif learner == 'tlearner':
            # create two models, one for each treatment value
            model_t0 = sklearn.base.clone(model)
            model_t1 = sklearn.base.clone(model)

            # Fit the model to observed data
            model_t0.fit(x_train[t_train == 0], y_train[t_train == 0])
            model_t1.fit(x_train[t_train == 1], y_train[t_train == 1])

            # Counterfactual prediction in train
            y0_train = model_t0.predict(x_train)
            y1_train = model_t1.predict(x_train)

            # Counterfactual predictions for t=0 and t=1 in test
            y0_test= model_t0.predict(x_test)
            y1_test= model_t1.predict(x_test)

            # Calculate causal effect as y1 - y0
            causal_effect_train = y1_train - y0_train
            causal_effect_test = y1_test - y0_test

            y_pred_train = np.zeros_like(y_train)
            y_pred_train[t_train == 0] = y0_train[t_train == 0]
            y_pred_train[t_train == 1] = y1_train[t_train == 1]

            y_pred_test = np.zeros_like(y_test)
            y_pred_test[t_test == 0] = y0_test[t_test == 0]
            y_pred_test[t_test == 1] = y1_test[t_test == 1]

        else:
            raise ValueError('learner must be slearner, tlearner or tarnet')

        # Calculate predicted outcomes for t=0, t=1, and observed t values
        predicted_y[model_name] = {
            "y0_train": y0_train,
            "y1_train": y1_train,
            "y_train": y_pred_train,
            "y0_test": y0_test,
            "y1_test": y1_test,
            "y_test": y_pred_test,
        }

        if model_name == 'Linear Regression' and learner=='slearner':
            predicted_y[model_name]['coefs'] = model.coef_

    return predicted_y

def evaluate_causal_inference(models, true_y, predicted_y):

    pehe = {model_name: {} for model_name in models}
    ate_error = {model_name: {} for model_name in models}

    for model_name in models:
        # Evaluate model performance on test data
        mse_train = mean_squared_error(true_y['y_train'], predicted_y[model_name]["y_train"])
        mse_test = mean_squared_error(true_y['y_test'], predicted_y[model_name]["y_test"])
        print(f"Model: {model_name}, MSE on train data: {mse_train:.2f}")
        print(f"Model: {model_name}, MSE on test data: {mse_test:.2f}")

        # Calculate PEHE for training and testing data
        true_effect_train = true_y['mu1_train'] - true_y['mu0_train']
        true_ate_train = np.mean(true_effect_train)
        predicted_effect_train = predicted_y[model_name]["y1_train"] - predicted_y[model_name]["y0_train"]
        predicted_ate_train = np.mean(predicted_effect_train)
        pehe[model_name]['train'] = np.sqrt(np.mean((predicted_effect_train- true_effect_train) ** 2))
        ate_error[model_name]['train'] = np.abs(predicted_ate_train - true_ate_train)

        true_effect_test = true_y['mu1_test'] - true_y['mu0_test']
        true_ate_test = np.mean(true_effect_test)
        predicted_effect_test = predicted_y[model_name]["y1_test"] - predicted_y[model_name]["y0_test"]
        predicted_ate_test = np.mean(predicted_effect_test)
        pehe[model_name]['test'] = np.sqrt(np.mean((predicted_effect_test - true_effect_test) ** 2))
        ate_error[model_name]['test'] = np.abs(predicted_ate_test - true_ate_test)
        # Display PEHE for training and testing data

        print(f"Model: {model_name}, PEHE (Train): {pehe[model_name]['train']:.2f}")
        print(f"Model: {model_name}, PEHE (Test): {pehe[model_name]['test']:.2f}")

    return pehe, ate_error

def plot_mean_pehes(Num_exps, learners, pehe_dict, divide_by='train/test', sem=None, axes=None, seed=1, i_exp=1):
    mean_pehes = {learner: {'train': {}, 'test': {}} for learner in learners}
    std_pehes = {learner:{'train': {}, 'test': {}} for learner in learners}
    # Plot the mean (across experiments) of each model for each setting
    for learner in learners:
        # compute the mean of pehe 'train' and 'test' for each model and each dataset
        for model_name in pehe_dict[learner][f'exp_{i_exp}'].keys():
            pehes_train = []
            pehes_test = []
            if Num_exps is None:
                pehes_train.append(pehe_dict[learner][f'exp_{i_exp}'][model_name]['train'])
                pehes_test.append(pehe_dict[learner][f'exp_{i_exp}'][model_name]['test'])
            else:
                for i_exp in range(1, Num_exps+1):
                    pehes_train.append(pehe_dict[learner][f'exp_{i_exp}'][model_name]['train'])
                    pehes_test.append(pehe_dict[learner][f'exp_{i_exp}'][model_name]['test'])

            mean_pehes[learner]['train'][model_name] = np.mean(pehes_train)
            mean_pehes[learner]['test'][model_name] = np.mean(pehes_test)
            std_pehes[learner]['train'][model_name] = np.std(pehes_train)
            std_pehes[learner]['test'][model_name] = np.std(pehes_test)

    # Plot the mean and the std (across experiments) of each model
    # Create a figure with 2x2 subplots
    if divide_by=='learner':
        fig, axs = plt.subplots(len(learners), 2, figsize=(12, 20))

        # Create a list of colors for the bars
        colors = ['b', 'g']  # Get model names from the first learner and setting

        for learner_idx, learner in enumerate(learners):
            for i_tr, tr in enumerate(['train', 'test']):
                model_names = list(mean_pehes[learner]['train'].keys())  # Get model names from the first learner and setting

                ax = axs[learner_idx, i_tr]

                x = np.arange(len(model_names))
                bar_width = 0.35

                # pos = -bar_width / 2 if setting == 'observed' else bar_width / 2

                # # Plot mean values
                # ax.bar(x + pos, [mean_pehes[learner][tr][model_name] for model_name in model_names], bar_width, label=setting,
                #        color=colors[setting_idx])
                # ax.bar(x + pos, [mean_pehes[learner]['test'][model_name] for model_name in model_names], bar_width, label=setting,
                #        color=colors[1])
                ax.bar(x, [mean_pehes[learner][tr][model_name] for model_name in model_names], bar_width, label=tr,
                          color=colors[i_tr])

                # Plot standard deviation as error bars
                # ax.errorbar(x + pos, [mean_pehes[learner][tr][model_name] for model_name in model_names],
                #             yerr=[std_pehes[learner][tr][model_name] for model_name in model_names],
                #             fmt='none', color='k', elinewidth=2, capsize=4, capthick=2)
                # ax.errorbar(x + bar_width / 2, [mean_pehes[learner]['test'][model_name] for model_name in model_names],
                #             yerr=[std_pehes[learner]['test'][model_name] for model_name in model_names],
                #             fmt='none', color='k', elinewidth=2, capsize=4, capthick=2)a
                ax.errorbar(x, [mean_pehes[learner][tr][model_name] for model_name in model_names],)

                ax.set_xticks(x)
                ax.set_xticklabels(model_names, rotation=45, ha="right")  # Rotate x-axis labels
                ax.set_xlabel('Model Name')
                ax.set_ylabel('Metric Value')
                ax.set_title(f'{learner} - {tr}')
                ax.grid(True)
                ax.legend()

    elif divide_by=='train/test':
        if axes is None:
            fig, axs = plt.subplots(2, 1, figsize=(12, 10))
        else:
            axs=axes
        # Create a list of colors for the bars
        colors = ['b', 'g']  # Get model names from the first learner and setting
        xticks = []
        combined_mean = {}
        combined_std = {}
        for learner in learners:
            for model_name in mean_pehes[learner]['train'].keys():
                xticks.append(f'{learner} - {model_name}')
                combined_mean[f'{learner} - {model_name}'] = {'train': mean_pehes[learner]['train'][model_name],
                                                                'test': mean_pehes[learner]['test'][model_name]}
                combined_std[f'{learner} - {model_name}'] = {'train': std_pehes[learner]['train'][model_name],
                                                                'test': std_pehes[learner]['test'][model_name]}


        for i_tr, tr in enumerate(['train', 'test']):
            model_names = list(mean_pehes[learner]['train'].keys())  # Get model names from the first learner and setting

            ax = axs[i_tr]

            x = np.arange(len(xticks))

            bar_width = 0.35

            # pos = -bar_width / 2 if setting == 'observed' else bar_width / 2

            # # Plot mean values
            # ax.bar(x + pos, [combined_mean[xtick][tr] for xtick in xticks], bar_width, label=setting,
            #        color=colors[setting_idx])
            # ax.bar(x + pos, [mean_pehes[learner]['test'][model_name] for model_name in model_names], bar_width, label=setting,
            #        color=colors[1])
            ax.bar(x, [combined_mean[xtick][tr] for xtick in xticks], bar_width, label=tr,
                        color=colors[i_tr])

            # # Plot standard deviation as error bars
            # ax.errorbar(x + pos, [combined_mean[xtick][tr] for xtick in xticks],
            #             yerr=[combined_std[xtick][tr] for xtick in xticks],
            #             fmt='none', color='k', elinewidth=2, capsize=4, capthick=2)
            # ax.errorbar(x + bar_width / 2, [mean_pehes[learner]['test'][model_name] for model_name in model_names],
            #             yerr=[std_pehes[learner]['test'][model_name] for model_name in model_names],
            #             fmt='none', color='k', elinewidth=2, capsize=4, capthick=2)
            ax.errorbar(x, [combined_mean[xtick][tr] for xtick in xticks],
                        yerr=[combined_std[xtick][tr] for xtick in xticks],
                        fmt='none', color='k', elinewidth=2, capsize=4, capthick=2)

            ax.set_xticks(x)
            ax.set_xticklabels(xticks, rotation=45, ha="right")  # Rotate x-axis labels
            ax.set_xlabel('Model Name')
            ax.set_ylabel('Metric Value')
            ax.set_title(f'{tr}')
            ax.grid(True)
            ax.legend()

    else:
        raise ValueError('divide_by must be learner or train/test')

    if axes is None:
        # Add a common title for the entire figure
        if sem is not None:
            fig.suptitle(f'Evaluation Metrics for Learners and Settings - {sem}', fontsize=16)
            # Adjust the layout to prevent overlapping labels

            if divide_by == 'train/test':
                plt.savefig(f'./pehe_train-test_{sem}.png')
        else:
            fig.suptitle('Evaluation Metrics for Learners and Settings', fontsize=16)

        plt.tight_layout()
        # else:
        #     plt.savefig(f'./pehe_{divide_by}.png')
        # # Show the figure

        plt.show()