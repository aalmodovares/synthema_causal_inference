 # Causal inference in Synthema project

 The idea of this repository is to establish a common procedure to validate causal inference methods in real databases.

 The main problem for validating causal inference methods is the ausence of counterfactual outcomes (and that is pecisely the objective of a great part of causal inference methods).

 It is common to test the algorithms for causal inference with synthetic or semi-synthetic databases, in which the potential outcomes are generated from features and the real value of treatment effects is known.

 With a real database, we only observe one potential outcome, and validate the performance of CI methods becomes a challenge.

 The proposed method to validate causal inference in real data have three steps:

 1. Fit parametric models to predict the outcomes for each treatment group
 2. Generate synthetic outcomes (factual and counterfactuals) from that parametric models.
 3. Train and validate causal inference methods with semi-synthetic data.

 ## 1. Fit a parametric model for all potential outcomes separately

 That is, if we have a binary treatment, fit a parametric model for predicting outcomes from features in control patients (non treated), and fit another parametric model for the control group. 
Note that if we select different parametric models, the surface of the treatment effects will be more complex.

````
control_outcome = alpha*features + C # fitted on the control group
treated_outcome = e^(beta*features) + D # fitted on the treatment group
````

In the notebooks of this repository, a linear regression is fitted for control patients and an exponential regression is fitted for treated patients.

We allow this models not to fit perfectly the data: we are not evaluating the performance of this models. They only are useful to generate synthetic outcomes. But, instead of creating them from scratch arbitrarily, we try to approximate the real outcomes.

 ## 2. Generate synthetic outcomes from the data

 Once the parametric models are fitted, we generate all the potential outcomes for all the patients.

````
 mu0 = alpha**features + C # for causal inference this will be the ground truth for control outcome
 mu1 = e^(beta*features) * D # for causal inference this will be the ground truth for treatment outcome
````

We can compute real ITEs (individual treatment effects) for all patients. ITE = y1-y0
And ATE (Average treatment effect): ATE = E[y1-y0]

 ## 3. Train and evaluate causal inference methods

 With this information, we can implement a causal inference model (that only have access to factual outcome) for predicting treatment effects.

 The performance of the model in ITE predictions can be measured use PEHE (precission in heterogeneous treatment effects): PEHE = sqrt(E[(ite_real - ite_pred)^2])

 # EXAMPLE NOTEBOOK

 In the notebooks: 'causal_inference_notebook_ihdp' and 'causal_inference_notebook_aml', you can find examples of this whole process. A well known collection of methods for causal inference has been implemented to illustrate how they work and how to evaluate the performance.

 In addition, you can find other two notebooks. 'eda_notebook' includes a brief exploratory data analysis of AML database to evaluate data characteristics and potential lacks of informations. We can find a brief study of positivity assumption, checking if there is overlapping of the features between the treated and control group.
 
 On the other hand 'predictors_notebook' has been used to evaluate if the features are good predictors of the treatments (considering the treatments are HSCT variations). This is important for the most of the algorithms of causal inference, specially for those that use propensity scores.
