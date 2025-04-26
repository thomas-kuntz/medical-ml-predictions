# Machine Learning for health predictions

## Context

This project was done as part of EPFL’s [CS-433 Machine Learning course](https://www.epfl.ch/labs/mlo/machine-learning-cs-433/) in fall 2024.

The project description and instructions can be found [here](https://raw.githubusercontent.com/epfml/ML_course/main/projects/project1/project1_description.pdf). We were only allowed to use numpy and Python’s standard library for the whole pipeline, hence the manual implementation of all the data cleaning, ML models, selection and validation.

## MICVD Prediction

This project aims to use to data from US health surveys to predict the risk of getting coronary heart disease (CHD) or myocardial infarction (MI). We use [data from the Behavioral Risk Factor Surveillance System (BRFSS)](https://www.cdc.gov/brfss/annual_data/annual_2015.html), a system of health-related telephone surveys that collects state data about U.S. residents regarding their health-related risk behaviors, chronic health conditions, and use of preventive services.

## How to use

You will need `numpy`, `jupyter` and `matplotlib` to run the file, and besides Python’s standard library, there should be nothing else required.

The dataset files should be placed in a folder called `dataset` next to the `run.ipynb` file. Run the `run.ipynb` file to reproduce our results. This includes data cleaning, feature transformation, feature selection, model training, hyperparameter selection, cross-validation, and generation of the predictions on the test set. Some cells will take several minutes to run, this is normal.

## Authors

- [Léo Paoletti](https://github.com/PaolettiLeo)
- [Thomas Kuntz](https://thomaskuntz.org)
- [Gabriel Alejandro Jiménez Calles](https://github.com/alejandrocalles)
