# eXplainable-AI

This repository contains all the code built for my thesis on Explainable AI.

The dataset is not included in the repository, since it is property of the Mobilytics project, but is intended to be released publicly soon after the publication of this thesis.

Inside the repository, the scripts section contains three folders:

* dataset_process, dedicated to preprocessing the dataset in data/raw into the feature sliding windows. The notebooks in this folder must be executed in order.
* ML_models, which contains all the tested models and their performance metrics, alongside experiments used in different interactions of our dataset.
* XAI, where explainability methods applied to our best performing model.

The src folder contains helper classes that are necessary to generate the dataset, developed by Gerard Caravaca.
