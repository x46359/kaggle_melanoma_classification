### melanoma_classification
==============================



#### Project Overview
- Use melanoma image data to classify benign and malignant test cases
- More details https://www.kaggle.com/c/siim-isic-melanoma-classification
- Leveraged pretrained pytorch models (e.g. resnet18, alexnet, vgg16, etc.) + LGBM as final ensemble model



Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data (predictions) used in downstream LightGBM ensemble model.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained models by image size
    │
    ├── notebooks          <- Jupyter notebooks, initial data exploration
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    |
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    |   |
    |   ├── main.py        <- main python module to train models, create predictions, and run LGBM ensemble
    │   │
    |   ├── paths.py       <- path setup file
    |   |
    |   ├── utility.py     <- utility functions
    |   |
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_data.py            <- creates pytorch dataloaders
    │   │
    │   ├── models         <- Scripts to train models, create predictions, create ensemble model
    │   │   ├── train_models.py         <- train models
    │   │   ├── create_predictions.py   <- create predictions for model, image size
    │   │   └── run_lgb.py              <- run final ensemble LGBM model
    │
    └── setup.py           <- makes project pip installable (pip install -e .) so src can be imported


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
