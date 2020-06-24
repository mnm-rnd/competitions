# AirQo Ugandan Air Quality Forecast Challenge

For ease of understanding of our approach, we have bundled our code in a package we made called `mlod`. This package uses publicly available packages (which are indicated in the `requirements.txt` file) and houses logic that tries to follow best practices when writting code.

Most of the functionality implemented inside our package, together with the contents of the `FinalNotebook.ipynb` notebook, are set up to ensure that the flow of the code is well understood and explainable.

Details on our approach to the solution can be found in the `Airqo-Challenge-MLOD-Approach.docx` file.

## Getting started

First, you'll be required to make sure that the folder structure is set as

```
--airqo-challenge-MLOD/
    - mlod.zip
    - requirements.txt
    - README.md (this file)
    - FinalNotebook.ipynb
```

### If you are running on your local machine,

... and if the file structure is set in the stated manner, the rest of the instructions together with scripts used to initiate the project are included within the first couple of cells of the `FinalNotebook.ipynb` file. 

Simply run the `jupyter lab` or `jupyter notebook` within this directory space from the terminal / anaconda prompt and you can get started. Be sure to have installed `anaconda` before hand

### If you are running on Colab,

... upload the FinalNotebook.ipynb into Colab then connect to a runtime. There after, upload the `mlod.zip` file and the `requirements.txt` to the colab storage (usually in the left side of the browser window)

-------------

After putting the files in their places, navigate to the first couple of cells under **"Run these cells only once during a session"** section and run the cells.

At this point, you have already set up the requirements needed to move along with the model preprocessing and training.

## You are set
Now that you have set up the needed code and packages, you'll have to save the original `Train.csv` and `Test.csv` inside a newly created `data` folder, otherwise, the code wont be able to process the files.

There after, you can begin running the other cells within the notebook one at a time, feel free to read the code within our package while you are at it. They include documentation for the most part

------------

**Information that follow from here are to provide further understanding on the parts of `mlod`**

------------

## Understanding `mlod`

The information to follow tries to explain the different parts of the `mlod`

The `mlod` was built with certain things in mind:
1. Building a **pipeline (work flow)** that can be **easily understood**.
3. **Modular code** that can allow use to change things **easily**, with **minimal risk**
3. Having a way to **track our code** and achieve **reproducible** results

That said, the main components that were created for the sake of this competition are the `PreProcessor`s, configuration code (`Config`) and `Model`s

### `PreProcessor`

These are classes that are built to perform preprocessing the data. This should convert the data from the original form, to the form that is needed just before feeding it to the model.

For our ensemble model, we preprocessed our original data in 2 different ways, for the 2 different models. In the code, these 2 ways are reflected in the:

 - `MlodPreProcessor` - which was originally devised way to preprocess the data
 - `AirQoPreProcessor` - which is another way that we modified from the borrowed approach in the Zindi Discussion. See: [Zindi Discussion #1116](https://zindi.africa/competitions/airqo-ugandan-air-quality-forecast-challenge/discussions/1116) of the competition.

### `Config` and other constants

This contains the different configuration that we'd use through out our code. This would include things like the model parameters, seed number, features to perform selective feature engineering among others.

### `Model`

These are the classes that contain the implementation of ML models
that we used in achieving our score. These models are wrapped in the classes:
- `LGBModel` - which contains the implementation of the [lightGBM](https://lightgbm.readthedocs.io/en/latest/) that was used as our base model
- `CatBoostModel` - which contains the implementation of the [CatBoost](https://catboost.ai/) model that is stacked on top of the base model

These classes are strategically called in the notebook file.
Details about these classes are explained in our `mlod` package