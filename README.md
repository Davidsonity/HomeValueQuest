## House Prices: Advanced Regression Techniques

![img](https://storage.googleapis.com/kaggle-competitions/kaggle/5407/media/housesbanner.png)
> View kaggle Notebook @ https://www.kaggle.com/code/vokeeshemitan/house-price

This project is part of the Kaggle competition titled "House Prices: Advanced Regression Techniques." The goal of this competition is to predict the final sale price of residential homes in Ames, Iowa, based on a set of 79 explanatory variables that describe various aspects of the houses.

### Dataset

The dataset provided for this competition contains both training and test data. The training dataset consists of labeled examples with known sale prices, while the test dataset contains unlabeled examples for which participants need to predict the sale prices. The dataset provides a comprehensive set of features, including information about the size, location, amenities, and quality of each house.

### Evaluation Metric

Submissions in this competition are evaluated based on the Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted sale prices and the logarithm of the observed sale prices. By taking the logarithm, the competition aims to ensure that errors in predicting expensive houses and cheap houses have an equal impact.

### Project Structure

The project can be divided into the following sections:

1. **Preliminary Wrangling**: This section involves loading the dataset, exploring the data, and understanding its structure. It includes steps like checking data information, performing descriptive statistics, and identifying columns with missing values.

2. **Data Cleaning/Preprocessing**: In this section, the dataset is preprocessed to handle missing values and convert categorical variables into numerical representations suitable for machine learning models. It includes tasks like filling NaN values, converting non-numeric data into numeric formats (label encoding and one-hot encoding), and scaling/normalizing features if necessary.

3. **Models**: This section focuses on building and evaluating various regression models using the preprocessed dataset. The models considered in this project include Linear Regression, Lasso Regression, Ridge Regression, ElasticNet Regression, LightGBM Regression, Gradient Boosting Regression, CatBoost Regression, Bayesian Ridge Regression, Random Forest Regression, and Extra Trees Regressor. Each model is trained on the training dataset and evaluated using the validation dataset.

4. **Results/Findings**: This section compares the performance of the different models based on their RMSE scores. It provides insights into the effectiveness of each model in predicting house prices.

5. **Submission**: Finally, the best-performing model is used to predict the sale prices for the test dataset. The predictions are then submitted in the required format for evaluation on the competition platform.

### Requirements

To run this project, the following packages and libraries are required:
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- LightGBM
- CatBoost
- BayesianRidge
- RandomForestRegressor
- ExtraTreesRegressor

Please make sure to install these dependencies before running the project.

### Usage

To use this project, follow these steps:
1. Download the dataset from the Kaggle competition page.
2. Ensure that the required dependencies are installed.
3. Run the code in the provided Jupyter Notebook or Python environment.
4. Follow the instructions within the notebook to explore, preprocess, and model the data.
5. Use the best-performing model to make predictions on the test dataset and submit the results to the competition platform.

### Conclusion

This project demonstrates the application of advanced regression techniques to predict house prices. By exploring and preprocessing the dataset, building and evaluating regression models, and submitting predictions, participants can gain experience in feature engineering, regression modeling, and working with real-world datasets.

For more detailed information about the competition and dataset, please refer to the [Kaggle competition page](https://www.kaggle.com/competitions/house-prices

-advanced-regression-techniques).

### Author

This project was developed by Emuejevoke Eshemitan. If you have any questions or feedback, feel free to contact me at eshemitanvoke@gmail.com
