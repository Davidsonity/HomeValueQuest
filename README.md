# House Price Prediction

![img](https://storage.googleapis.com/kaggle-competitions/kaggle/5407/media/housesbanner.png)

This project aims to predict the sales prices of residential homes in Ames, Iowa. The dataset contains 79 explanatory variables describing various aspects of the houses. By leveraging advanced regression techniques like random forest and gradient boosting, we can accurately estimate the final price of each home.

### Competition Description

Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this Kaggle competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.

With the dataset compiled by Dean De Cock, we have an incredible opportunity to showcase our skills in creative feature engineering and advanced regression techniques. The goal is to predict the sales price for each house in the test set.

### Dataset

The dataset used in this project is the Ames Housing dataset, which provides comprehensive information about residential homes in Ames, Iowa. It serves as a modernized and expanded version of the widely known Boston Housing dataset.

For detailed information about the dataset and its columns, please refer to the [data_description.txt](data_description.txt) file.

### Repository Structure

The repository is organized as follows:

- **README.md**: This file you're currently reading, providing an overview of the project.
- **data_description.txt**: A text file containing a detailed description of the dataset.
- **notebook.ipynb**: Jupyter Notebook containing the code, analysis, and implementation of the project.

#### Notebook Structure

The notebook can be divided into the following sections:

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

1. Clone the repository to your local machine.
2. Ensure you have the necessary requirements installed. 
3. Open the `notebook.ipynb` file in Jupyter Notebook or any compatible environment.
4. Execute the code cells in the notebook sequentially to reproduce the analysis and predictions.
5. Feel free to explore and modify the code according to your needs.

### Evaluation Metric

The submissions in this competition are evaluated based on the Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price. Taking logarithms helps to equalize the impact of errors on expensive and cheap houses.

### Resources

- [Competition Page on Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- [Kaggle Notebook Example](https://www.kaggle.com/code/vokeeshemitan/house-price)

By following the instructions and exploring the provided resources, you will gain a deep understanding of the project and be able to make accurate predictions on house prices.

Feel free to contribute to the project, provide feedback, or reach out with any questions or suggestions. Happy coding!

### Conclusion

This project demonstrates the application of advanced regression techniques to predict house prices. By exploring and preprocessing the dataset, building and evaluating regression models, and submitting predictions, participants can gain experience in feature engineering, regression modeling, and working with real-world datasets.

---

### Author

This project was developed by Emuejevoke Eshemitan. If you have any questions or feedback, feel free to contact me at eshemitanvoke@gmail.com
