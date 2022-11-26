IN PROGRESS 

<h1 align="center">Customer loyalty program for E-commerce</h1>

<p align="center">A clustering project</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/66283452/204003651-a72b8c68-b360-40cb-8a03-eb21c2dcd3fa.jpeg" alt="drawing" width="850"/>
</p>

*Obs: The company and business problem are both fictitious, although the data is real.*

*The in-depth Python code explanation is available in [this](https://github.com/brunodifranco/project-insuricare-ranking/blob/main/insuricare.ipynb) Jupyter Notebook.*

# 1. **Outleto and Business Problem**
<p align="justify"> Outleto is a multibrand outlet company, meaning it sells second line products of various companies at lower prices, through an E-commerce platform. Outleto's Marketing Team noticed that some customers tend to buy more expensive products, in higher quantities and more frequently than others, therefore contributing to a high share of Outleto's total gross revenue. Because of that, Outleto's Marketing Team wishes to launch a customer loyalty program, dividing the 5,702 customers in clusters, on which the best customers will be placed in a cluster named Insiders. 

To achieve this goal, the Data Science Team was requested to provide a business report regarding the clusters, containing a list of customers that will participate in Insiders, as well as answers to the following questions: 
  
##### 1) **How many customers will be a part of Insiders?**
##### 2) **How many clusters were created?**
##### 3) **How are the customers distributed amongst the clusters?**
##### 4) **What are these customers' main features?**
##### 5) **What's the gross revenue percentage coming from Insiders? and what about other clusters?**
##### 6) **How many items were purchased by each cluster?**
   
##### 7) **What are the requirements for a customer to be a part of Insiders? and for a customer to be removed?**
     
With that report the Marketing Team will promote actions to each cluster, in order to increase revenue, but of course focusing mostly in the Insiders cluster. </p>

# 2. **Data Overview**
The data was collected from Kaggle in the csv format. The initial features descriptions are available below:

| **Feature**          | **Definition** |
|:--------------------:|----------------|
|       InvoiceNo      | A 6-digit integral number uniquely assigned to each transaction |
|       StockCode      | Product (item) code | 
|       Description    | Product (item) name |
|       Quantity       | The quantities of each product (item) per transaction |
|       InvoiceDate    | The day when each transaction was generated |
|       UnitPrice      | Unit price (product price per unit) |
|       CustomerID     | Customer number (unique id assigned to each customer) |
|       Country        | The name of the country where each customer residers |

# 3. **Business Assumptions**

- All observations on which unit_price <= 0 were removed, as we're assuming those are gifts when unit_price = 0, and when unit_price < 0 it's described as "Adjust bad debt".  
- Some stock_code identifications weren't actual products, therefore they were removed.  
- Both description and country columns were removed, since those aren't relevant when modelling.  
- <p align="justify">Customer number 16446 was removed because he(she) bought 80995 items and returned them in the same day, leading to extraordinary values in other features. Other 12 customers were removed because they returned all items bought. In addition to that, three other users were also removed because they were considered to be data inconsistencies, since they had their return values greater than quantity of items bought, which doesn't make sense. These 16 were named "bad users".<p>

# 4. **Solution Plan**
## 4.1. How was the problem solved?

<p align="justify"> To provide the clusters final report the following steps were performed: </p>

- <b> Understanding the Business Problem</b>: Understanding the main objective we are trying to achieve and plan the solution to it. 

- <b> Collecting Data</b>: Collecting data from Kaggle.

- <b> Data Cleaning</b>: Checking data types, treating Nan's, renaming columns, dealing with outliers and filtering data.

- <b> Feature Engineering</b>: Creating new features from the original ones, so that those could be used in the ML model. More information in <a href="https://github.com/brunodifranco/project-insuricare-ranking#6-machine-learning-models">Section 5</a>.</p>

- <p align="justify"> <b> Exploratory Data Analysis (EDA)</b>: Exploring the data in order to obtain business experience, look for data inconsistencies, useful business insights and find important features for the ML model. This was done by using the <a href="https://pypi.org/project/pandas-profiling/">Pandas Profiling</a> library. Two EDA profile reports are available <a href="https://pypi.org/project/pandas-profiling/"> here</a>, one still with the bad users and one without them. 

- <b> Data Preparation</b>: Applying <a href="https://www.atoti.io/articles/when-to-perform-a-feature-scaling/">Rescaling Techniques</a> in the data.

- <b> Feature Selection</b>: Selecting the best features to use in the Machine Learning algorithm.

- <b> Space Analysis and Dimensionality Reduction</b>: <a href="https://builtin.com/data-science/step-step-explanation-principal-component-analysis">PCA</a>, <a href="https://umap-learn.readthedocs.io/en/latest/">UMAP</a> and <a href="https://gdmarmerola.github.io/forest-embeddings/">Tree-Based Embedding</a> were used to get a better data separation. 

- <p align="justify"> <b> Machine Learning Modeling</b>: Selecting the number of clusters (K) and then training Clustering Algorithms. More information in <a href="https://github.com/brunodifranco/project-insuricare-ranking#6-machine-learning-models">Section 6</a>.</p>

- <b> Model Evaluation</b>: Evaluating the model by using Silhouette Score and Silhouette Visualization.

- <b>Cluster Exploratory Data Analysis</b>: Exploring the clusters to obtain business experience and to find useful business insights. In addition to that, this step also helped building the business report. The top business insights found are available at <a href="https://github.com/brunodifranco/project-insuricare-ranking#5-top-business-insights"> Section 5</a>. 

- <p align="justify"> <b>Final Report and Deployment </b>: Providing a business report regarding the clusters, containing a list of customers that will participate in Insiders, as well as answering those previous questions. This is the project's <b>Data Science Product</b> and it was deployed in a <a href="https://www.metabase.com/">Metabase</a> application, so that it could be acessed from anywhere. More information in <a href="https://github.com/brunodifranco/project-insuricare-ranking#7-business-and-financial-results"> Section 7</a>.</p>
  
## 4.2. Tools and techniques used:
- [Python 3.10.8](https://www.python.org/downloads/release/python-3108/), [Pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/), [Sklearn](https://scikit-learn.org/stable/), [SciPy](https://scipy.org/) and [Pandas Profiling](https://pypi.org/project/pandas-profiling/).
- [SQL](https://www.w3schools.com/sql/) and [PostgresSQL](https://www.postgresql.org/).
- [Jupyter Notebook](https://jupyter.org/) and [VSCode](https://code.visualstudio.com/).
- [Metabase](https://www.metabase.com/). 
- [Render Cloud](https://render.com/) and [AWS Elastic Beanstalk](https://aws.amazon.com/pt/elasticbeanstalk/).
- [Git](https://git-scm.com/) and [Github](https://github.com/).
- [Exploratory Data Analysis (EDA)](https://towardsdatascience.com/exploratory-data-analysis-8fc1cb20fd15). 
- [Techniques for Feature Selection](https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/).
- [Clustering Algorithms (K-Means,  Gaussian Mixture Models, Agglomerative Hierarchical Clustering and DBSCAN)](https://towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68).

# 5. **Feature Engineering**
In total, 10 new features were created by using the original ones: 

| **New Feature** | **Definition** |
|:--------------------:|----------------|
| Gross Revenue | Gross revenue for each customer, which is equal to quantity times unit price |
| Average Ticket | Average monetary value spent on each purchase |
| Recency Days | Period of time from current time to the last purchase | 
| Max Recency Days | Max time a customer's gone without making any purchases |
| Frequency | Average purchases made by each customer during their latest purchase period and first purchase period |
| Purchases Quantity | Amount of times a customers's made any purchase |
| Quantity of Products | Total of products purchased |
| Quantity of Items | Total quantity of items purchased |
| Returns | Amount of items returned |
| Purchased and Returned Difference | Natural log of difference between items purchased and items returned | 

# 6. **Machine Learning Models**

<p align="justify"> In order to get better data separation a few dimensionality reduction were be tested: PCA, UMAP and Tree-Based Embedding. Results were satisfactory with Tree-Based Embedding, which consists of:

- Setting gross_revenue as a response variable, so it becomes a supervised learning problem; the
- Training a Random Forest (RF) model to predict gross_revenue using all other features.
- Plotting the embedding based on RF's leaves.

In total four clustering algorithms were tested, for a number of cluster varing from 2 to 24:
- K-Means
- Gaussian Mixture Models (GMM) with Expectation–Maximization (EM)
- Agglomerative Hierarchical Clustering (HC)
- DBSCAN 

The models were evaluated by silhouette score, as well as clustering vizualization. DBSCAN parameters' were optimized with Bayesian Optimization, however because it provided a very high number of clusters it was withdrawn as a possible final model. Our maximum cluster number was set to 11, due to practical purposes for Outleto's Marketing Team, so they can come up with exclusive actions to each cluster. 

Results were similar for K-Means, GMM and HC for cluster 8 to 11.


Results for were ver

This was the most fundamental part of this project, since it's in ML modeling where we can provide an ordered list of these new customers, based on their propensity score of buying the new insurance. Seven models were trained using cross-validation: </p>

- KNN Classifier
- Logistic Regression
- Random Forest Classifier
- AdaBoost Classifier
- CatBoost Classifier
- XGBoost Classifier 
- Light GBM Classifier

The initial performance for all seven algorithms are displayed below:

<div align="center">

|         **Model**        | **Precision at K** | **Recall at K** |
|:------------------------:|:------------------:|:---------------:|
|    LGBM Classifier       | 0.2789 +/- 0.0003  |0.9329 +/- 0.001 |
|    AdaBoost Classifier   | 0.2783 +/- 0.0007	|0.9309 +/- 0.0023|
|      CatBoost Classifier | 0.2783 +/- 0.0005	|0.9311 +/- 0.0018| 
|   XGBoost Classifier     | 0.2771 +/- 0.0006  |0.9270 +/- 0.0022|
|    Logistic Regression   | 0.2748 +/- 0.0009  |0.9193 +/- 0.0031|
| Random Forest Classifier | 0.2719 +/- 0.0005  |0.9096 +/- 0.0016|
|      KNN Classifier      | 0.2392 +/- 0.0006  |0.8001 +/- 0.0019|
</div>

<i>K is either equal to 20,000 or 40,000, given our business problem. </i>

<p align="justify"> The <b>Light GBM Classifier</b> model will be chosen for hyperparameter tuning, since it's by far the fastest algorithm to train and tune, whilst being the one with best results without any tuning. </p>

LGBM speed in comparison to other ensemble algorithms trained in this dataset:
- 4.7 times faster than CatBoost 
- 7.1 times faster than XGBoost
- 30.6 times faster than AdaBoost
- 63.2 times faster than Random Forest

<p align="justify"> At first glance the models performances don't look so great, and that's due to the short amount of variables, on which many are categorical or binary, or simply those don't contain much information. 

However, <b>for this business problem</b> this isn't a major concern, since the goal here isn't finding the best possible prediction on whether a customer will buy the new insurance or not, but to <b>create a score that ranks clients in a ordered list, so that the sales team can contact them in order to sell the new vehicle insurance</b>.</p>

After tuning LGBM's hyperparameters using [Bayesian Optimization with Optuna](https://optuna.readthedocs.io/en/stable/index.html) the model performance has improved on the Precision at K, and decreased on Recall at K, which was expected: 

<div align="center">

|         **Model**        | **Precision at K** | **Recall at K** |
|:------------------------:|:------------------:|:---------------:|
|      LGBM Classifier     | 0.2793 +/- 0.0005  |0.9344 +/- 0.0017| 

</div>

## <i>Metrics Definition and Interpretation</i>

<p align="justify"> <i> As we're ranking customers in a list, there's no need to look into the more traditional classification metrics, such as accuracy, precision, recall, f1-score, aoc-roc curve, confusion matrix, etc.

Instead, **ranking metrics** will be used:

- **Precision at K** : Shows the fraction of correct predictions made until K out of all predictions. 
  
- **Recall at K** : Shows the fraction of correct predictions made until K out of all true examples. 

In addition, two curves can be plotted: 

- <b>Cumulative Gains Curve</b>, indicating the percentage of customers, ordered by probability score, containing a percentage of all customers interested in the new insurance. 

- <b>Lift Curve</b>, which indicates how many times the ML model is better than the baseline model (original model used by Insuricare). </i> </p>

# 7. **Business and Financial Results**

## 7.1. Business Results

**1) By making 20,000 calls how many interested customers can Insuricare reach with the new model?**
<p align="center">
  <img src="https://user-images.githubusercontent.com/66283452/198152035-48c27ead-53f8-440e-af92-f049456dac33.png" alt="drawing" width="1000"/>
</p>

<p align="justify"> 

- 20,000 calls represents 26.24% of our database. So if the sales team were to make all these calls Insuricare would be able to contact 71.29% of customers interested in the new vehicle insurance, since 0.7129 is our recall at 20,000. </p>

- As seen from the Lift Curve, our **LGBM model is 2.72 times better than the baseline model at 20,000 calls.** 

**2) Now increasing the amount of calls to 40,000 how many interested customers can Insuricare reach with the new model?**

<p align="center">
  <img src="https://user-images.githubusercontent.com/66283452/198152040-929e3f17-d07e-401a-892c-50bf9c01f475.png" alt="drawing" width="1000"/>
</p>

- 40,000 calls represents 52.48% of our database. So if the sales team were to make all these calls Insuricare would be able to contact 99.48% of customers interested in the new vehicle insurance, since 0.9948 is our recall at 40,000.

- At 40,000 calls, our **LGBM model is around 1.89 times better than the baseline model.**  

## 7.2. Expected Financial Results

To explore the expected financial results of our model, let's consider a few assumptions:

- The customer database that will be reached out is composed of 76,222 clients.
- We expect 12.28% of these customers to be interested in the new vehicle insurance, since it's the percentage of interest people that participated in the Insuricare research. 
- The annual premium for each of these new vehicle insurance customers will be US$ 2,630 yearly. *

*<i> The annual premium of US$ 2,630 is set for realistic purposes, since it's the lowest and most common value in the dataset. </i>

The expected financial results and comparisons are shown below:

<div align="center">

|    **Model**    |  **Annual Revenue - 20,000 calls** | **Annual Revenue - 40,000 calls** |  **Interested clients reached out - 20,000 calls** | **Interested clients reached out - 40,000 calls** |
|:---------------:|:---:|:-----------------------------------:|:---:|:---------------------------------------:|
|       LGBM      | US$ 17,515,800.00    |US$ 24,440,590.00          | 6660   |9293                  |
|     Baseline    |  US$ 6,446,130.00    |US$ 12,894,890.00           | 2451  |4903                  |
| $\Delta$ (LGBM, Baseline) |  11,069,670.00     |US$ 11,545,700.00         |  4209   |   4390                  |

</div>

<i> $\Delta$ (LGBM, Baseline) is the difference between models. </i>

As seen above the LGBM model can provide much better results in comparison to the baseline model, with an annual financial result around 172% better for 20,000 calls and 89% better for 40,000 calls, which is exactly what was shown in the Lift Curve. 

# 5. **Top Business Insights**

 - ### 1st - Customers from Insiders are responsible for 58.3% of total items purchased.
<p align="center">
  <img src="https://user-images.githubusercontent.com/66283452/204071720-ab1d3d7c-e603-48e7-a13e-1cc0e97a4436.png" alt="drawing" width="850"/>
</p>

--- 
- ### 2nd - Customers from Insiders are responsible for 53.5% of the of total gross revenue.
<p align="center">
  <img src="https://user-images.githubusercontent.com/66283452/204071721-5af58f13-3bcc-420c-87a8-7a5ececfe2a2.png" alt="drawing" width="850"/>
</p>

--- 

- ### 3rd - Customers from Insiders have a number of returns higher than other customers, on average.
<p align="center">
  <img src="https://user-images.githubusercontent.com/66283452/204071722-6f5827cc-a4f8-465e-81ae-ee10981caf62.png" alt="drawing" width="850"/>
  
<i>That's probably because those customers buy a really high amount of items.</i>  </p>
  
---

# 8. **Propensity Score List and Model Deployment**

[![Metabase](https://img.shields.io/static/v1?style=for-the-badge&message=Metabase&color=509EE3&logo=Metabase&logoColor=FFFFFF&label=)](http://outletoapp-env.eba-ztruzhhu.us-east-1.elasticbeanstalk.com/public/dashboard/20d721e4-6c15-4538-896f-f3da4aff432b)

<p align="justify"> The full list sorted by propensity score is available for download <a href="https://github.com/brunodifranco/project-insuricare-ranking/blob/main/insuricare_list.xlsx">here</a>. However, for other new future customers it was necessary to deploy the model. In this project Google Sheets and Render Cloud were chosen for that matter. The idea behind this is to facilitate the predictions access for any new given data, as those can be checked from anywhere and from any electronic device, as long as internet connection is available. The spreadsheet will return you the sorted propensity score for each client in the requested dataset, all you have to do is click on the "Propensity Score" button, then on "Get Prediction".

<b> Click here to access the spreadsheet </b>[![Sheets](https://www.google.com/images/about/sheets-icon.svg)](https://docs.google.com/spreadsheets/d/1K2tJP6mVJwux4qret1Dde9gQ23KsDRGRl8eJbsigwic/edit?usp=sharing)

<i> Because the deployment was made in a free cloud (Render) it could take a few minutes for the spreadsheet to provide a response, <b> in the first request. </b> In the following requests it should respond instantly. </i>

</p>


# 9. **Conclusion**
In this project the main objective was accomplished:

 <p align="justify"> <b> We managed to provide a list of new customers ordered by their buy propensity score and a spreadsheet that returns the buy propensity score for other new future customers. Now, the Sales Team can focus their attention on the 20,000 or 40,000 first customers on the list, and in the future focus on the top K customers of the new list. </b> In addition to that, five interesting and useful insights were found through Exploratory Data Analysis (EDA), so that those can be properly used by Insuricare, as well as Expected Financial Results. </p>
 
# 10. **Next Steps**
<p align="justify"> Further on, this solution could be improved by a few strategies:
  
 - Conducting more market researches, so that more useful information on customers could be collected, since there was a lack of meaningful variables.
  
 - Applying <a href="https://builtin.com/data-science/step-step-explanation-principal-component-analysis">Principal Component Analysis (PCA) </a> in the dataset.
  
 - Try other classification algorithms that could better capture the phenomenon.

# Contact

- brunodifranco99@gmail.com
- [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/BrunoDiFrancoAlbuquerque/)

