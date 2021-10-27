# Machine Learning From Scratch
![image](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white) ![image](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252) 

This is my repository on learning Machine Learning from scratch, if you want to checkout my repository on Deep learning with TensorFlow click here 👉: [TensorFlow-Deep-Learning](https://github.com/BaoLocPham/Tensorflow-Deep-Learning.git)

## Table of contents
| Number     | Notebook | Description | Extras |
| ----------- | ----------- | ----------- | ----------- |
| 00 | [Basic ML Intuition](#basic-intuition) | What is ML, Bias and Variances? | |
| 01 | [Data Preprocess Template](https://github.com/BaoLocPham/MachineLearningFromScratch/blob/main/Part%201%20-%20Data%20Preprocessing/data_preprocessing_template.ipynb) | Data preprocess template | |
| 02 | [Regression](#regression) | Simple linear regression, multiple, poly, ... | |
| 03 | [Classification](#classification) | Logistic regression, knn, svm, ... | |
| 04 | [Clustering](#clustering) | KMeans clustering, Hierarchical clustering | |
| 05 | [Association Rule Learning](#association-rule-learning) | Apriori, Eclat | |
| 06 | [Reinforcement Learning](#reinforcement-learning) | UCB, Thomson Sampling | |
| 07 | [NLP](#nlp) | Introduction to nlp | |
| 08 | [Dimensionality Reduction](#dimensionality-reduction) | PCA, Kernel PCA, LDA | |
| ## | [Model Selection](#model-selection) | Model selection: regression, classifcation | |
| ## | [Case Study](#case-study) | Case study | |

## Details
### Basic Intuition
#### Math
* [StatQuest: The Normal Distribution clearly explained](https://youtu.be/rzFX5NWojp0)

#### Machine learning Fundamentals
* [Bias and Variance](https://medium.datadriveninvestor.com/bias-and-variance-in-machine-learning-51fdd38d1f86)
* [Confusion Matrix, Accuracy, Precision, Recall, F1-score](https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62)
* [L1 and L2 Regularization](https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c)
* [StatsQuest L1 Regularization](https://youtu.be/Q81RR3yKn30)
* [StatsQuest L2 Regularization](https://youtu.be/NGf0voTMlcs)
* [Regularization and Cross Validation](https://medium.com/analytics-vidhya/regularization-and-cross-validation-how-to-choose-the-penalty-value-lambda-1217fa4351e5)
***
### Regression
| Number     | Notebook | Extras |
| ----------- | ----------- | ----------- |
| 01 | [Simple Linear Regression](https://github.com/BaoLocPham/MachineLearningFromScratch/blob/main/Part%202%20-%20Regression/Section%204%20-%20Simple%20Linear%20Regression/simple_linear_regression.ipynb) |  | 
| 02 | [Multiple Linear Regression](https://github.com/BaoLocPham/MachineLearningFromScratch/blob/main/Part%202%20-%20Regression/Section%205%20-%20Multiple%20Linear%20Regression/multiple_linear_regression.ipynb) | [When to multiple linear regression](https://towardsdatascience.com/understanding-when-simple-and-multiple-linear-regression-give-different-results-7cf6c787766c) |
| 03 | [Polynomial Regression](https://github.com/BaoLocPham/MachineLearningFromScratch/blob/main/Part%202%20-%20Regression/Section%206%20-%20Polynomial%20Regression/polynomial_regression.ipynb) | [Polynomial Regression](https://towardsdatascience.com/polynomial-regression-bbe8b9d97491) |
| 04 | [Support Vector Regression](https://github.com/BaoLocPham/MachineLearningFromScratch/blob/main/Part%202%20-%20Regression/Section%207%20-%20Support%20Vector%20Regression%20(SVR)/support_vector_regression.ipynb) | [Introduction to SVR](https://towardsdatascience.com/an-introduction-to-support-vector-regression-svr-a3ebc1672c2), [`kernels`](https://data-flair.training/blogs/svm-kernel-functions/) |
| 05 | [Decision Tree Regression](https://github.com/BaoLocPham/MachineLearningFromScratch/blob/main/Part%202%20-%20Regression/Section%208%20-%20Decision%20Tree%20Regression/decision_tree_regression.ipynb) | [Decision Tree Regression](https://towardsdatascience.com/machine-learning-basics-decision-tree-regression-1d73ea003fda), [Decision Tree ML](https://machinelearningcoban.com/tabml_book/ch_model/decision_tree.html) |
| 06 | [Random Forest Regression](https://github.com/BaoLocPham/MachineLearningFromScratch/blob/main/Part%202%20-%20Regression/Section%209%20-%20Random%20Forest%20Regression/random_forest_regression.ipynb) | [Random Forest](https://towardsdatascience.com/machine-learning-basics-random-forest-regression-be3e1e3bb91a), [Random Forest ML](https://machinelearningcoban.com/tabml_book/ch_model/random_forest.html)|

#### Regression: Pros and cons
| Regression Model | Pros | Cons |
| ----------- | ----------- | ----------- |
| Linear Regression | Works on any size of the dataset, gives informations about relevance of features. | [The Linear Regression Assumptions](https://www.statisticssolutions.com/free-resources/directory-of-statistical-analyses/assumptions-of-linear-regression/). |
| Polynomial Regression | Works on any size of dataset, works very well on non linear problems. | Need to choose the right polynomial degree for a good bias, variance tradeoff. |
| SVR | Easily adaptable, works very well on non linear problems, not bias by outlier. | Compulsory to apply feature scaling, not well documentated, more difficult to understand. |
| Decision Tree Regression | Interpretablity, no need for feature scaling, works on both linear, nonlinear problems. | Poor Results on too small datasets, overfitting can easily occur. |
| Random Forest Regression | Powerful and accurate, good performance on may problems, including nonlinear. | Poor Results on too small datasets, overfitting can easily occur. |
***
### Classification
| Number     | Notebook | Extras |
| ----------- | ----------- | ----------- |
| 01 | [Logistic Regression](https://github.com/BaoLocPham/MachineLearningFromScratch/blob/main/Part%203%20-%20Classification/Section%2014%20-%20Logistic%20Regression/logistic_regression.ipynb) | [StatQuest: Logistic Regression](https://youtu.be/yIYKR4sgzI8) | 
| 02 | [K-Nearest-Neighbours](https://github.com/BaoLocPham/MachineLearningFromScratch/blob/main/Part%203%20-%20Classification/Section%2015%20-%20K-Nearest%20Neighbors%20(K-NN)/k_nearest_neighbors.ipynb) | [StatQuest: KNN](https://youtu.be/HVXime0nQeI) |
| 03 | [Support Vector Machine](https://github.com/BaoLocPham/MachineLearningFromScratch/blob/main/Part%203%20-%20Classification/Section%2016%20-%20Support%20Vector%20Machine%20(SVM)/support_vector_machine.ipynb) |  [StatQuest: SVM](https://youtu.be/efR1C6CvhmE) |
| 04 | [Kernel SVM](https://github.com/BaoLocPham/MachineLearningFromScratch/blob/main/Part%203%20-%20Classification/Section%2017%20-%20Kernel%20SVM/kernel_svm.ipynb) |  [StatQuest: Polinomial Kernel](https://youtu.be/Toet3EiSFcM) [StatQuest: RBF kernel](https://youtu.be/Qc5IyLW_hns) |
| 05 | [Naive Bayes](https://github.com/BaoLocPham/MachineLearningFromScratch/blob/main/Part%203%20-%20Classification/Section%2018%20-%20Naive%20Bayes/naive_bayes.ipynb) |  [StatQuest: Naive Bayes](https://youtu.be/O2L2Uv9pdDA) [StatQuest: Gaussian Naive Bayes](https://youtu.be/H3EjCKtlVog) |
| 06 | [Decision Tree](https://github.com/BaoLocPham/MachineLearningFromScratch/blob/main/Part%203%20-%20Classification/Section%2019%20-%20Decision%20Tree%20Classification/decision_tree_classification.ipynb) |  [StatQuest: Decision Tree Regression](https://youtu.be/7VeUPuFGJHk) |
| 04 | [Random Forest Regression](https://github.com/BaoLocPham/MachineLearningFromScratch/blob/main/Part%203%20-%20Classification/Section%2020%20-%20Random%20Forest%20Classification/random_forest_classification.ipynb) |  [StatQuest: Random Forest](https://youtu.be/J4Wdy0Wc_xQ) |

#### Classifications: Pros and Cons
| Classification Model | Pros | Cons |
| ----------- | ----------- | ----------- |
| Logistic Regression | Probabilistics approach, gives informations about statiscal significance of features. | [The Logistic Regression Assumptions](https://www.statology.org/assumptions-of-logistic-regression/). |
| K-NN | Simple to understand, fast and efficient. | Need to choose the number of neighbours K. |
| SVM | Performant, not biased by outliers, not sensitive to overfitting. | Not appropriate for nonlinear problems, not the best choice for large number of features. |
| Kernel SVM | High performance on nonlinear problems, not biased by outliers, not sensitive to overfitting. | Not the best choice for large number of features, more complex. |
| Naive Bayes | Efficient not biased by outliers, works on nonlinear problems, probabilitstic approach. | Based on the assumption that features have same statistical relevance. |
| Decision Tree Classification | Interpretability, no need for feature scaling, works on both linear, nonlinear problems. | Poor results on too small datasets, overfitting can easily occur. |
| Random Forest Classification | Powerful and accurate, good performance on many problems, including nonlinear. | No interpretability, overfitting can easily occur, need to choose the number of trees. |
***
### Clustering
| Number     | Notebook | Extras |
| ----------- | ----------- | ----------- |
| 01 | [KMean](https://github.com/BaoLocPham/MachineLearningFromScratch/blob/main/Part%204%20-%20Clustering/Section%2024%20-%20K-Means%20Clustering/k_means_clustering.ipynb) | [StatQuest: KMeans Clustering](https://youtu.be/4b5d3muPQmA) , [WCSS and Elbow method](https://analyticsindiamag.com/beginners-guide-to-k-means-clustering/) | 
| 02 | [Hierarchical](https://github.com/BaoLocPham/MachineLearningFromScratch/blob/main/Part%204%20-%20Clustering/Section%2025%20-%20Hierarchical%20Clustering/hierarchical_clustering.ipynb) | [StatQuest: Hierarchical Clustering](https://youtu.be/7xHsRkOdVwo) , [Dendrogram method](https://towardsdatascience.com/agglomerative-clustering-and-dendrograms-explained-29fc12b85f23) | 

#### Clustering: Pros and Cons
| Regression Model | Pros | Cons |
| ----------- | ----------- | ----------- |
| K-Means | Simple to understand, easily adaptable, works well on small or large datasets, fast, efficient and performant. | Need to choose the number of cluster. |
| Hierarchical Clustering | The optimal number of clusters can be obtained by the model itself, pratical visualization with the [**dendrogram**](https://en.wikipedia.org/wiki/Dendrogram). | Not appropriate for large datasets. |
***
### Association Rule Learning
| Number     | Notebook | Extras |
| ----------- | ----------- | ----------- |
| 01 | [Apriori](https://github.com/BaoLocPham/MachineLearningFromScratch/blob/main/Part%205%20-%20Association%20Rule%20Learning/Section%2028%20-%20Apriori/apriori.ipynb) | [Apriori Algorithm](https://towardsdatascience.com/apriori-algorithm-for-association-rule-learning-how-to-find-clear-links-between-transactions-bf7ebc22cf0a?gi=7825eb554041) |
| 02 | [Eclat](https://github.com/BaoLocPham/MachineLearningFromScratch/blob/main/Part%205%20-%20Association%20Rule%20Learning/Section%2029%20-%20Eclat/eclat.ipynb) | |
***
### Reinforcement Learning
| Number     | Notebook | Extras |
| ----------- | ----------- | ----------- |
| 01 | [Upper Confidence Bound](https://github.com/BaoLocPham/MachineLearningFromScratch/blob/main/Part%206%20-%20Reinforcement%20Learning/Section%2032%20-%20Upper%20Confidence%20Bound%20(UCB)/upper_confidence_bound.ipynb) | [Confidence Bounds](https://www.weibull.com/hotwire/issue34/relbasics34.htm), [UCB and Multi-armed bandit problem](https://www.aionlinecourse.com/tutorial/machine-learning/upper-confidence-bound-%28ucb%29)  | 
| 02 | [Thomson Sampling](https://github.com/BaoLocPham/MachineLearningFromScratch/blob/main/Part%206%20-%20Reinforcement%20Learning/Section%2033%20-%20Thompson%20Sampling/thompson_sampling.ipynb) | [Thomson Sampling](https://towardsdatascience.com/thompson-sampling-fc28817eacb8) | 
#### The Multi-Armed Bandit Problem
* [ritvikmath: The Multi-Armed Bandit Stategies](https://youtu.be/e3L4VocZnnQ)
* [ritvikmath: The strategies and UCB approach](https://youtu.be/FgmMK6RPU1c)
* [ritvikmath: The Thomson sampling algorithm](https://youtu.be/Zgwfw3bzSmQ)
***
### NLP
| Number     | Notebook | Extras |
| ----------- | ----------- | ----------- |
| 01 | [Introduction to nlp](https://github.com/BaoLocPham/MachineLearningFromScratch/blob/main/Part%207%20-%20NLP/Section%2036%20-%20Introduction%20to%20NLP/natural_language_processing.ipynb) |   | 
***
### Dimensionality Reduction
| Number     | Notebook | Extras |
| ----------- | ----------- | ----------- |
| 01 | [Principal Component Analysis](https://github.com/BaoLocPham/MachineLearningFromScratch/blob/main/Part%208%20-%20Dimensionality%20Reduction/Section%2043%20-%20PCA/principal_component_analysis.ipynb) | [setosa-PCA example](https://setosa.io/ev/principal-component-analysis/), [StatQuest-PCA](https://youtu.be/FgakZw6K1QQ), [plotly-PCA visualization](https://plotly.com/python/pca-visualization/) | 
***
### Model selection
| Number     | Notebooks | Extras |
| ----------- | ----------- | ----------- |
| 01 | [Regression](https://github.com/BaoLocPham/MachineLearningFromScratch/tree/main/Part%20Extra%20-%20Model%20Selection/Regression) |  |
| 02 | [Classification](https://github.com/BaoLocPham/MachineLearningFromScratch/tree/main/Part%20Extra%20-%20Model%20Selection/Classification) | [The Accuracy paradox](https://towardsdatascience.com/accuracy-paradox-897a69e2dd9b), [AUC-ROC and CAP Curves](https://medium.com/geekculture/classification-model-performance-evaluation-using-auc-roc-and-cap-curves-66a1b3fc0480), [Precision, Recall and F-1 score](https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9) |

### Case study
| Number     | Notebooks | Extras |
| ----------- | ----------- | ----------- |
| 01 | [Logistic Regression](https://github.com/BaoLocPham/MachineLearningFromScratch/blob/main/Part%20Extra%20-%20CaseStudy/Breast%20Cancer%20Logistic%20Regression/logistic_regression.ipynb) | Breast Cancer classifier  |
## Extras
### Datasets:
1. [ICU dataset](https://archive.ics.uci.edu/ml/index.php)
2. [Repo datasets](https://github.com/BaoLocPham/MachineLearningFromScratch/tree/main/Data)
### Blogs:
* [machinelearningcoban](https://machinelearningcoban.com)
* [Deep Ai Khanh blog](https://phamdinhkhanh.github.io/deepai-book/intro.html)

## Acknowledge:
* Thanks [Kirill Eremenko](https://twitter.com/kirill_eremenko), [Hadelin de Ponteves](https://twitter.com/hadelin2p) for creating such an awesome about machine learning online.
* Thanks [Josh Starmer aka StatQuest](https://www.youtube.com/channel/UCtYLUTtgS3k1Fg4y5tAhLbw) for your brilliant video about machine learning, help me alot of understanding the math behind the ML algorithm.
* Thanks mr [Vũ Hữu Tiệp](https://machinelearningcoban.com/about/) for your brilliant blogs about machine learning, helps me a lot from the day i didn't know what is machine learning is.
* Thanks mr [Phạm Đình Khánh](https://phamdinhkhanh.github.io/deepai-book/intro.html) for your blogs about machine learning and deep learning.
