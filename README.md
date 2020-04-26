# Detecting Fake Jobposting :smile:
## Group members
| Name  | Student number |
| ------------- | ------------- |
| [魏麟懿 Linyi Wei](https://github.com/Linyi-Wei)  |  1901212647 |
| [庞博 Bo Pang]()  | 1901212498  |
| [赵舒婷 Shuting Zhao](https://github.com/Shuuting) | 1901212679 |
# PART 1 Introduction
## 1.1 Motivation
Recently, there's a news that a finance major pretended to be a interviewer of CICC or CITIC to get answesrs of written examination from those interviewees. With such scams keeping emerging, abilities to identfy fake jobposting becomes more important. In the past years, people distinguish fake jobposting by intuition. For example, abnormal high wage may suggest fake jobposting. While nowadays big data techique enable us to process these jobposting data and identify fake ones more reliable using model.
<br>`Our goal of this project is to train classifiers to recognize fake or real jobposting` using features like salary_range, benefits, required_experience, required_education and so on.
## 1.2 Data Sources
Dataset of real and fake job postings created by Shivam Bansal:<br>https://www.kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction 
<br>Our data preprocessing also referred to this notebook:<br>https://www.kaggle.com/nikitaalbert/is-this-job-for-real

## 1.3 Data Description
There are 18 variables initially contains 17 features and one target label.
|Variables|Description|
| ------------- | ------------- |
|job_id| Unique Job ID|
|title| The title of the job ad entry.Most likely unique for each entry.|
|location| Geographical location of the job ad：Country, State, City.|
|department| Corporate department (e.g. sales).Most likely unique for each posting.|
|salary_range| Indicative salary range (e.g.  50,000− 60,000).From an initial glance of the head, we see its blank; However, in subsequent analysis, we see that it is in format MIN-MAX.|
|company_profile| A brief company description.|
|description| The details description of the job ad.|
|requirements| Enlisted requirements for the job opening.|
|benefits| Enlisted offered benefits by the employer.|
|telecommuting| True for telecommuting positions.|
|has_company_logo| True if company logo is present.|
|has_questions| True if screening questions are present.|
|employment_type| Full-type, Part-time, Contract, etc.|
|required_experience| Executive, Entry level, Intern, etc.|
|required_education| Doctorate, Master’s Degree, Bachelor, etc.|
|industry| Automotive, IT, Health care, Real estate, etc.|
|function| Consulting, Engineering, Research, Sales etc.|
|fraudulent| target - Classification attribute.|
## 1.4 Framework
<img hight=400 width=300 src="https://github.com/Linyi-Wei/2020MLF-PROJECT/blob/master/4.Model%20training/Format.jpg"/>

# Part 2 Exploratory Data Analysis
## 2.1 The distribution of category labels
Let's take a look at how many counts of real and fake posts there are, in relation to the top unique values of a feature.
![](https://github.com/Linyi-Wei/2020MLF-PROJECT/blob/master/2.Exploratory%20Data%20Analysis/Catagory%20Label.png)
From the graph, we can see that:
* Fraudulent posts are mostly not posted as telecommuting ones, like real posts.
* Fraudulent posts mostly do not contain a company logo, unlike real posts.
* Fraudulent posts have an equal mix of either having a questionnaire or not, like real posts.
* Fraudulent posts are mostly full-time, like real posts.
* Fraudulent posts, also do not specify the required experience and education necessary, like real posts.
## 2.2 The distribution of text features' length
We want to see whether the length of 'company_profile', 'description', 'requirements', 'benefits' can be used to detect fake jobposting.
![](https://github.com/Linyi-Wei/2020MLF-PROJECT/blob/master/2.Exploratory%20Data%20Analysis/company%20profile.png)
![](https://github.com/Linyi-Wei/2020MLF-PROJECT/blob/master/2.Exploratory%20Data%20Analysis/company%20description.png)
![](https://github.com/Linyi-Wei/2020MLF-PROJECT/blob/master/2.Exploratory%20Data%20Analysis/company%20requirements.png)
![](https://github.com/Linyi-Wei/2020MLF-PROJECT/blob/master/2.Exploratory%20Data%20Analysis/company%20benefits.png)
Fake jobposting posts similar-length description, requirements and benefit to make them more reliable. While there's differences in company profile. Fake jobpostings are not tend to post short & long company profile.

# Part 3 Data Preprocessing
## 3.1 Feature trimming
#### a. Delete 6 features for reasons as follows:
  * 'job_id' is unique for each sample and useless for detecting fake job.
  * 'title' is nearly unique and difficult to deal with.
  * 'location' is also difficult to handle and we want to find the generality of fake jobposting wherever it is.
  * 'department' is just like job title with many different values.
  * 'industry' and 'function' is also difficult to deal with and uselless for finding generality.
#### b. Replace text features with length
 * To avoid text analysis and simplify the model, use the length of 'company_profile'，'description'，'requirements'，'benefits' to replace the text.
#### c. Divide 'salary_range' into two features: 'salary_range_min' and 'salary_range_max'

#### d. Map ordinal feature: 'required_education'，'required_experience'
 * Map required education level including 'Bachelor's Degree', 'High School or equivalent'.etc with integer from 0 to 10.
Map required experience including 'Mid-Senior level','Associate'.etc with integer from 0 to 5.
#### e. Use one-hot encoding to encode nominal features
 * Employment type is encoded to 4 columns to represent 5 different types. 

## 3.2 Feature Selection
After feature trimming, there are 17 features. In the meantime,  one-hot excoding may cause multicollinearity. So we need select important features. 
<br>We choose KNN to do sequential background selection.
<br><div align=center>![](https://github.com/Linyi-Wei/2020MLF-PROJECT/blob/master/3.%20Data%20Preprocessing/KNN.png)</div>
<br>We finally choose 6 features ('required_education', 'salary_range_min', 'employment_type_Other',
       'employment_type_Part-time', 'employment_type_Temporary',
       'employment_type_Unknown') which perform a relative high accuracy over 0.975. 
<br>After feature selection, training accuracy is 0.979 and test accuracy is 0.977 with only no more than 0.5 percent differences to that before. Six features is reasonable to be selected.

## 3.3 Feature correlation
After feature selection based on KNN, we also test the correlation of these features. And the results are shown below.
<br><div align=center>![](https://github.com/Linyi-Wei/2020MLF-PROJECT/blob/master/3.Data%20Preprocessing/data%20correlation.jpg)</div>
<br><div align=center>![](https://github.com/Linyi-Wei/2020MLF-PROJECT/blob/master/3.Data%20Preprocessing/feature%20correlation.jpg)</div>

# Part 4 Model training, evaluation, and hyperparameter tuning in cross validation
For the y variable and the six X variables obtained in 3.2, we used LR / SVM / Tree three classification methods to fit the model, and called the GridSearchCV package to carry out Cross-Validation.
<br>As mentioned in our motivation section, many fake JD senders will often ask the delivery person to solve many additional problems, and these answers are often used for profit. Our research goal is to avoid the loss of time, energy and personal information exposure caused by delivering information to fake JD. In our sample, we use y = 0 (true JD) as the positive label parameter in the code, so we use PRE as the model cross-validation screening and evaluation standard. The larger the PRE value, the smaller the FP/P ratio, and the easier it is for job seekers to avoid false JD.
<br>Listed below are the optimal parameters and corresponding confusion_matrix we obtained for different models.

## 4.1	Logistic Regression
<br><img align="right" src="https://github.com/Linyi-Wei/2020MLF-PROJECT/blob/master/4.Model%20training/KNN-LR.png"/>
#### C = 0.001:
Precision: 0.951
<br>Recall: 1.000 
<br>F1: 0.975

## 4.2	SVM
(We only run linear kernel due to the CPU limitation.)
<img align="right" src="https://github.com/Linyi-Wei/2020MLF-PROJECT/blob/master/4.Model%20training/KNN-SVM.jpg"/>
#### C = 0.001:	
Precision: 0.951
<br>Recall: 1.000
<br>F1: 0.975

## 4.3	Decision Tree
<img align="right" src="https://github.com/Linyi-Wei/2020MLF-PROJECT/blob/master/4.Model%20training/KNN-Tree.jpg"/>

#### max_depth = 14:	
Precision: 0.957
<br>Recall: 0.997
<br>F1: 0.977
<br>
<br>
<br><div align=center>![](https://github.com/Linyi-Wei/2020MLF-PROJECT/blob/master/4.Model%20training/Decision%20Tree.jpg)</div>
<br>
<br>
<br>Among these 3 methods, Decision Tree provides the best results with PRE= 95.7%. Notice that both LR and SVM give the same PRE as randomly trusting every JD, so we guess there may be some problems in KNN feature selection. This is why we try the PCA method in next part.

# Part 5 Redo the process in PCA dimensionality-reduced data
In this part, we use PCA method to replace the KNN method in 3.2. We set components n=2 for better program velocity. After that, we conduct LR/SVM/Tree algorithms again. Notice in code we use pipeline method to package them together.
The following table illustrates the final results for our models.
|Model Type| PRE* | REC | F1-score |
| ------|------- | ------|------- |
| PCA_LR | 0.884 | 0.936 | 0.909 |
| PCA_SVM(rbf) | 0.972 | 0.919 | 0.945 |
| PCA_Tree | 0.974* | 0.875 | 0.922 |

We can see that for Logistic Regression the PCA method is worse than the KNN, but SVM and Decision Tree show the different, which means PCA covers some additional info than KNN. (PS: We only have two X features now, so we use better Grid -Search hyper parameters. Hence the increase in SVM may due to this cross-validation change.)
<br>Notice that PCA-Tree, with 0.974 PRE, is the best method among them. We draw an ROC curve to deeply analyze this model.
<br><div align=center>![](https://github.com/Linyi-Wei/2020MLF-PROJECT/blob/master/4.Model%20training/ROC.jpg)</div>

# Part 6 Apply more advanced model: Random forest/Bagging/Adaboost
In this part, we use both PCA dimensionality-reduced data and KNN feature selection data. These data are both put into training Random forest, Bagging and Adaboost these three model. 
<br>In order to show our results more succinctly, the following table is used to show the results. 
<br>
|Model Type| PRE* | REC | F1-score |
| ------|------- | ------|------- |
| KNN_RF | 0.991 |  0.988 | 0.989 |
| KNN_Bagging | 0.993 | 0.987 | 0.990 |
| KNN_Adaboost | 0.994 | 0.976 | 0.985 |
| PCA_RF | 0.943 | 0.941 | 0.942 |
| PCA_Bagging | 0.946 | 0.943 | 0.945 |
| PCA_Adaboost | 0.848 | 0.960 | 0.901 |
<br>
We can see the models based on KNN data shows a higher Precision, which are better models. The reason behind this is that we just use 2 pca components. But part3.3 show that the feature correlation is very weak, which means 2 pca components are not enough to represent the whole feature and explain the results.
<br>As we mentioned above, we care more about the precision index. In the models based on KNN data, the adaboost shows the best results.
<br>
Generally speaking, the following table shows all the traing model results based both PCA and KNN data.
<br>
|Model Type| PRE* | REC | F1-score |
| ------|------- | ------|------- |
| KNN_LR | 0.951 | 1.000 | 0.975 |
| KNN_SVM(line) | 0.951 | 1.000 | 0.975 |
| KNN_Tree | 0.957 | 0.997 | 0.922 |
| PCA_LR | 0.884 | 0.936 | 0.909 |
| PCA_SVM(rbf) | 0.972 | 0.919 | 0.945 |
| PCA_Tree | 0.974* | 0.875 | 0.977 |
| KNN_RF | 0.991 |  0.988 | 0.989 |
| KNN_Bagging | 0.993 | 0.987 | 0.990 |
| KNN_Adaboost | 0.994 | 0.976 | 0.985 |
| PCA_RF | 0.943 | 0.941 | 0.942 |
| PCA_Bagging | 0.946 | 0.943 | 0.945 |
| PCA_Adaboost | 0.848 | 0.960 | 0.901 |
<br>
Based on the table, Adaboost based on KNN data shows the best results.
