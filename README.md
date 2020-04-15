# Detecting Fake Jobposting
## Group members
| Name  | Student number |
| ------------- | ------------- |
| [魏麟懿Linyi Wei](https://github.com/Linyi-Wei)  |  1901212647 |
| [庞博Bo Pang]()  | 1901212498  |
| [赵舒婷Shuting Zhao](https://github.com/Shuuting) | 1901212679 |
# PART 1 Introduction
## 1.1 Motivation
Recently, there's a news that a finance major pretended to be a interviewer of CICC or CITIC to get answesrs of written examination from those interviewers. With such scams keeping emerging, abilities to identfy fake jobposting becomes more important. In the past years, people distinguish fake jobposting by intuition. For example, abnormal high wage may suggest fake jobposting. While nowadays big data techique enable us to process these jobposting data and identify fake ones more reliable using model.
<br>`Our goal of this project is to train classifiers to recognize fake or real jobposting` using features like location,department,salary_range,benefits,required_experience,required_education,industry and so on.
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
（插入一张流程图）

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
a. Delete 6 features for reasons as follows:
  * 'job_id' is unique for each sample and useless for detecting fake job.
  * 'title' is nearly unique and difficult to deal with.
  * 'location' is also difficult to handle and we want to find the generality of fake jobposting wherever it is.
  * 'department' is just like job title with many different values.
  * 'industry' and 'function' is also difficult to deal with and uselless for finding generality.
<br>
<br>b. To avoid text analysis and simplify the model, use the length of 'company_profile'，'description'，'requirements'，'benefits' to replace the text.
<br>
<br>c. Map ordinal feature: 'required_education'，'required_experience'
* Map required education level including 'Bachelor's Degree', 'High School or equivalent'.etc with integer from 0 to 10.
* Map required experience including 'Mid-Senior level','Associate'.etc with integer from 0 to 5.
<br>
<br>d. Use one-hot encoding to encode nominal features
* Employmnet type is encoded to 4 columns to represent 5 different types. 
