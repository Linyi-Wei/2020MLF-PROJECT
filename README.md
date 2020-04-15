# Detecting Fake Jobposting
## Group members
| Name  | Student number |
| ------------- | ------------- |
| [魏麟懿Linyi Wei](https://github.com/Linyi-Wei)  |  1901212647 |
| [庞博Bo Pang]()  | 1901212498  |
| [赵舒婷Shuting Zhao](https://github.com/Shuuting) | 1901212679 |
## PART 1 Introduction
### 1.1 Motivation
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
### 1.4 Framework
（插入一张流程图）

## Part 2 Exploratory Data Analysis
### 2.1 The explanation power of category labels
Let's take a look at how many counts of real and fake posts there are, in relation to the top unique values of a feature.
<div align="center">
<img src="https://github.com/Linyi-Wei/2020MLF-PROJECT/blob/master/2.Exploratory%20Data%20Analysis/telecommuting.jpg" height="300px" width="430px" alt="图片说明" > 
 <img src="https://github.com/Linyi-Wei/2020MLF-PROJECT/blob/master/2.Exploratory%20Data%20Analysis/logo.jpg" height="300px" width="430px" alt="图片说明" >
</div>

<div align="center">
<img src="https://github.com/Linyi-Wei/2020MLF-PROJECT/blob/master/2.Exploratory%20Data%20Analysis/question.jpg" height="300px" width="430px" alt="图片说明" > 
 <img src="https://github.com/Linyi-Wei/2020MLF-PROJECT/blob/master/2.Exploratory%20Data%20Analysis/type.jpg" height="300px" width="430px" alt="图片说明" >
</div>

<div align="center">
<img src="https://github.com/Linyi-Wei/2020MLF-PROJECT/blob/master/2.Exploratory%20Data%20Analysis/exp.jpg" height="300px" width="430px" alt="图片说明" > 
 <img src="https://github.com/Linyi-Wei/2020MLF-PROJECT/blob/master/2.Exploratory%20Data%20Analysis/edu.jpg" height="300px" width="430px" alt="图片说明" >
</div>

![](https://www.kaggleusercontent.com/kf/31604270/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..JEYS1ldY5tjCSNSWllKWXg.txUS6QP9IBKHsMny1Y1fcZFwRQohk2gklBroGRwqWSBcpqAL9HwRpzViUkY8P2l0Ue8S7qF8-vYgaDw8JtJ5dTksVgeMVNpL-73RNlNOZlLrSdco-4rEg9Exn-YhhyDdmX-voJuLEKatzR5-_NCQSjFxKswjPRd8HKbCcXlvdzKzbtlaPRalf8jZ8o4rh1ro1YVdvS7ToKMqn-3fsMGTxNe0VwBdkxk8YzpupjbmmI-AIMaOvg_pu-PYMfGsQnEjzthjOABMvPSXcDsrefb6aKo49ZVP1o2sJAjcPYWtL9V1OyyJQuIbT4UoIlNMsSiej3njzjh-M3aDLUIlthCn00kACI31XKk8eXzhiOFILC07VdOnCCdcgXlE-jaIaqwrFOh67c4eRqUzPxQ6BS4kUICJkHI-7e7oZvdOdZofDCQ9VIl5ZCM17jThXUOQgQvYlfcr5TE8VMIHHOH9vMeCJfV2rSsHgblORxG6Rc23W-3r7gk5wlCe3u3xbXX6q-b-s2D1ggeN518oOTYJShYXQfC_B315UIxCzbcKkiy4taSyeWH2m5hJ6n1mkGkF3M7MC9tURT4AMCgEdXPU1kLoTqxLVq10_b_-4iK1NJIBDgAyZLqTg4pDhqk6Z_-YhcV8PKhVidUEiVs6Z-yPg6vP8cn955VqphxYUmg_RF3N4rY.Us-BEQYY-Vb1vtBSc7eN4w/__results___files/__results___19_0.png)
