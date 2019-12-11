# Using Data to Bring Customers Home

## Authors
1. Devanshi Gariba
2. Alefiya Naseem

## Summary
Wayfair is an e-commerce retailer that sells furniture and other home goods. While they sell to individual consumers, they also have a large B2B (Business-to-Business) division that sells to business customers such as interior design firms, contractors, and hotels.  
As a data-driven company, they ensure that their B2B customers receive best-in-class service by leveraging data science models to predict customer needs and purchasing patterns. They primarily aim to reinvent the way the world shops for home and utilizes machine -learning models in many of their departments including marketing, sales, and operations teams to guide business decisions. Often times, business vendors bring in most profits and it becomes really important for a company to personalize efforts for them and retain them and take appropriate measures if they are not retained.  
Keeping this in mind, Wayfair hosted a challenge on the ScholarJet website this year for which they released their B2B customer interaction dataset that will be used here. The goal of the challenge was to build a model to predict customer behavior for Wayfair. The goal of our project is to gain insights from the business customer data like customer information,sales call records, purchase history etc. and build predictive models to work on the following problems:

1. B2B customer conversion (classification): Whether a B2B customer will purchase or not in the next 30 days
2. B2B customer expected revenue (regression): How much a B2B customer will spend in the next 30 days

## Data and Dictionary
1. Training data
This data includes 181 features. These features can be divided into the following feature groups.
	1. Two outcome variables:
		* convert_30 (boolean)
		* revenue_30 (numeric)
	2. Customer
	3. Enrollment questions
	4. Order
	5. Satisfaction
	6. Visit
	7. Search
	8. SKU
	9. Task
	10. Call
	11. Email-BAM
	12. Email-Wayfair

2. Holdout data
This data includes features but not outcome variables for a different set of customers. We use these features to predict the missing outcome variables.

3. Dictionary to Define Variables

##  Proposed Plan of Research
Our project will take place over the entire semester. We aim to complete the data cleaning, preprocessing and the classification problem in the first half. In the second half of the semester we aim to complete the regression problem, enhance our model performances and deploy the model on the web. We plan to deploy and experiment with different regression and classification machine learning models including (but not limited to) typical regression and classification methods like linear regression,regression trees,decision trees, random forests, neural networks (depending on how different methods perform on our dataset) different dimensionality techniques and feature extraction methods to make a typical business case study. The web app has the potential to be a customer activity indicator as once it has the revenue it can display items like: 
1. The obvious mean and median.
2. Revenue by customer type (pie chart for e.g)
3. Lineplots showing satisfaction and correlation indicating good brand service.
4. Predicted revenue by Task type to show top revenue generating tasks.

More insight oriented extensions can be added to the web app over time. Our aim is to not just take care of high retention and revenue prediction accuracies but also to drive insights from these predictions, evaluate and provide reasoning for which models seem to work best for our use-case. 


## Preliminary Results

[Click here to see the notebook containing the preliminary analysis](https://github.com/alefiya-naseem/CustomerRetention-Revenue/blob/master/Notebooks/dataviz_prelim.ipynb)

## References
https://app.scholarjet.com/challenges/wayfairdata
