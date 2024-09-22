## Amazon Sentiment Analysis with Machine Learning and NLP

This project explores advanced sentiment analysis techniques on Amazon product reviews using Natural Language Processing (NLP) and Machine Learning. The model predicts the sentiment of customer reviews—whether positive, neutral, or negative—providing actionable business insights that can inform product strategy, customer engagement, and brand management.

### Project Overview

The goal of this project is to automate the sentiment classification of Amazon product reviews to help businesses understand customer perceptions at scale. By classifying reviews as positive, neutral, or negative, companies can identify key trends, track product performance, and better align their offerings with customer expectations.

### Business Application

•	Product Feedback Analysis: Identify products with high levels of customer dissatisfaction and take corrective actions.

•	Customer Retention Strategies: Detect negative reviews early to engage with dissatisfied customers and prevent churn.

•	Market Research: Analyze sentiments across product categories to understand market demand and consumer preferences.

### Tech Stack and Tools Used

•	Languages: Python

•	Data Manipulation & Analysis: **pandas**, **numpy**

•	Data Visualization: **matplotlib**, **seaborn**

•	NLP Techniques:

  o	Tokenization using **CountVectorizer**
  
  o	Text feature extraction using **TF-IDF**
  
•	Machine Learning:

  o	  Model: **LinearSVC** (Support Vector Classifier)
  
  o	  Hyperparameter tuning with **GridSearchCV**
  
  o	  Pipeline for vectorization, transformation, and classification
  
•	Interactive UI: **Gradio** (Real-time sentiment prediction interface)

### Pipeline Summary

1.	Data Preprocessing:

•	Dataset: The dataset used consists of over 34,000 Amazon product reviews with attributes such as review text and product rating.
  
•	Handling missing values: All missing values in critical columns like reviews.rating were removed, and text data was cleaned for tokenization.
  
•	Text Vectorization:

  • Implemented **CountVectorizer** to tokenize text and convert it into a matrix of token counts.
  
  • Applied **TF-IDF** transformation to weight the importance of terms based on their frequency across reviews.

2.	Model Building & Evaluation:
   
•	Classifier: Trained a **LinearSVC** model within a pipeline that includes **CountVectorizer** and **TF-IDF**.

•	Initial Model Accuracy: The initial model achieved **93.94%** accuracy on the test set.

•	After Hyperparameter Tuning: With **GridSearchCV**, the model's accuracy improved to **94.08%**.

•	Hyperparameter Tuning:

  •	Explored **ngram_range** and **use_idf** parameters in **GridSearchCV** for optimizing text representation.
  
  •	Final parameter selection: **ngram_range=(1, 2), use_idf=True**.
  
•	Classification Report:
  
  	Precision: 0.94
    Recall: 0.94
    F1-score: 0.94

3.	Model Performance Metrics:
   
•	Accuracy before tuning: 93.94%

•	Accuracy after tuning: 94.08%

•	Precision: 0.94

•	Recall: 0.94

•	F1-score: 0.94

•	Model performance was validated using a test set from a stratified split of the dataset to ensure even distribution of review ratings.

4.	Interactive Sentiment Prediction Interface:

Designed an intuitive web interface using **Gradio** for real-time sentiment predictions. Users can input review text and instantly receive the predicted sentiment as output.

### Business Insights Derived from the Project

•	Identifying Customer Pain Points: By analyzing negative reviews, businesses can quickly identify common customer pain points, such as issues with product quality, customer service, or delivery, and take corrective action.

•	Improving Product Strategy: Insights from sentiment analysis can highlight what customers appreciate most about products. Companies can leverage this feedback to enhance product features and optimize their offerings for future releases.

•	Real-time Customer Feedback: With an interactive sentiment analysis tool, businesses can monitor customer feedback in real-time, enabling them to be proactive in addressing concerns and improving customer satisfaction.

•	Segmenting Customer Responses: By classifying reviews by sentiment, businesses can segment customers based on their feedback and develop tailored marketing or customer engagement strategies for different groups.

### Future Enhancements

•	Model Expansion: Incorporate deep learning models like **LSTM** or **BERT** for more complex text representation and sentiment detection.

•	Fine-Grained Sentiment Classification: Expand the model to predict more nuanced sentiment classes, such as highly positive, neutral, or highly negative.

•	Feature Engineering: Explore additional features such as review length, product categories, or customer metadata to improve prediction accuracy.




