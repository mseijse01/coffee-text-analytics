# Leveraging Text Analytics and Predictive Modeling to Analyze Consumer Coffee Reviews: A Data-Driven Approach

**Author:** Marcelo Seijas (446662)  
**Program:** Data Science and Marketing Analytics  
**Supervisor:** Eoghan O'Neill  
**Second Assessor:** Sean Brüggemann  
**Institution:** Erasmus University Rotterdam, Erasmus School of Economics

---

## Abstract

In the growing specialty coffee market, understanding the factors that influence consumer ratings can provide valuable insights for producers, marketers, and retailers. This study investigates the key sensory and non-sensory attributes that drive consumer preferences by analyzing coffee reviews from CoffeeReview.com using a combination of text analytics, sentiment analysis, Multinomial Inverse Regression (MNIR), and machine learning. A diverse set of features, including flavor attributes, categorical variables such as country of origin and roast level, and text-based features derived from BERT embeddings, GloVe vectors, and LDA topics, were used to predict coffee ratings.

XGBoost emerged as the best-performing model in handling both structured (e.g., flavor and categorical features) and unstructured (e.g., text-based) data. MNIR provided additional insights by focusing on the role of text features in predicting sensory attributes such as acidity, body, and flavor, which were themselves key drivers of higher consumer ratings. Acidity was identified as the most significant predictor of higher ratings, while geographical origin and roast level also played important roles in shaping consumer perceptions.

This research provides a comprehensive framework for leveraging consumer-generated reviews to gain actionable insights. The findings suggest that businesses should highlight key flavor attributes, origin, and narrative elements in their product descriptions to align with consumer expectations. The study also demonstrates the importance of integrating machine learning techniques with natural language processing (NLP) to capture the nuances of consumer sentiment. While this research focuses on the specialty coffee market, the methods and insights can be applied to other artisanal industries, such as wine and craft beer, where sensory experiences and non-sensory attributes play critical roles in consumer decision-making.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Literature Review](#literature-review)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Conclusions](#conclusions)
6. [References](#references)
7. [Appendix](#appendix)

---

## Introduction

The global coffee industry has evolved into a dynamic market, now valued at more than $100 billion annually, according to International Coffee Organization (2021). Coffee remains one of the most traded commodities worldwide, with the specialty coffee segment experiencing particularly strong growth. This growth is driven by an increasing demand for high-quality, ethically sourced, and uniquely flavored coffee, which has led to a compound annual growth rate (CAGR) of approximately 7% over the past decade (SCA, 2020). This expansion underscores the importance of understanding consumer preferences, as businesses strive to differentiate themselves by aligning with the values and tastes of a discerning consumer base.

Today's coffee consumers, as Carman (2020) describes, are increasingly engaged in various aspects of their coffee consumption beyond flavor, including the origin, sustainability, and ethical practices behind production. This shift is characteristic of the "Third Wave Coffee" movement, which emphasizes artisanal quality and the celebration of coffee as a craft rather than a commodity. As businesses attempt to cater to these discerning consumers, online reviews on platforms like CoffeeReview.com provide valuable data capturing consumer preferences, ranging from casual drinkers to aficionados. These reviews offer insights into sensory experiences like taste and aroma, as well as broader considerations such as coffee variety and processing methods.

By analyzing reviews from CoffeeReview.com, this study has identified that sensory attributes, particularly **acidity** and **body**, are the strongest predictors of higher coffee ratings. Additionally, advanced text analytics revealed that the narrative and flavor descriptors within the text of reviews play a crucial role in driving consumer satisfaction. The study shows that terms associated with geographical origin and flavor richness, as well as transparent descriptions of processing methods, are significantly correlated with positive ratings.

Furthermore, this study finds that machine learning models, particularly **XGBoost**, outperform other techniques in predicting coffee ratings by integrating both structured (numerical, categorical) and unstructured (text-based) features. The use of Multinomial Inverse Regression (MNIR) also provided valuable insights by focusing on the predictive power of text features for sensory attributes such as acidity, body, aroma, aftertaste, and flavor. By utilizing SHAP (SHapley Additive exPlanations) values, we have also demonstrated which features—both sensory and non-sensory—are most influential in shaping consumer ratings.

### Research Questions

The primary goal of this study is to understand the factors that lead to higher coffee ratings in consumer reviews on CoffeeReview.com. The central research question guiding this study is:

> **What specific factors in consumer-generated reviews on CoffeeReview.com are most strongly associated with higher coffee ratings?**

To explore this overarching question, the study is structured around the following subquestions:

1. **Which types of features—numerical, categorical, or text—are most strongly associated with higher consumer coffee ratings on CoffeeReview.com?**
2. **How do different aspects of text descriptions, such as flavor notes and processing methods, correlate with higher coffee ratings?**
3. **How do various machine learning models perform in predicting coffee ratings, and which techniques are most effective in capturing associations between review features and ratings?**

These questions aim to determine the importance of different data types in predicting coffee ratings, uncover key textual attributes influencing these ratings, and evaluate the performance of different machine learning techniques in interpreting review data. By addressing these questions, the study has provided actionable insights into the factors driving consumer satisfaction and coffee ratings.

### Methodological Overview and Relevance

This research employs a mixed-methods approach, combining text analytics and machine learning techniques to analyze consumer-generated coffee reviews from CoffeeReview.com. The methodology includes text preprocessing, feature extraction using TF-IDF vectorization and word embeddings (GloVe and BERT), sentiment analysis, and topic modeling (LDA and NMF). To better understand sensory attributes such as acidity, body, aroma, aftertaste, and flavor based on text features, Multinomial Inverse Regression (MNIR) was applied. MNIR allowed for a focused analysis on how text features correlate with sensory aspects of coffee reviews. To perform feature selection, the LASSO (Least Absolute Shrinkage and Selection Operator) regression technique is applied, followed by predictive modeling using Linear Regression, Decision Trees, Random Forest, XGBoost, and Support Vector Regressors. Each model's effectiveness is assessed using mean absolute error (MAE), root mean squared error (RMSE), and R-squared (R²) as performance metrics.

The study's results indicate that XGBoost, due to its ability to handle high-dimensional data, performed best among the models, particularly when integrating both text-based features and sensory attributes. Specific text-based features, such as topics (flavor descriptions) and certain embeddings, were strongly correlated with higher ratings, revealing the importance of narrative elements in the reviews. The MNIR analysis added an additional layer of insight, specifically focusing on how textual features alone predict sensory attributes, which are themselves key drivers of ratings.

The relevance of this study lies in its ability to provide actionable insights for stakeholders in the specialty coffee market. By identifying the factors that contribute to higher consumer ratings, this research offers guidance for product development, marketing strategies, and consumer satisfaction enhancement. These findings can influence decisions on packaging, marketing, and product presentation in a highly competitive market where differentiation is key. Additionally, the application of advanced data science techniques contributes to the evolving methodologies in consumer behavior analysis.

### Contribution and Innovation

This research is innovative in its methodological approach, as it integrates advanced text analytics, machine learning techniques, and MNIR to extract valuable insights from unstructured data. Liu (2012) emphasizes the potential of natural language processing (NLP) and sentiment analysis in extracting meaningful patterns from consumer-generated reviews. Unlike previous studies that focused mainly on numerical and categorical data, this research capitalizes on the richness of text data to identify trends and preferences embedded in consumer reviews.

The key contribution of this study lies in shifting the focus from expert reviews to consumer perspectives. While expert reviews have traditionally been central to coffee quality assessments (Ponte, 2002), this research provides a more comprehensive understanding by analyzing consumer-generated content. By focusing on consumer experiences and preferences, the findings offer practical recommendations for aligning products and marketing strategies with evolving consumer demands.

Moreover, the combination of text-based insights with predictive modeling techniques contributes to the advancement of data-driven approaches in the coffee industry. The research bridges the gap between qualitative review content and quantitative analysis, demonstrating that text analytics can uncover critical drivers of consumer satisfaction. This methodology can be extended to other sectors, such as wine or craft beer, where sensory experiences and provenance play a significant role in shaping consumer behavior.

By providing both academic contributions and practical applications, this study highlights the value of combining data science with consumer behavior analysis, offering a robust framework for businesses in the specialty coffee market and beyond.

---

## Literature Review

This literature review examines the intersection of consumer preferences, online reviews, and advanced data analytics in the specialty coffee market, a rapidly growing segment valued at over $100 billion annually (International Coffee Organization, 2021). The shift toward high-quality, ethically sourced products has driven the growth of this market (SCA, 2020). Consumers today are more engaged with coffee production practices, ethical sourcing, and the sensory aspects of coffee, making it essential for businesses to understand these evolving preferences (Ponte, 2002).

Platforms like CoffeeReview.com provide valuable consumer-generated content, offering insights into the tastes and preferences of coffee enthusiasts. Systematic analysis of these reviews can reveal patterns that allow businesses to align their products with consumer expectations (Guo, 2017). Predicting coffee ratings from such reviews is increasingly recognized as a critical tool for the industry.

This review synthesizes existing literature on consumer reviews in the specialty coffee market, with a focus on the methodologies used to predict coffee ratings and understand consumer preferences. It explores the use of text analytics and sentiment analysis for extracting insights from unstructured data, such as online reviews (Pang, 2008). The review also covers market trends, the impact of online reviews on consumer decision-making, and the factors influencing consumer ratings.

Recent advancements in text analytics and machine learning have enabled more precise predictions and insights from consumer-generated content. This review highlights these methodological innovations and discusses their application within the specialty coffee industry, with an emphasis on how these techniques can improve our understanding of consumer preferences and behavior.

### The Specialty Coffee Market and Consumer Preferences

The specialty coffee market has emerged as a rapidly growing segment within the broader coffee industry, distinct from the mass retail and HoReCa (Hotel/Restaurant/Café) sectors. This market is characterized by a consumer preference for high-quality, ethically sourced, and uniquely flavored coffee, often driven by the "Third Wave Coffee" movement, which redefines coffee as an artisanal product rather than a commodity (Ponte, 2002). Specialty coffee consumers are willing to pay premiums for single-origin beans, small-batch roasts, and certifications like fair trade or organic, distinguishing their preferences from the mass market, which prioritizes price and convenience.

As noted by Sepulveda (2016), this shift aligns with broader consumer behavior trends favoring sustainability and ethical consumption, particularly among demographics that value transparency in production and sourcing practices. Understanding these preferences is crucial for businesses aiming to differentiate themselves in this competitive landscape, as noted by Mitas (2024). The interplay of sensory and non-sensory attributes in coffee—such as flavor, aroma, and body alongside ethical sourcing and transparency—drives consumer decisions. Ufer (2019) highlight that in specialty coffee markets, consumption often reflects personal identity and values, making it essential to consider how these non-sensory attributes contribute to consumer satisfaction.

This study draws on these insights to explore how the sensory and non-sensory attributes highlighted in consumer reviews can predict coffee ratings on platforms like CoffeeReview.com. By integrating data science techniques, this research aims to uncover the factors driving consumer preferences in this market and provide actionable insights for businesses seeking to cater to value-driven consumers.

### Online Reviews and Consumer Decision-Making

Online consumer reviews (OCRs) have become integral to decision-making processes across industries, providing valuable feedback that shapes brand perception and consumer behavior. As Luca (2016) observes, OCRs significantly influence consumer purchasing decisions, particularly in sectors where product quality is difficult to assess before purchase, such as specialty coffee. OCRs offer a rich source of qualitative data, capturing sensory and non-sensory evaluations from consumers, which makes platforms like CoffeeReview.com critical for understanding consumer preferences.

In the context of the specialty coffee market, reviews often include detailed descriptions of sensory attributes like aroma, acidity, body, and flavor, as well as non-sensory factors such as origin and processing methods. This information provides valuable insights into consumer perceptions, allowing businesses to identify key trends and better align their offerings with consumer preferences. Furthermore, studies by Chevalier (2006) and Liu (2012) show that review volume and overall rating scores contribute to a product's perceived credibility, highlighting the importance of maintaining a positive online presence.

This research builds on these insights by applying text analytics and sentiment analysis to consumer reviews, aiming to predict coffee ratings based on detailed review content. By analyzing both sensory and non-sensory attributes, this study seeks to develop a model that captures the drivers of consumer satisfaction in the specialty coffee market, offering businesses a data-driven approach to product development and marketing.

### Text Analytics and Sentiment Analysis in Consumer Reviews

With the increasing volume of online reviews, advanced analytical techniques are required to extract meaningful insights from unstructured data. Text analytics and sentiment analysis provide powerful tools for understanding consumer feedback in industries like specialty coffee, where qualitative information is abundant (Pang, 2008). Text analytics enables the identification of key attributes mentioned in reviews, such as flavor notes and brewing methods, which can be quantified and analyzed to reveal consumer preferences.

Sentiment analysis, as Guo (2017) note, allows businesses to gauge the emotional tone of reviews, categorizing them as positive, neutral, or negative. This technique is particularly useful in understanding consumer satisfaction, as it highlights how consumers feel about specific attributes of coffee, such as body or aftertaste. By systematically analyzing consumer feedback, businesses can identify trends and make data-driven decisions.

While text analytics and sentiment analysis are widely used in industries like restaurants and hotels, their application in the specialty coffee market remains underexplored. This research aims to address this gap by applying these techniques to reviews from CoffeeReview.com, uncovering the key factors that drive consumer ratings and satisfaction. By doing so, the study contributes to the broader literature on consumer review analysis and offers practical insights for businesses in the specialty coffee market.

### Determinants of Consumer Ratings in Specialty Coffee

Understanding the drivers behind consumer ratings in the specialty coffee market involves analyzing a combination of sensory and non-sensory factors. Sensory attributes—such as flavor, aroma, body, acidity, and aftertaste—are crucial for influencing consumer satisfaction. Studies by Samoggia (2018) demonstrate that flavor complexity, often described through descriptors like "bright," "chocolatey," or "citrus," is highly correlated with higher consumer ratings. Similarly, Sepulveda (2016) highlights that attributes such as aroma and body contribute significantly to the overall sensory experience.

However, the importance of non-sensory attributes has grown substantially, particularly in the specialty coffee market, where ethical sourcing, certifications, and transparency are becoming significant drivers of consumer preferences (Ponte, 2002). Consumers increasingly value factors like fair trade, organic certifications, and direct trade relationships between producers and roasters. Ufer (2019) note that these non-sensory factors can enhance the perceived quality of a coffee, contributing to higher ratings even when sensory attributes are comparable.

This section directly informs our research by underscoring the need to analyze both sensory and non-sensory factors in predicting coffee ratings. By applying text analytics to consumer reviews on CoffeeReview.com, this study will quantify the impact of these attributes on ratings, providing a holistic view of what influences consumer satisfaction in specialty coffee.

### Methodological Advances in Consumer Review Analysis

The rise of advanced data analytics has transformed how consumer reviews are analyzed, especially in industries reliant on qualitative feedback. Methods such as text analytics, natural language processing (NLP), and machine learning allow researchers to extract structured data from unstructured reviews (Farhadloo, 2016). Text analytics, as applied by Guo (2017), enables businesses to identify key product attributes mentioned in reviews, such as flavor notes or roast levels, and quantify their frequency and importance in driving consumer satisfaction.

Sentiment analysis complements text analytics by categorizing the emotional tone of reviews as positive, neutral, or negative (Pang, 2008). This technique allows researchers to capture the sentiment associated with specific attributes, providing deeper insights into consumer preferences. In the context of specialty coffee, sentiment analysis can highlight how certain sensory or ethical attributes impact consumer perceptions of coffee quality.

Machine learning models such as random forests, support vector machines (SVR), and XGBoost have further enhanced the ability to predict consumer ratings based on review content. Liu (2012) emphasizes that machine learning approaches excel at handling large datasets and complex relationships between variables. Moreover, ensemble methods, which combine multiple models, have shown increased accuracy in predicting outcomes such as consumer ratings.

This study will apply these methods to analyze consumer reviews from CoffeeReview.com, aiming to develop predictive models that accurately forecast coffee ratings based on textual content. By combining text analytics, sentiment analysis, and machine learning, this research seeks to provide actionable insights into the factors driving consumer preferences in the specialty coffee market, contributing to both the academic literature and industry practices.

---

## Methodology

This section outlines the systematic approach used to collect, preprocess, and analyze consumer-generated coffee reviews from CoffeeReview.com. The primary aim is to identify the factors that most influence coffee ratings, using a combination of data science and text analytics techniques. The methodology covers key stages, including data collection, exploratory analysis, text preprocessing, feature extraction, data transformation, feature selection, and predictive modeling.

Given the dataset's complexity—combining numerical ratings and unstructured text—the methodology applies both statistical and machine learning approaches to ensure a robust and reproducible analysis. By integrating exploratory data analysis, advanced text processing (such as TF-IDF and word embeddings), and predictive modeling, this methodology bridges qualitative and quantitative data, offering a comprehensive understanding of the factors driving consumer satisfaction.

### Data Collection

The data collection forms the foundation of this study, providing reliable and comprehensive data for analysis. The primary dataset, sourced from CoffeeReview.com, offers both quantitative ratings and qualitative reviews from coffee enthusiasts and experts, making it ideal for analyzing consumer preferences in the specialty coffee market.

#### Data Source and Access

The dataset was accessed through Kaggle, where a user-curated web scraper generated and shared the data. It includes numerical ratings and text reviews, providing a rich basis for analysis. All data were collected in compliance with CoffeeReview.com and Kaggle's terms of service, adhering to ethical and privacy guidelines. Data processing was carried out using Python.

#### Dataset Description

The dataset consists of thousands of coffee reviews, each including:

- **Rating:** A score between 0 and 100, reflecting the overall product evaluation based on flavor, aroma, body, and aftertaste.
- **Review Text:** Detailed feedback divided into:
  - *Blind Assessment (desc_1):* Describes sensory attributes such as flavor notes (e.g., "chocolatey", "citrus"), aroma and body.
  - *Contextual Information (desc_2):* Includes information such as coffee origin, roast level and processing methods.
  - *Bottom Line (desc_3):* Summarizes the overall impression of the coffee, combining sensory and contextual insights.

**Table: Data Variables and Description**

| Variable Name | Description | Variable Type |
|---------------|-------------|---------------|
| slug | Unique identifier for the coffee review. Typically contains a URL or a link reference to the review page. | Categorical (Unordered) |
| rating | The overall score given to the coffee, typically on a scale from 0 to 100. | Continuous (Bounded 0-100) |
| roaster | The name of the company or entity that roasted the coffee beans. | Categorical (Unordered) |
| name | The name of the specific coffee or coffee blend being reviewed. | Categorical (Unordered) |
| location | The geographical location of the roaster. | Categorical (Unordered) |
| origin | The region or country where the coffee beans were grown. | Categorical (Unordered) |
| roast | The roast level of the coffee beans, such as light, medium, or dark roast. | Categorical (Ordered) |
| est_price | The estimated price of the coffee, usually in the format $XX.XX/unit. | Continuous (Unbounded) |
| review_date | The date the coffee review was published or posted. | Categorical (Ordered by Date) |
| agtron | The Agtron score, a numerical value used to describe the roast level based on color analysis. | Continuous (Bounded) |
| aroma | A score (typically 0-10) that rates the aroma or fragrance of the coffee. | Continuous (Bounded 0-10) |
| acid | A score (typically 0-10) that rates the acidity of the coffee, which affects its brightness and tanginess. | Continuous (Bounded 0-10) |
| body | A score (typically 0-10) that rates the body or mouthfeel of the coffee, describing its weight or texture. | Continuous (Bounded 0-10) |
| flavor | A score (typically 0-10) that rates the overall flavor profile of the coffee, including taste notes. | Continuous (Bounded 0-10) |
| aftertaste | A score (typically 0-10) that rates the lingering flavor or finish after swallowing the coffee. | Continuous (Bounded 0-10) |
| with_milk | A score (typically 0-10) describing how the coffee performs when combined with milk. | Continuous (Bounded 0-10) |
| desc_1 | Blind assessment: Describes the sensory attributes of flavor, aroma, and body without prior knowledge of the origin. | Text |
| desc_2 | Contextual information: Provides background on the coffee's origin, how it was processed, and its general attributes. | Text |
| desc_3 | Bottom line: Summarizes the overall impression of the coffee, often including the reviewer's final thoughts or recommendation. | Text |
| all_text | Concatenated text of desc_1, desc_2 and desc_3 | Text |

**Table: Example of Main Text Variables**

| Statistics | desc_1 | desc_2 | desc_3 |
|------------|--------|--------|--------|
| count | 2774 | 2775 | 2773 |
| unique | 2772 | 2684 | 2772 |
| top | Deeply rich, chocolaty and fruit-toned. Raspberry tart, chocolate fudge, grappa barrel, gardenia, molasses in aroma and cup. | Mystic Monk Coffee is a small-batch roaster associated with a Carmelite monastery located in the Rocky Mountains of northern Wyoming. | A pungent, deep-toned Costa Rica cup processed by the black honey method. Juicy-bright acidity and lush, viscous mouthfeel. |

#### Data Integrity and Quality

Before proceeding with the analysis, the dataset was thoroughly examined to ensure data integrity and quality. The following steps were taken:

- **Missing Values:** The dataset was checked for missing values. Reviews with significant gaps in our columns of interest were flagged and excluded, depending on the extent of missing information.
- **Duplicate Entries:** The dataset was checked for duplicates. Any duplicate entries were removed to maintain the integrity of the dataset.
- **Review Text Consistency:** Reviews with incomplete or incoherent text were inspected and excluded if necessary.

With this data cleaning process, the dataset was well-prepared for subsequent preprocessing and analysis steps.

### Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) was conducted to identify patterns, relationships, and anomalies in the dataset before proceeding with predictive modeling. This stage provided critical insights into the data's structure and informed decisions regarding feature removal, transformation, and encoding.

Descriptive statistics were calculated for numerical variables like the coffee ratings provided by the consumers, to understand central tendencies and dispersion. Visualizations such as histograms, scatter plots, heatmaps, and boxplots were used to examine variable distributions, correlations, and detect potential outliers. These tools helped highlight key patterns and inconsistencies in the dataset.

**Table: Descriptive Statistics of Flavor Variables**

| Statistics | rating | aroma | acid | body | flavor | aftertaste | with_milk |
|------------|--------|-------|------|------|--------|------------|-----------|
| count | 2775 | 2749 | 2395 | 2773 | 2773 | 2773 | 402 |
| mean | 92.97 | 8.81 | 8.50 | 8.60 | 8.94 | 8.09 | 8.84 |
| std | 1.99 | 0.49 | 0.63 | 0.53 | 0.43 | 0.55 | 0.51 |
| min | 63 | 2 | 1 | 5 | 2 | 2 | 5 |
| 25% | 92 | 9 | 9 | 9 | 9 | 8 | 9 |
| 50% | 93 | 9 | 9 | 9 | 9 | 9 | 9 |
| 75% | 94 | 9 | 9 | 9 | 9 | 9 | 9 |
| max | 98 | 10 | 10 | 10 | 10 | 10 | 10 |

The target variable (coffee ratings) exhibited skewness, which was addressed later in the pipeline through a Box-Cox transformation to normalize its distribution. The numerical features exhibited right skewness.

The decision to apply a Box-Cox transformation was driven by the fact that one of the models in the pipeline, linear regression, assumes normally distributed residuals and homoscedasticity. By applying this transformation to the target variable (coffee ratings), the residuals of the linear regression model are more likely to meet these assumptions, which improves the reliability of the model's statistical inference. Specifically, normality of residuals ensures that t-statistics for the estimated coefficients are appropriately distributed, leading to valid confidence intervals and p-values.

However, it is important to note that while the Box-Cox transformation helps satisfy the assumptions of the linear regression model, it may not significantly improve the performance of the non-linear models used in this study. These models do not require the target variable to follow a normal distribution, as they are more robust to skewed distributions (e.g. Random Forest, XGBoost). Despite this, maintaining the Box-Cox transformation across the pipeline ensures consistency and does not negatively affect the performance of these non-linear models.

The exploratory data analysis process refined the dataset, highlighted relevant features, and gave insight into possible variable transformations for later in the pipeline.

### Data Cleaning

The data cleaning process aimed to ensure the dataset was consistent, reliable, and ready for analysis by addressing common data quality issues such as missing values, duplicates, and irrelevant information. Several columns were removed after evaluating their relevance to the analysis. Each variable's removal was based on specific considerations, which are detailed below:

**est_price**: Although price is an important factor in consumer decision-making, this variable was removed due to significant inconsistencies in how it was recorded. The *est_price* variable contained multiple currencies and units of measurement, making standardization a complex task. While converting all prices to a single currency or unit could be an option, this was not feasible within the scope of this study. Additionally, the variable's inconsistent format and limited relevance to the model's objectives led to its exclusion from the analysis.

**review_date**: This variable represented the date the review was collected. Although temporal variables can sometimes reveal trends or seasonal patterns, the *review_date* in this dataset spanned only six months, making it insufficient to capture any meaningful seasonality. Moreover, time-based trends or preferences over time are not the focus of this thesis. While it is possible that certain coffee beans are more popular in specific seasons (e.g., "spring coffee" or "summer coffee"), the limited time span in the dataset constrained the possibility of exploring these trends. Given that the temporal aspect was not central to the research objectives, the *review_date* variable was removed.

**slug**: This column contained the URL from which each review was scraped. Since it had no predictive value in the context of the analysis, the *slug* variable was removed.

**all_text**: Initially, the *all_text* variable, which was a concatenation of *desc_1*, *desc_2*, and *desc_3*, was used in modeling. However, after further consideration, the decision was made to use the three separate description fields instead. These fields—*desc_1* (tasting notes), *desc_2* (contextual information), and *desc_3* (the reviewer's conclusion or "bottom line")—were already present in the dataset as distinct fields. By using them separately, we aimed to capture the different topics each description field focuses on, which allowed the model to better identify specific influences or associations between the content of the reviews and the coffee ratings. This approach provided more meaningful insights compared to using a single concatenated text field.

**agtron**: The *agtron* variable is a numeric measure used to assess the roast level of coffee beans. While roast level is an important factor in coffee quality, the *agtron* measurement is not standardized across different roasters, making it unreliable for consistent comparison. Different roasters may interpret roast levels (such as "light roast") differently. Therefore, rather than relying on this variable, the roast level was captured directly from the textual descriptions provided by the roasters, who classify their coffee according to their own definitions. This approach ensured that roast level information was more representative of the roasters' intended classification.

**location**: The location of the roaster (whether at a city, state, or region level) was not deemed informative for this analysis. Roasters often have multiple locations, which can introduce ambiguity into the data. Additionally, this variable lacked standardization and could refer to various administrative levels, such as city or state. Given this lack of consistency and relevance to the coffee ratings, the *location* variable was excluded from the analysis.

**name**: This variable represented the name of the specific coffee being reviewed. However, this variable was almost entirely unique, with very few repeated values, making it less useful for categorical analysis. Since the name variable was highly granular with an excessive number of unique categories, it was removed as it did not provide meaningful grouping for the predictive models.

**origin**: The *origin* variable was removed because it was replaced by a more relevant and standardized variable, *country_of_origin*. This new feature was created through feature engineering by using a custom function that applied regular expressions (regex) and conditionals to pinpoint the country of origin from the textual data in the reviews. The original *origin* variable was not sufficiently structured for analysis, so this approach provided a more accurate and usable representation of the coffee's origin, making the original *origin* variable unnecessary.

**with_milk**: This variable indicated whether the coffee was reviewed with milk. However, the number of instances where this variable was used was too small to provide any meaningful statistical value. Due to the limited data for this variable, it was excluded from the analysis.

During the data cleaning process, the initial number of rows (representing individual coffee reviews) was recorded, ensuring that any changes to the dataset, such as dropping rows with missing values, were carefully monitored. After the removal of irrelevant variables, a final check for missing values was conducted.

Overall, the data cleaning process ensured that the final dataset was structured, free from irrelevant information, and ready for the subsequent stages of analysis. The removal of these variables was based on a careful evaluation of their relevance, consistency, and potential to contribute meaningful information to the predictive models.

### Text Preprocessing

Text preprocessing was essential for structuring the unstructured review text into a usable format for feature extraction and predictive modeling. This process involved multiple steps to ensure the text was clean, consistent, and semantically accurate.

First, text cleaning was performed to remove irrelevant elements such as HTML tags, URLs, special characters, and digits, leaving only meaningful content. English negations (e.g., "isn't" to "is not") were handled to preserve the intended meaning. Tokenization followed, splitting each review into individual words (tokens) for more detailed analysis.

To maintain uniformity, all tokens were converted to lowercase, ensuring that capitalized and lowercase versions of the same word were treated consistently. Common stopwords (e.g., "and," "the", "in") were removed, though stopwords were retained in a separate pipeline for topic modeling, where context is important. Lemmatization was then applied, reducing words to their base form (e.g., "brewing" to "brew"), which helped reduce data dimensionality by treating variations of the same word as a single entity.

Two distinct preprocessing pipelines were developed based on the requirements of different models. For embedding-based models and sentiment analysis, text was preprocessed to remove stopwords while retaining punctuation. In contrast, for topic modeling techniques, stopwords were retained and punctuation was removed to preserve context within the text. These processing flags were explicitly applied in the code to ensure the appropriate configuration for each task:

- For embeddings and sentiment analysis, punctuation was retained, and stopwords were removed, as these tasks benefit from a cleaner representation of the core content of the text.
- For topic modeling, stopwords were retained and punctuation was removed, as the presence of common words aids in understanding thematic patterns, while punctuation may introduce noise.

After preprocessing, three datasets were created:

- The first dataset was tailored for embedding-based models, where stopwords were removed and punctuation was retained to support TF-IDF and BERT embeddings.
- The second dataset was developed for topic modeling, where stopwords were preserved and punctuation was excluded to enhance the understanding of underlying topics using LDA and NMF techniques.
- The third dataset was used for sentiment analysis, which followed the same preprocessing pipeline as embedding-based models, as both share the same requirements for punctuation and stopwords.

Once text preprocessing was complete, any remaining rows with missing values were removed. A final check was conducted to ensure there were no missing values before saving the preprocessed datasets. The three datasets—one for embeddings, one for topic modeling, and one for sentiment analysis—were saved separately to allow for seamless integration into their respective modeling tasks. This comprehensive preprocessing ensured that the text data were clean, consistent, and ready for their respective modeling tasks.

### Feature Extraction

Feature extraction plays a pivotal role in transforming raw text into structured numerical representations that are suitable for predictive modeling. In this study, a variety of techniques were employed to capture both the syntactic structure and the semantic meaning embedded in the text. These techniques included TF-IDF vectorization, word embeddings using both GloVe and BERT, sentiment analysis, and topic modeling with Latent Dirichlet Allocation (LDA) and Non-Negative Matrix Factorization (NMF). Each method contributed uniquely to building a comprehensive feature set that reflects the richness of the textual data.

#### TF-IDF Vectorization

Term Frequency-Inverse Document Frequency (TF-IDF) is a widely adopted technique in text mining for evaluating the significance of a word within a document relative to a collection of documents, also known as a corpus (Manning, 2008). This method is especially useful for identifying terms that are both frequent and distinctive within certain contexts, making it ideal for analyzing coffee reviews in this study.

In this research, TF-IDF vectorization was applied to the cleaned and preprocessed text from the three description columns: *desc_1*, *desc_2*, and *desc_3*. By capturing unigrams, bigrams, and trigrams (i.e., sequences of one, two, or three words), the vectorizer allowed the identification of both individual words and meaningful word combinations that reviewers frequently use to describe coffee attributes. The vectorizer was configured to limit the feature space to 5000 terms, ensuring computational efficiency while still capturing the richness of the textual data.

For each description column, the TF-IDF vectorizer produced a sparse matrix, quantifying the importance of each term or n-gram in the reviews. These matrices were then transformed into DataFrames, where each column corresponded to a specific n-gram, such as *tfidf_desc_1_0*, representing the first term extracted from the first description column. This method allowed us to capture the linguistic patterns reviewers used when discussing various coffee attributes, including flavor, aroma, and body.

The importance of a word in TF-IDF is calculated by multiplying its term frequency (TF) in the document with the inverse document frequency (IDF), as shown in the following equation:

**TF-IDF(t, d, D) = TF(t, d) × log(|D| / |{d ∈ D : t ∈ d}|)**

Where:
- t represents the term (word or n-gram),
- d is a document within the corpus D,
- |D| is the total number of documents,
- {d ∈ D : t ∈ d} represents the number of documents where the term t appears.

By applying this formula, the model prioritizes terms that are common within individual reviews but rare across the entire set of coffee reviews, helping to highlight unique and meaningful descriptors.

Finally, the extracted TF-IDF features were combined with other feature sets, including word embeddings and topic distributions. This comprehensive feature matrix was used for the subsequent stages of predictive modeling, contributing to a robust representation of the textual data from coffee reviews.

#### Word Embeddings

Word embeddings are essential for capturing semantic relationships and context in text data. Two prominent methods, GloVe and BERT embeddings, were employed in this research to capture word meanings in a dense vector format.

**GloVe Embeddings**

Global Vectors for Word Representation (GloVe) is a widely-used algorithm designed to generate dense vector representations of words by analyzing the statistical co-occurrence of words within a large corpus (Pennington, 2014). Unlike traditional frequency-based methods, such as TF-IDF, which are primarily concerned with the importance of individual terms within documents, GloVe focuses on capturing semantic relationships between words by factoring in the global context in which words appear.

GloVe creates a co-occurrence matrix that counts how frequently pairs of words occur together within a specific context window. By factoring in this co-occurrence, GloVe produces embeddings where words with similar meanings have similar vector representations. This feature makes GloVe especially powerful for detecting nuanced relationships between words. For example, words like "strong" and "powerful" are likely to have similar vectors because they often appear in similar contexts.

In this study, a pre-trained GloVe model with 300-dimensional vectors was used, which provides dense representations for a wide vocabulary of English words. The pre-trained model was chosen for its generalization capability, having been trained on vast amounts of text, which makes it suitable for a broad range of tasks, including the analysis of coffee reviews.

Each review in the dataset was first tokenized, splitting the text into individual words. For each token, the corresponding GloVe vector was retrieved from the pre-trained model. Since each review consists of multiple tokens, a single review-level embedding was created by averaging the vectors of all tokens in the review, yielding a dense vector that represents the overall semantic content of the review. This vector was then included in the feature set, with each of the 300 dimensions represented by separate columns in the DataFrame (e.g., *glove_desc_1_0* for the first dimension of the GloVe vector for *desc_1*).

The formula for generating the GloVe embedding vector for a review is given by:

**v_review = (1/n) Σ(i=1 to n) v_i**

Where:
- v_review is the embedding vector for the entire review,
- n is the number of tokens in the review,
- v_i is the GloVe vector for the i-th token in the review.

GloVe embeddings were chosen for their ability to capture static, context-independent semantic relationships between words. For tasks like coffee review analysis, where terms like "chocolatey" or "acidic" tend to carry similar meanings across different reviews, GloVe provides an efficient and powerful way to encode these words. GloVe's dense word representations enable the model to capture subtle similarities between words, even when they are used in different contexts. Since the coffee review corpus is not massive, GloVe's pre-trained vectors are particularly valuable for transferring knowledge from larger, more generalized corpora.

**BERT Embeddings**

Bidirectional Encoder Representations from Transformers (BERT) (Devlin, 2019) is a more advanced and context-aware embedding method compared to GloVe. While GloVe creates static embeddings, BERT generates dynamic embeddings for each word based on the context provided by surrounding words. BERT leverages a deep transformer-based architecture that reads text bidirectionally, meaning it looks at both the left and right contexts of each word in a sentence, allowing it to capture the subtle nuances of word meaning.

In this research, the DistilBERT model (Sanh, 2019), a lighter and faster variant of BERT, was employed. DistilBERT retains approximately 97% of the language understanding capabilities of BERT while being 60% faster and more computationally efficient, making it well-suited for the task of embedding large volumes of reviews.

To generate BERT embeddings, the entire review text was first tokenized and then fed into the DistilBERT model. DistilBERT outputs embeddings for each token, taking into account the surrounding tokens' context to create dynamic, context-sensitive representations. The embeddings were extracted from the last hidden layer of the model, which contains the most refined representation of the input text.

As with GloVe, the final step was to average the token embeddings across the entire review to produce a single dense vector that represents the review's overall meaning. This review-level embedding was then incorporated into the feature set, with each dimension represented by separate columns in the DataFrame (e.g., *bert_desc_1_0* for the first dimension of the BERT embedding for *desc_1*).

The formula for the BERT embedding is similar to that of GloVe:

**v_review = (1/n) Σ(i=1 to n) BERT_i**

Where:
- v_review is the BERT embedding vector for the entire review,
- n is the number of tokens in the review,
- BERT_i is the BERT embedding for the i-th token, which depends on the context of the surrounding tokens.

BERT embeddings were selected due to their ability to dynamically adjust word representations based on context. This is particularly important in coffee reviews, where words like "bright" or "smooth" may have different meanings depending on the specific coffee attributes being described. BERT's bidirectional understanding allows it to capture these nuanced differences, offering a more refined semantic representation compared to static embeddings like GloVe. This makes BERT embeddings especially valuable for tasks that require an understanding of the specific context in which words are used, which is critical for improving the accuracy of predictions in sentiment-rich domains like coffee reviews.

#### Sentiment Analysis

Sentiment analysis plays a pivotal role in this research, enabling the quantification of the emotional tone embedded within consumer coffee reviews. By extracting the positive or negative sentiment of each review, we aim to better understand how consumer emotions influence coffee ratings. This adds a crucial interpretive layer, allowing us to investigate the emotional drivers behind numerical evaluations.

For this research, we utilized the pre-trained **DistilBERT** model, fine-tuned on the SST-2 dataset, which is widely used for binary sentiment classification. DistilBERT was chosen for its efficient performance and contextual understanding, striking a balance between speed and accuracy (Sanh, 2019). The model provides an output that predicts the probability of a review being classified as either **positive** or **negative** sentiment.

Mathematically, the model outputs logits, which are converted into probabilities using the softmax function:

**P(positive) = e^z_positive / (e^z_positive + e^z_negative)**
**P(negative) = e^z_negative / (e^z_positive + e^z_negative)**

where z_positive and z_negative are the logits for the positive and negative sentiment classes, respectively. The softmax function ensures that the outputs represent probabilities, with P(positive) + P(negative) = 1.

Each review was processed through DistilBERT, generating these probability scores for both positive and negative sentiment. These scores were subsequently incorporated into the dataset as new features:

- P(positive): the probability that the review conveys positive sentiment.
- P(negative): the probability that the review conveys negative sentiment.

These scores reflect not only the dominant sentiment but also the confidence level of the model in classifying each review, adding a nuanced layer to the analysis. This approach was particularly valuable for reviews that may contain mixed sentiments, providing richer interpretive insights into consumer feedback.

The rationale behind using DistilBERT over traditional lexicon-based methods lies in its contextual awareness. Unlike lexicon-based approaches, which treat words like "bitter" or "acidic" as strictly negative, DistilBERT understands the context in which these words appear. For instance, "bitter" could be a desirable attribute in the context of certain coffee flavor profiles. This capacity to adapt sentiment classification to domain-specific language is essential when analyzing consumer reviews in niche industries like coffee.

Furthermore, sentiment scores were integrated into the predictive modeling framework alongside other textual features, such as TF-IDF and word embeddings, to assess their contribution to predicting coffee ratings. By including sentiment as a feature, we quantify the impact of emotional tone on the perceived quality of coffee. The use of sentiment analysis in this manner allows for an econometric analysis of the relationship between consumer emotions and numerical ratings, bringing qualitative insights into the quantitative domain.

In summary, sentiment analysis using DistilBERT enabled a rigorous extraction of emotional signals from coffee reviews, providing valuable insights into how sentiments align with consumer ratings. This aligns with the broader literature on natural language processing, where sentiment analysis has proven to be a powerful tool for understanding subjective opinions and their impact on decision-making (Cambria, 2017).

#### Topic Modeling

Topic modeling is a robust technique employed to reveal latent thematic structures within a corpus of documents. In this study, it was applied to the coffee reviews to identify underlying topics that may influence consumer ratings. Two widely recognized methods—Latent Dirichlet Allocation (LDA) and Non-Negative Matrix Factorization (NMF)—were used to extract key topics from the text data, providing a thematic breakdown of the reviews.

LDA assumes that each document d (in this case, each review) is a mixture of latent topics k, and each topic is characterized by a distribution over words w. The generative process can be described as:

**p(w|α, β) = Π(d=1 to D) [∫ (Π(n=1 to N_d) Σ(k=1 to K) p(w_dn|β_k)p(z_dn = k|θ_d)) p(θ_d|α) dθ_d]**

Where:
- α and β are the hyperparameters for the Dirichlet prior distributions over the document-topic and topic-word distributions, respectively.
- θ_d is the topic distribution for document d.
- z_dn represents the topic assignment for the n-th word in document d.

The model was applied to the TF-IDF-transformed coffee reviews to extract topics such as flavor notes, brewing methods, or even ethical considerations related to coffee. Hyperparameters, including the number of topics and learning decay, were fine-tuned using grid search to achieve optimal performance. The coherence score, which measures the semantic similarity of words within each topic, was used to evaluate the quality of the topics, ensuring they were meaningful and interpretable.

NMF, in contrast, decomposes the document-term matrix V into two lower-dimensional matrices W and H, such that:

**V ≈ W H**

Where:
- V is the document-term matrix (with documents as rows and terms as columns).
- W is the document-topic matrix, indicating the strength of each topic in each document.
- H is the topic-term matrix, representing the distribution of terms within each topic.

NMF ensures that both W and H are non-negative, providing a part-based representation of the data. Hyperparameters such as the number of components and regularization terms were optimized, and the performance of the NMF model was evaluated using the perplexity score to ensure the topics were generalizable.

The topic distributions generated by both LDA and NMF were subsequently integrated into the feature set, offering a structured representation of the thematic content within the coffee reviews. This provided a deeper understanding of the latent themes, allowing the modeling process to capture influential aspects that may significantly impact consumer ratings. By incorporating both LDA and NMF, the topic modeling approach offered a comprehensive analysis of the thematic structure, enriching the dataset with features that reflect the nuances of consumer feedback.

Interpreting the topics involved extracting the top words associated with each topic, ranked by their importance scores in both the LDA and NMF models. For each topic, the top words were visualized to provide an intuitive understanding of the dominant themes. This process enabled an interpretation of how specific topics, such as flavor profiles or production methods, relate to the overall coffee rating. These visualizations, supported by bar plots of word importance, allowed for the practical interpretation of the latent thematic structures, making it easier to connect the topics with consumer preferences.

As suggested by Blei (2003), topic models like LDA are particularly powerful in applications where identifying hidden structures in large textual datasets is crucial for understanding the relationships between text and other variables.

#### Text-Feature Correlation with Flavor Attributes

To quantify the relationship between text-based features and sensory attributes such as acidity, body, aroma, aftertaste, and flavor, a Multinomial Inverse Regression (MNIR) model was used. This approach was implemented following Lasso regression feature selection, which helped identify the most relevant predictors from the high-dimensional text data. The text features included a variety of representations such as TF-IDF vectors, embeddings (BERT and GloVe), and topics derived from LDA.

Given the high-dimensional nature of the text data, feature selection was a necessary step to avoid overfitting and ensure that only the most informative features were retained. A Lasso regression model with cross-validation (cv=5) was used to select features by penalizing less relevant ones, reducing the overall complexity of the model. This step ensured that only the most significant text features were used in the subsequent analysis.

After feature selection, the MNIR model was fitted to predict the five flavor attributes—acidity, body, aroma, aftertaste, and flavor—using the selected text-based features. This regression model allowed for simultaneous predictions of all five sensory attributes based on the selected text features, offering insights into how well the textual descriptions in consumer reviews aligned with the sensory characteristics of the coffee.

The performance of the MNIR model was evaluated using regression metrics such as Mean Squared Error (MSE) and the coefficient of determination (R²) for each flavor attribute. These metrics provided a quantitative assessment of how effectively the text features could predict each sensory attribute. Additionally, SHAP (SHapley Additive exPlanations) values were used to interpret the model's predictions, offering insights into how specific text features contributed to the prediction of each flavor attribute.

This methodology, based on Taddy (2019), provided a robust framework to assess the degree to which consumer-generated text descriptions can explain the sensory profile of a coffee. The combination of Lasso feature selection and MNIR allowed for the analysis of high-dimensional text data while maintaining model interpretability through SHAP analysis.

### Transformation, Splitting, Encoding and Scaling

Following feature extraction, the next step was to integrate all extracted features into a single dataset for predictive modeling. This involved merging the textual features (TF-IDF, embeddings, sentiment scores, and topic distributions) with the numerical and categorical features to capture both qualitative and quantitative aspects of the coffee reviews. Each feature extraction technique produced new columns, which were then consolidated into a single data matrix.

First, all extracted textual features, including those from embeddings, topic modeling (LDA and NMF), and sentiment analysis, were consolidated into a single DataFrame. The combined DataFrame was then merged with the original numerical and categorical variables to create a unified dataset. In this context, the numerical features refer to the flavor features, which represent consumer ratings on attributes such as flavor, aroma, body, and acidity (all rated from 0 to 10).

Initially, a Box-Cox transformation was attempted to normalize the distribution of coffee ratings due to the skewness identified during exploratory analysis. The goal was to reduce skewness and improve normality, which is often beneficial for predictive models, particularly linear regression. However, after testing its impact on model performance, the transformation was ultimately discarded, as it did not improve the models tested. This highlights the variability in how transformations can affect different models.

The flavor features were standardized using *standard scaling*, transforming each feature to have a mean of 0 and a standard deviation of 1. Standardization ensures that all features contribute equally to the models, particularly for algorithms sensitive to the scale of features, such as distance-based methods (e.g., SVR) and gradient-based methods (e.g., XGBoost).

In addition to the flavor features, the text-based features—including those from GloVe, BERT, TF-IDF, topic modeling (LDA, NMF), and sentiment analysis—were also standardized to ensure consistent scaling across all feature types. This step ensured that text features contributed appropriately to the model and avoided any imbalance that could arise from differing feature scales.

For categorical variables such as *roaster*, *country of origin*, and *roast level*, one-hot encoding was applied. This transformation converted categorical values into binary columns, making them suitable for machine learning models.

Once the dataset was fully prepared, it was split into training and testing subsets using a 70/30 train-test split. This division reserved 70% of the data for model training and hyperparameter tuning, with the remaining 30% held out for model evaluation. Stratified sampling was used to ensure the distribution of coffee ratings remained consistent across both subsets, helping to maintain the representativeness of the data in each split.

Finally, the processed training and testing datasets, now containing transformed, encoded, and scaled features, were saved for use in predictive modeling. The fitted parameters from the encoding and scaling processes were also saved to ensure consistency when applying these transformations in future steps.

### Feature Selection Using LASSO

Feature selection is crucial in predictive modeling, especially when dealing with high-dimensional datasets, such as the coffee reviews in this study. LASSO (Least Absolute Shrinkage and Selection Operator) was chosen for feature selection because of its ability to both regularize the model and eliminate irrelevant variables. This helps improve model interpretability, reduce overfitting, and maintain computational efficiency (Tibshirani, 1996).

LASSO allows us to shrink the coefficients of less important features to zero, effectively eliminating them from the model. This is particularly valuable in a dataset that includes various feature types—such as numerical, categorical, and text-derived features (e.g., TF-IDF, embeddings, sentiment analysis, and topic modeling). By doing so, we ensure that only the most relevant predictors are retained, which is critical when dealing with high-dimensional data.

The regularization parameter, λ, controls the degree of shrinkage applied to the coefficients. Optimal values for λ were determined using k-fold cross-validation, ensuring the best balance between model complexity and predictive performance.

In this study, LASSO was applied to different groups of features (e.g., flavor ratings, categorical variables, text features). This structured approach allowed us to independently assess the importance of the variables per each group.

LASSO not only improved interpretability by reducing the feature space but also provided key insights into which variables had the greatest influence on coffee ratings. This streamlined model is less prone to overfitting and computationally more efficient.

### Predictive Modeling

Once feature selection was completed using LASSO, the selected features were used to train several predictive models to forecast coffee ratings. These models integrated numerical, categorical, and text-derived features, capturing both simple and complex relationships within the dataset.

#### Model Selection and Rationale

A combination of linear and non-linear models was employed to address the diverse nature of the data and ensure that both simple linear relationships and more complex non-linear interactions were captured. The models used included:

- **Linear Regression:** A baseline model for comparison, assuming a linear relationship between features and coffee ratings.
- **Random Forest Regressor:** A tree-based ensemble model effective for high-dimensional feature spaces (Breiman, 2001).
- **XGBoost Regressor:** A gradient-boosting model that handles complex interactions and residual errors (Chen, 2016).
- **Support Vector Regressor (SVR):** A model designed to capture non-linear relationships using kernel functions (Smola, 2004).

These models were selected to balance interpretability and predictive power, with further details on their individual configurations provided in the appendix.

#### Model Training and Hyperparameter Tuning

LASSO was applied to the dataset to ensure that only the most relevant features were used for model training, reducing dimensionality and preventing overfitting. Each model was then fine-tuned using a combination of Randomized Search and Grid Search to optimize key hyperparameters. The specifics of this tuning process, including the hyperparameters tested, are available in the appendix.

#### Cross-Validation and Model Selection

To ensure robust model evaluation, 5-fold cross-validation was applied. This method divided the training data into five subsets, using four for training and one for validation in each iteration. Cross-validation helped prevent overfitting and provided a reliable estimate of model performance. After cross-validation, the best-performing model was selected based on validation results, with detailed performance metrics discussed in the Results section.

### Model Evaluation Metrics

After training and hyperparameter tuning, each model was evaluated based on several key metrics to assess their predictive performance. These metrics provided a comprehensive understanding of both the accuracy and reliability of the models when predicting coffee ratings. The metrics used in this study include:

- **Mean Absolute Error (MAE):** MAE measures the average magnitude of the errors between predicted and actual values. It is calculated as the mean of the absolute differences between predicted values (ŷ) and actual values (y):

  **MAE = (1/n) Σ(i=1 to n) |y_i - ŷ_i|**

  MAE is easy to interpret as it is in the same units as the target variable, which in this case are coffee ratings.

- **Root Mean Squared Error (RMSE):** RMSE is the square root of the average of squared errors between predicted and actual values. It is sensitive to outliers and gives more weight to larger errors. RMSE is computed as:

  **RMSE = √[(1/n) Σ(i=1 to n) (y_i - ŷ_i)²]**

  RMSE provides a more interpretable error metric that can highlight the magnitude of the prediction errors, particularly in cases where larger errors need to be penalized more heavily.

- **R-squared (R²):** R-squared measures the proportion of variance in the target variable (coffee ratings) that is explained by the model. It is calculated as:

  **R² = 1 - [Σ(i=1 to n) (y_i - ŷ_i)²] / [Σ(i=1 to n) (y_i - ȳ)²]**

  A higher R-squared value indicates that the model explains a greater proportion of the variance in the coffee ratings. An R-squared value of 1 indicates perfect prediction, while a value of 0 indicates that the model explains none of the variance in the target variable.

The performance of each model was summarized and compared using the aforementioned metrics. This summary provided a comprehensive view of the strengths and weaknesses of each model, based on how well they predicted coffee ratings on the test set.

---

## Results

This section presents the results of the analyses conducted in the study. The focus is on three key areas: the interpretation of topics extracted from the coffee reviews, the performance of the predictive models, and the importance of features in driving coffee ratings.

### Topic Interpretation

The topics extracted using Latent Dirichlet Allocation (LDA), Non-Negative Matrix Factorization (NMF), and Multinomial Inverse Regression (MNIR) shed light on how different descriptors in coffee reviews influence sensory attributes such as acidity, body, aroma, aftertaste, and flavor. The text feature extraction techniques, such as TF-IDF, BERT, GloVe, and LDA, provided critical insights into how language correlates with these sensory ratings. The following summarizes the key insights from these models, with detailed figures and additional information available in the Appendix.

#### Key Findings from LDA

LDA applied to the tasting notes (*desc_1*) highlighted topics such as geographical origin ("costa", "honduras") and sensory descriptors like "mysterious" and "bird," suggesting that consumers value both the provenance of the coffee and the unique narrative elements in their experience.

For contextual information (*desc_2*), LDA revealed a strong focus on origin and production methods (e.g., "kenya", "cup", "explore"), indicating that consumers associate quality with specific regions and the coffee production process.

In the conclusions (*desc_3*), terms like "muted," "nib," and "grumpy" highlighted the importance of balance in flavor, suggesting that a harmonious sensory experience influences higher ratings, while negative descriptors correlate with lower satisfaction.

#### Key Findings from NMF

NMF provided complementary insights, with topics similarly emphasizing origin and flavor diversity. In tasting notes (*desc_1*), key terms like "coffee offer" and "from costa" reiterated the importance of geographical provenance.

Contextual topics (*desc_2*) reinforced the role of clarity in flavors from specific regions (e.g., "new guinea", "grumpy", "muted"), suggesting that consumers value transparency in the production process.

In conclusions (*desc_3*), NMF identified a focus on well-rounded flavor profiles, with terms such as "bittersweet," "chocolate," and "balanced," indicating that richness and harmony in flavors drive positive consumer ratings.

#### Key Findings from MNIR and SHAP Analysis

The application of MNIR to the text features extracted from the coffee reviews provided robust predictive performance across the sensory attributes. The regression metrics, including Mean Squared Error (MSE) and R-squared values, demonstrated a high degree of accuracy, particularly for the attributes of acidity (R² = 0.95) and body (R² = 0.94). While aroma, aftertaste, and flavor exhibited lower R² scores, they still indicated strong predictive power.

The SHAP analysis provided further insights into how specific text features, particularly TF-IDF and BERT embeddings, impacted the model's predictions. For example, text features such as *tfidf_processed_desc_1_641* (caramel gardenia) and *tfidf_processed_desc_1_2452* (lead citrus) were significant for predicting acidity, indicating that sharp or vibrant flavor descriptors were predictive of higher acidity ratings. For body, terms like *tfidf_processed_desc_1_4640* (visit www jbccoffeeroasters) and *tfidf_processed_desc_1_3857* (rocky mountain) suggest that regional or brand-related descriptors were influential in body perception.

The SHAP summary plots revealed that certain TF-IDF features, such as *tfidf_processed_desc_1_641* (caramel gardenia) and BERT embeddings, had the highest SHAP values across different flavor attributes, demonstrating their influence on predicting ratings. The visual impact of these features on each attribute is depicted in the SHAP summary plots for each sensory attribute.

These SHAP values highlight how the textual features extracted from the reviews are being effectively leveraged by the model to make accurate predictions of sensory attributes, with text descriptors influencing both positive and negative directions for each attribute.

#### Correlation Between LDA Topics and Flavor Attributes

The pairwise correlation between LDA topics and sensory attributes shows that certain topics, such as `lda_processed_desc_1_topic_1`, which often involved origin descriptors, are significantly correlated with attributes like acidity and flavor. This further reinforces the idea that the geographical origin and narrative aspects in the reviews are tightly linked with higher consumer ratings.

### Embeddings Interpretation

This section presents the interpretation of the embeddings generated from the GloVe and BERT models, visualized using t-SNE with clustering. The embeddings allow for the identification of patterns within the review text that align with specific coffee characteristics based on flavor, aroma, and other attributes. Each embedding model provides a unique perspective on how coffee reviews are semantically represented.

#### Key Insights from GloVe and BERT Embeddings

The t-SNE visualizations of GloVe and BERT embeddings revealed distinct clusters representing various patterns of coffee descriptions. Below are the main insights from both models:

- **Cluster 0:** Balanced and common coffee descriptions, often characterized by terms like "smooth" and "sweet".
- **Cluster 1:** Richer profiles with robust, intense flavors such as dark chocolate or fruit tones.
- **Cluster 2:** Lighter, delicate flavors, typically described as "floral" or "bright".
- **Cluster 3:** Specialty or experimental coffees with unique notes, including exotic or contrasting flavor profiles.

GloVe embeddings capture more general groupings, while BERT embeddings offer more nuanced representations, distinguishing between standard and unique coffee descriptions. The more complex semantic details captured by BERT make it particularly effective for reviews describing intricate flavor profiles.

### Sentiment Analysis Interpretation

The sentiment analysis conducted on the coffee reviews offers valuable insights into the emotional tone expressed across the three descriptions. Positive and negative sentiment scores were analyzed to reveal patterns of consumer satisfaction.

**Key Insights:**
- **Positive Sentiment:** Description 3 (conclusions) exhibited the highest levels of positive sentiment, suggesting that consumers tend to express satisfaction in their final thoughts on the coffee.
- **Negative Sentiment:** Description 2 (contextual information) showed the highest levels of negative sentiment, indicating that consumers are more critical of the production process or contextual aspects of the coffee.

### Predictive Model Performance

This section evaluates the performance of the predictive models using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R². XGBoost emerged as the best-performing model across both structured and unstructured data, particularly for text-derived features, achieving an R² of 0.683 for all text features and 0.992 for flavor features.

**Table: Detailed Model Performance Metrics**

| Feature Group | Model | MAE | RMSE | R² |
|---------------|-------|-----|------|----|
| **Flavor** | Linear Regression | 0.016 | 0.093 | 0.998 |
| | Decision Tree | 0.038 | 0.184 | 0.991 |
| | Random Forest | 0.025 | 0.130 | 0.995 |
| | XGBoost | 0.038 | 0.165 | 0.992 |
| | SVR | 0.013 | 0.092 | 0.998 |
| **Categorical** | Linear Regression | 1.023 | 1.618 | 0.275 |
| | Decision Tree | 1.109 | 1.692 | 0.208 |
| | Random Forest | 1.045 | 1.516 | 0.364 |
| | XGBoost | 1.027 | 1.593 | 0.298 |
| | SVR | 1.053 | 1.557 | 0.329 |
| **All Text** | Linear Regression | 0.823 | 1.152 | 0.632 |
| | Decision Tree | 0.933 | 1.314 | 0.522 |
| | Random Forest | 0.719 | 1.078 | 0.678 |
| | XGBoost | 0.651 | 1.069 | 0.683 |
| | SVR | 0.734 | 1.247 | 0.569 |

### Feature Importance Results

SHAP (SHapley Additive exPlanations) values were used to assess feature importance across flavor, categorical, and text-derived features. Below are the summarized results.

#### Flavor Features

The most impactful flavor features were **Acid**, followed by **Body** and **Aroma**. SHAP values suggest that higher acidity correlates with better ratings.

#### Categorical Features

Country of Origin and Roast Level were the most significant categorical features. SHAP values indicate that Ethiopian and Kenyan coffees tend to have higher ratings.

#### Text-Based Features

The most impactful text-derived feature was **lda_processed_desc_1_topic_1**. BERT embeddings also contributed significantly to predictions.

### Discussion

The results of this study provide valuable insights into the factors influencing coffee ratings on CoffeeReview.com. By analyzing numerical, categorical, and text-based features, this research addressed the central research question: *What specific factors in consumer-generated reviews on CoffeeReview.com are most strongly associated with higher coffee ratings?*

#### Key Findings from Topic Interpretation and Text Features

The topic interpretation revealed that flavor descriptors and processing methods mentioned in consumer reviews were significantly associated with coffee ratings. The LDA-based topics provided valuable insights into the dominant themes present in the reviews, such as sweetness, acidity, and specific origin descriptors, which were highly correlated with positive ratings. This aligns with the literature on consumer preferences in the specialty coffee market, emphasizing the importance of sensory attributes like acidity and body (Samoggia, 2018).

MNIR was specifically employed to analyze how text features related to predicting sensory attributes like acidity and body. This distinct use of MNIR helped quantify how language describing the coffee—such as terms related to "floral," "citrus," or "caramel"—could predict sensory experiences. However, MNIR was not used in the primary rating prediction models, which combined text, categorical, and sensory attributes to predict overall coffee ratings.

The integration of advanced text features like BERT embeddings and LDA-derived topics showed a strong ability to capture textual nuances and their correlations with sensory attributes, further confirming the effectiveness of natural language processing in extracting meaningful insights from unstructured data. The inclusion of MNIR offered an additional layer of understanding by narrowing the focus to how descriptive language directly correlates with specific sensory experiences.

#### Performance of Predictive Models

The performance of different machine learning models in predicting coffee ratings yielded several key insights. **XGBoost** emerged as the best overall performer across all feature groups, particularly excelling with text-based features. XGBoost's ability to handle high-dimensional data and capture complex, non-linear relationships made it particularly effective for predicting coffee ratings based on both structured and unstructured data. This result aligns with Liu (2012), who noted the robustness of gradient boosting algorithms in handling diverse feature types.

Random Forest, while not as strong as XGBoost in the text feature groups, performed competitively with structured data, particularly flavor-based features. This reflects its strength in handling numerical and categorical data, emphasizing the reliability of ensemble methods for structured data tasks.

The weaker performance of simpler models like Decision Tree and Linear Regression, particularly with text features, was anticipated given the complexity of the dataset. These models struggled to capture the intricate relationships between descriptors and ratings, reflecting the challenges posed by natural language data (Pang, 2008).

#### Importance of Feature Types

The feature importance analysis provided a nuanced understanding of which feature types—numerical, categorical, or text-based—were most predictive of higher coffee ratings. Among the numerical features, acidity emerged as the dominant predictor, confirming findings from Samoggia (2018), who identified acidity as a key attribute driving consumer satisfaction. Other flavor features, such as aftertaste and aroma, also proved significant, reflecting consumer appreciation for complex and well-rounded coffee profiles.

For categorical features, the country of origin and roast level were crucial predictors of coffee ratings. This was consistent with the literature that highlights the premium placed on certain origins, particularly from Panama and Ethiopia, as well as a preference for lighter roasts, which are associated with higher-quality beans (Ufer, 2019). These findings underscore the importance of geographic and roasting profiles in shaping consumer perceptions of specialty coffee.

Text-based features, particularly those derived from BERT embeddings and LDA topics, were among the most influential predictors of coffee ratings. This confirms the value of advanced text analytics in uncovering deeper insights into consumer preferences, as noted by Guo (2017). The ability of BERT embeddings to capture contextual meaning and sentiment in consumer reviews was particularly evident, with features such as specific flavor descriptors and processing methods strongly associated with higher ratings.

#### Unexpected Findings

One of the more surprising findings of this study was the relatively low importance placed on ethical considerations such as sustainability and fair trade. While previous research (Sepulveda, 2016) has emphasized the role of ethical practices in shaping consumer preferences, this study found that sensory and narrative elements, particularly related to origin and flavor, played a more significant role in driving higher ratings. This suggests that CoffeeReview.com's audience may be more focused on the sensory experience and the story behind the coffee rather than on ethical factors, at least in the context of their ratings.

Another unexpected result was the relatively close performance of Random Forest compared to XGBoost in some cases. While XGBoost was expected to outperform the other models, a more dominant performance with a clear distinction was anticipated. However, Random Forest's ability to approach XGBoost's results, particularly with structured data, was unexpected. This suggests that ensemble methods like Random Forest can perform competitively, even in tasks where XGBoost is typically expected to excel, especially when it comes to structured features such as flavor attributes. The surprising finding here was not XGBoost's strong performance, but rather the narrower margin between XGBoost and Random Forest, particularly in handling structured data.

---

## Conclusions

This study investigated the factors influencing coffee ratings on CoffeeReview.com, with the aim of uncovering key sensory and non-sensory attributes that drive consumer preferences. By integrating text analytics, sentiment analysis, Multinomial Inverse Regression (MNIR), and machine learning, the research provided valuable insights for both academia and the specialty coffee market.

### Summary of Key Findings

The study identified several key findings:

- **Flavor attributes**, particularly *acidity*, were the strongest predictors of higher coffee ratings, emphasizing the central role of sensory experiences in shaping consumer satisfaction.

- **Text-based features** from BERT embeddings, GloVe vectors, and LDA topics were among the most influential predictors, demonstrating the effectiveness of advanced text analytics in capturing nuanced consumer feedback and preferences. Additionally, MNIR offered strong predictive performance when focusing on sensory attributes, reinforcing the importance of textual descriptors in predicting coffee characteristics.

- **Categorical features**, such as *country of origin* and *roast level*, also significantly shaped consumer perceptions, with origins like Panama and Ethiopia consistently linked to higher ratings, and lighter roasts being viewed more favorably.

- **XGBoost** outperformed other models in the prediction of coffee ratings, particularly in handling both structured and unstructured data, underscoring its flexibility and strength in capturing complex relationships across diverse feature types.

### Contributions to Research and Practice

This study makes several contributions to both the academic field and the specialty coffee industry. By integrating text analytics, MNIR, and machine learning, the research provides a comprehensive framework that captures both sensory and non-sensory attributes from consumer reviews, shedding light on the drivers of consumer satisfaction. The findings offer actionable insights for businesses, enabling them to better align their products and marketing strategies with consumer expectations. The use of LASSO for feature selection, MNIR for text-based sensory predictions, and SHAP for model interpretability enhances the transparency of the analysis, allowing industry stakeholders to understand the most influential factors shaping consumer preferences.

### Implications for the Specialty Coffee Market

The findings of this study have practical implications for the specialty coffee market. Producers and marketers should focus on emphasizing key flavor attributes such as acidity and aftertaste in product descriptions, as these qualities are closely linked to positive consumer ratings in the specialty coffee market. Additionally, origin and narrative elements, such as stories behind sourcing and production, should be highlighted, as these non-sensory attributes are increasingly important to consumers and are linked to higher ratings. By leveraging text analytics and models like MNIR, businesses can gain deeper insights into consumer sentiment and preferences, enabling them to craft marketing strategies that resonate with evolving consumer demands, particularly in areas like provenance and storytelling.

### Limitations and Future Research Directions

While this study provides valuable insights, several limitations should be acknowledged. The dataset is limited to reviews from CoffeeReview.com, which may not fully represent global consumer preferences. In particular, the platform may attract a consumer segment that prioritizes the quality and sensory attributes of coffee, potentially overlooking other factors like price or convenience. Future research could broaden the scope by including data from other platforms or social media to capture a more diverse range of opinions and consumer types.

Further hyperparameter tuning of models like XGBoost could enhance predictive accuracy, and the exploration of hybrid models or deep learning techniques may improve the prediction of coffee ratings in future studies. Additionally, expanding MNIR's application beyond sensory attributes to encompass categorical features could reveal new insights.

Lastly, the methods and findings from this study could be applied to other artisanal markets, such as wine or craft beer, where sensory experiences and sourcing considerations play a significant role in consumer decisions. Both markets share similarities with the specialty coffee industry, including a focus on quality, provenance, and narrative elements.

### Final Remarks

This study has expanded the understanding of consumer preferences in the specialty coffee market, offering valuable insights into the factors that drive coffee ratings. The use of text analytics, sentiment analysis, MNIR, and machine learning presents a novel approach to analyzing consumer behavior, providing both academic contributions and practical strategies for businesses in competitive markets. The findings highlight the growing importance of sensory attributes, narrative elements, and advanced data-driven methods for gaining actionable insights into consumer preferences.

---

## References

*[Note: In the original LaTeX document, this would be populated by the bibliography file. For this markdown version, I'm noting that references would be formatted according to APA style as indicated in the original document.]*

---

## Appendix

### LASSO Regression Details

#### LASSO Regression Objective Function

LASSO is a linear regression technique that extends ordinary least squares by adding a penalty (regularization term) to the sum of the absolute values of the model coefficients. The LASSO objective function can be expressed as follows:

**min_{β₀, β} [Σ(i=1 to n) (y_i - β₀ - Σ(j=1 to p) β_j x_{ij})² + λ Σ(j=1 to p) |β_j|]**

Where:
- y_i is the target variable (coffee ratings).
- x_ij are the features (e.g., text-derived features, embeddings, sentiment scores).
- β_j are the coefficients associated with each feature.
- λ is the regularization parameter that controls the degree of shrinkage.

#### Regularization and the Role of λ

The regularization parameter λ determines the amount of shrinkage applied to the model coefficients. When λ is zero, the model is equivalent to ordinary least squares, and all features are included. As λ increases, more coefficients are shrunk towards zero, and the model becomes sparser.

The optimal λ was chosen using k-fold cross-validation, ensuring a trade-off between model complexity and predictive performance.

#### Application of LASSO to Coffee Reviews

LASSO was applied independently to different groups of features, including flavor ratings, categorical variables, and text-derived features (e.g., TF-IDF vectors, GloVe embeddings, BERT embeddings, sentiment scores, and topic distributions). This allowed the model to focus on the most relevant variables within each group.

#### LASSO Feature Selection Process

The key steps in the LASSO application included:
- **Data Standardization:** All features were standardized to have a mean of zero and a standard deviation of one, ensuring uniform application of the regularization penalty across features.
- **Model Fitting:** The LASSO model was fitted to each group of features, and λ was tuned using cross-validation.
- **Feature Elimination:** Features with coefficients shrunk to zero were eliminated, simplifying the model.

### Predictive Model Methodology Details

#### Predictive Models

The following section provides in-depth details on the models used for predictive analysis, including their configurations and specific hyperparameters optimized during training.

**Linear Regression**

Linear regression was used as a baseline model, assuming a simple linear relationship between features and coffee ratings. LASSO was employed for feature selection to reduce overfitting, but no further regularization was applied beyond the LASSO penalty.

**Random Forest Regressor**

The Random Forest Regressor builds multiple decision trees and aggregates their predictions to make the final forecast. The following hyperparameters were fine-tuned:
- Number of trees in the forest
- Maximum depth of the trees
- Minimum samples required for a split

This model's strength lies in handling large feature spaces and being less prone to overfitting due to its ensemble nature (Breiman, 2001).

**XGBoost Regressor**

XGBoost is a powerful gradient-boosting model that incrementally improves predictions by focusing on the residuals of prior iterations. Key hyperparameters optimized include:
- Number of boosting rounds
- Learning rate
- Maximum depth of trees
- Regularization terms (L1, L2 penalties)

This model is highly efficient for datasets with complex interactions and was optimized for speed and accuracy (Chen, 2016).

**Support Vector Regressor (SVR)**

SVR works by constructing hyperplanes in high-dimensional space to separate data points. The primary hyperparameters tuned for SVR were:
- Regularization parameter C
- Kernel function (linear or radial basis function)
- Gamma value for RBF kernel

SVR is particularly effective in capturing non-linear patterns and relationships in the dataset (Smola, 2004).

#### Hyperparameter Tuning and Cross-Validation

**Hyperparameter Tuning**: A two-step process was employed to tune hyperparameters:
- **Randomized Search:** Explored a wide range of potential hyperparameters to identify a promising region in the hyperparameter space.
- **Grid Search:** Refined the hyperparameters based on the most promising configurations from the Randomized Search.

**Cross-Validation**: For each model, 5-fold cross-validation was used to ensure that the model was robust and generalizable to new data. This involved splitting the dataset into five subsets, where four subsets were used for training and one for validation. This process was repeated five times, and the average performance across the five iterations was used to select the optimal model configuration.

*[Additional appendix sections would include topic interpretation visualizations, sentiment analysis plots, and SHAP dependence plots as referenced in the original document]* 