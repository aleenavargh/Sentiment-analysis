# Sentiment Analysis and Modeling for Amazon Reviews

## Project Overview

This project focuses on sentiment analysis of Amazon reviews using advanced natural language processing (NLP) techniques and machine learning models, including deep learning models like BERT (Bidirectional Encoder Representations from Transformers) and LSTM (Long Short-Term Memory). The primary objective is to classify the sentiment of product reviews as positive or negative based on their textual content.

## Project Structure

- **Text Preprocessing**: Steps to clean and prepare the text data for analysis, including:
  - Normalizing: Converting text to lowercase.
  - Removing punctuations, numbers, stopwords, and rare words.
  - Tokenization and lemmatization of text.

- **Text Visualization**: Visual representation of the text data to understand the frequency of terms:
  - Term frequency calculations.
  - Bar plots.
  - Wordclouds, including word clouds by specific templates.

- **Sentiment Analysis**: Techniques to analyze the sentiment of the text data and label it accordingly.

- **Feature Engineering**: Methods to transform the text into a format suitable for machine learning models:
  - Count Vectors: Representing text as a vector of word counts.
  - TF-IDF: Term Frequency-Inverse Document Frequency representation.

- **Sentiment Modeling**: Building machine learning and deep learning models to predict the sentiment of reviews:
  - Logistic Regression.
  - Random Forests.
  - **LSTM (Long Short-Term Memory)**: A deep learning model that captures the sequential dependencies in text data, particularly effective for understanding the context in sequences like sentences and paragraphs.
  - **BERT (Bidirectional Encoder Representations from Transformers)**: A state-of-the-art deep learning model that provides contextualized word representations by considering both left and right contexts in all layers, significantly improving the performance of sentiment classification.
  - Hyperparameter optimization for improving model performance.

## Deep Learning Models

### LSTM (Long Short-Term Memory)
LSTM is a type of Recurrent Neural Network (RNN) that is particularly well-suited for tasks involving sequential data, such as text. In this project, LSTM was used to model the sentiment of Amazon reviews by capturing the context and dependencies between words in a sentence. The model's ability to retain information over long sequences makes it an effective choice for sentiment analysis.

**Results with LSTM**:
- The model achieved high accuracy in classifying the sentiment of reviews, with consistent performance across training and validation datasets.

### BERT (Bidirectional Encoder Representations from Transformers)
BERT is a transformer-based model that has revolutionized NLP tasks by providing deep bidirectional context representations. Unlike traditional models that consider only the left or right context, BERT looks at both sides simultaneously, leading to a better understanding of the text's meaning. In this project, BERT was fine-tuned on the Amazon reviews dataset to classify sentiments with exceptional accuracy.

**Results with BERT**:
- BERT outperformed traditional machine learning models by leveraging its ability to understand context at a deeper level, leading to improved accuracy and generalization on unseen data.

## Dataset

The project utilizes Amazon reviews, specifically the `reviewText` column, to analyze and model sentiment. The dataset is pre-processed to handle missing data, and text is cleaned and transformed into suitable features for modeling.

## Installation and Requirements

To run the project, you need the following Python libraries:

```bash
pip install pandas numpy scikit-learn nltk wordcloud matplotlib seaborn tensorflow transformers
```

Additional steps:
- Ensure you have Jupyter Notebook or Google Colab to execute the notebook.

## Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your_username/amazon_sentiment_analysis.git
   cd amazon_sentiment_analysis
   ```

2. **Run the Notebook**:
   Open `Group6_sentiment_analysis_and_modeling_for_amazon.ipynb` in Jupyter Notebook or Google Colab and run the cells to reproduce the results.

3. **Explore the Results**:
   The notebook provides detailed outputs, including visualizations, model performance metrics, and deep learning model predictions.

## Results

The project includes results from multiple models:
- **Logistic Regression**: A simple and interpretable model for sentiment classification.
- **Random Forests**: An ensemble model that improves accuracy by reducing overfitting.
- **LSTM**: A deep learning model that captures sequential dependencies, achieving high accuracy on sentiment classification.
- **BERT**: A state-of-the-art model that leverages deep contextual understanding to outperform other models in sentiment prediction.
- **Hyperparameter Optimization**: Fine-tuning model parameters for better performance.

## Conclusion

This project demonstrates a comprehensive approach to sentiment analysis using NLP, machine learning, and deep learning techniques. The detailed analysis, from text preprocessing to model evaluation, provides valuable insights into the sentiment of Amazon product reviews.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This project was completed as part of a group effort, with contributions from Meghann Sandhu, Aleena Varghese, and Priyadarshini Venkatesh.
