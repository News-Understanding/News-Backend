# News understanding Backend 

## Project Overview

The backend of our NLP News Understanding Project is built using Flask to deploy five Language Models (LLMs). We have four models dedicated to classification tasks (Bias, Fake vs Real, Sentiment, Topics) and one model for summarization. This README provides an overview of the backend structure, API functions, and how to interact with the deployed models.

## Models

1. **Fake vs Real Classification Model**: Fine-tuned T5-small using datasets from Hugging Face and Kaggle. Achieved accuracy ranging from 87% to 97%.

2. **Bias Classification Model**: Fine-tuned DistilBERT using data from AllSides News and MediaBiasFactCheck. Accuracy scores are 75% and 94% on separate datasets.

3. **Sentiment Analysis Model**: Utilizes Gemini Pro and GPT-4 for sentiment labeling. Fine-tuned on various models, with RoBERTa Large providing the highest accuracy.

4. **Topics Classification Model**: Fine-tuned DistilBERT using data from Kaggle and Hugging Face. K-train is employed for rapid fine-tuning, with DistilBERT achieving the highest accuracy.

5. **Summarization Model**: Pegasus, BART Large, T5-Small, and T5-Base are used for summarization. BART and T5-Base outperform other models in terms of word count and overall information extraction.

## API Functions

### 1. Classification API (Bias, Fake, Sentiment, Topics)

#### Endpoint: `/classify`
- **Method**: POST
- **Input**: JSON object with the news article text.
- **Output**: List of classifications, including Bias, Fake vs Real, Sentiment, and Topics (e.g., [Fake - L-Biased - Negative - Tech]).

### 2. Summarization API

#### Endpoint: `/summarize`
- **Method**: POST
- **Input**: JSON object with the news article text.
- **Output**: Summarized version of the input article.


## Getting the Models

To obtain the models, run the notebooks provided in the NLP repository [NLP repo](https://github.com/AnasElbattra/News-Understanding-NLP?tab=readme-ov-file#nlp-news-understanding-project). Follow the steps outlined in the notebooks to download and fine-tune the models.

