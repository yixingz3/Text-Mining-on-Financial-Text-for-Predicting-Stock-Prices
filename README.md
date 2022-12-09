# Text Mining on Financial Text for Predicting Stock Prices

## Course Project for CS410 FAll 2022

### Yixing Zheng (yixingz3@illinois.edu) - team captain
### Zerui Tian (zeruit2@illinois.edu)

#### [Project proposal](https://github.com/yixingz3/Text-Mining-on-Financial-Text-for-Predicting-Stock-Prices/blob/main/CS%20410%20Project%20Proposal.pdf)

#### [Status Report](https://github.com/yixingz3/Text-Mining-on-Financial-Text-for-Predicting-Stock-Prices/blob/main/CS%20410%20Project%20Status%20Report.pdf)

## Final Submission Documentation

### Overview of the project

#### End goal and background
As initially proposed, we hope to apply text mining technology to the financial
field by mining financial texts with Machine Learning and Deep Learning models. 
As result, we want to track down the best model that can help us predict stock prices (or at least provide some insights in that regard) and validate if it's by any means helpful in the real world.

This is interesting to us for two main reasons. One reason is that
there are not many applications of text mining in financial markets,
and many of them are related to sentiment analysis, and do not directly
link text data and stock price data. And the other reason is that we would like to explore the utilization of machine
learning and deep learning algorithms in natural language processing and
extend, or at least validate, that machine learning/deep learning is
able to improve the performance for some NLP tasks.

#### Project process
For the process, we have followed the following steps:
1. Locate proper source of data and label the records 0 and 1, indicating stock price falling and raising
2. Browser candidate Machine Learning/Deep Learning models and decide the ones we want to try out
3. Select models (winners are Vectorize + ML Model, LSTM, and BERT) and build them out
4. Train and fine-tune all 3 models with the same labeled dataset
5. Evaluate and compare the results

#### Model implementation
- Vectorize + ML Model
    - We picked this model as a reference model that is less advanced so we can compare if indeed the newer/more advanced models provide better results
    - We have tried different combinations for the Vectorize + ML Model. 
    - Count-Vectorizer and TFIDF-Vectorizer for the vectorize model
    - Logistic regression, Naive Bayes, and Random Forest for the ML model
- LSTM
    - We just implemented the standard LSTM model with our labeled dataset
    - The main reason we picked this model is that the model works well with connections between arbitrary long-term dependencies in the input, thus is suitable for modeling time series data that we are providing and hoping to obtain insights from
- BERT
    - BERT being a state-of-the-art NLP model that learns contextual relations between words shows a lot of potentials to outperform other models in this project, so we picked it as a seed candidate
    - implementation-wise, it's standard BERT with fine-tuning to provide the best result it can for our dataset

#### Results

| Metrics/Models | CountVectorizer + LogisticRegression | TfidfVectorizer + Naive Bayes | VCountVectorizer + RandomForest | DL - LSTM | BERT    |
|:--------------:|:------------------------------------:|:-----------------------------:|:------------------------------:|:---------:|:-------:|
| F1             | 0.69                                 | 0.72                          | 0.65    | N/A            | N/A |
| Accuracy       | 0.59                                 | 0.59                          | 0.57    | 0.5690         | 0.593 |

### How to run it?

#### [The demo presentation is shared in Google Drive and needs Illinois credential to access](https://drive.google.com/file/d/11xNdViTfy1zxQTRXV_NHI8cqkWWyshbY/view?usp=share_link) or it can be downloaded from this [repo](https://github.com/yixingz3/Text-Mining-on-Financial-Text-for-Predicting-Stock-Prices/blob/main/CS410_Project_demo.mp4)
1. Download/copy these files `TIS_Project.ipynb` ([code](https://github.com/yixingz3/Text-Mining-on-Financial-Text-for-Predicting-Stock-Prices/blob/main/TIS_Project.ipynb)), `410_dataset.csv (dataset)` along with `lstm_model.pth and bert_model.pth (trained models)` from [our Google Drive](https://drive.google.com/drive/folders/1su2LbMR0FD3CYRQrWV0wl9af0VUp95OZ?usp=sharing) - you can download the folder and upload all of them into your Illinois Google Drive
    - make sure they are placed in the same directory, ideally ```/content/drive/MyDrive/CS410Project```
2. Open TIS_Project.ipynb from Google Drive using [Google Colab](https://colab.research.google.com/)
    - This should already be available with Illinois Google Account
3. Run through the steps of each code block to build the model
    - Make sure the runtime type is set to `GPU` in the notebook settings 
    - Notice in the data mounting section, if a different dataset might be used
        - The dataset needs to be uploaded to the same directory
        - The directory in the code needs to be updated

- Potential blockers?
    - It should be a straightforward and painless process to run the application since the code, dataset, and environment configs are managed by Google Drive and Google Colab. On the top of the file, all the packages/libs are installed and imported.
    - So, as long as the dataset (current or new) is uploaded and mounted correctly, without any code/dependency lib changes, the app should run as expected without any issues.

### Team member contributions
- As a team, we worked together on the initial brainstorming, models and dataset picking, and building and training the models.
- Individually:
    - Zerui Tian - Building and fine-tuning the models
    - Yixing Zheng - Preparing dataset and assisting with the model fine-tuning and training process (along with team captain duties)
    
### Future work
- Validate the trained model with current/recent text data
- Try out new models/fine-tune even more to improve model performance in terms of accuracy

### Reference and Libraries used

- We refer to this [paper](https://drive.google.com/file/d/1hZunGLg940XeJw_MP2ir4551WwRZZZ66/view?usp=sharing) and build our work on top of it (need Illinois credential to access)
- Main libraries used in the project:
    - pytorch
    - scikit-learn
    - numpy
    - pandas
