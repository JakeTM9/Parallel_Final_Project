
# Python Flask App Positive/Negative Sentiment Analysis

This is a minimal Flask app that has the user input a user review and hit the submit button. The user is then redirected to a page that displays the review as well as the prediction.

The app has a saved tensorflow model to make the prediction. The model uses [Twitter Airline Review Data](https://www.kaggle.com/crowdflower/twitter-airline-sentiment) to train.

I followed [this guide](https://techvidvan.com/tutorials/python-sentiment-analysis/) to create the sentiment analysis model and [this guide](https://www.tensorflow.org/tutorials/keras/save_and_load) to save the model

# To Re-Train the Model

If you want to train the model using more epochs or a different way. Trainer.py is included and can be run. Be sure to delete "sentiment_analysis_model.h5" from the app directory in order to save the new model.

# Instructions

The app is easy to run.
Requirements: Python 3 and pip 
1. Download the code
2. Navigate to the blackjackApp directory in CMD
3. run "py -3 -m venv .venv"
4. run ".venv\scripts\activate"
5. run "pip install -r requirements.txt"
6. run "flask run"
7. Navigate to the address output in the console

