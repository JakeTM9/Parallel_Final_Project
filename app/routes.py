from flask import render_template, request, url_for, redirect
from app import app, kernel
import sys
# use decorators to link the function to a url
@app.route('/')
def home():
    if request.method == 'POST':
        numThreads = request.form["numThreads"]
        playerHand = request.form["playerHand"]
        dealerHand = request.form["dealerHand"]
        return redirect(url_for('result'), numThreads = numThreads, playerHand = playerHand, dealerHand = dealerHand)
    return render_template('index.html')

@app.route("/result", methods=['GET', 'POST'])
def result():
    numThreads = request.form.get('numThreads')
    playerHand = request.form.get('playerHand')
    dealerHand = request.form.get('dealerHand')
    
    array = kernel.super_simple_example_runner(int(numThreads))
    player_hand_cards, player_hand_values, dealer_hand_cards, dealer_hand_values, playerTotal, dealerTotal = kernel.formatInputForBlackJack(playerHand,dealerHand)

    return render_template("result.html",
    numThreads = numThreads, array = array ,player_hand_cards = player_hand_cards,
    player_hand_values = player_hand_values, dealer_hand_cards = dealer_hand_cards,
    dealer_hand_values = dealer_hand_values, playerTotal = playerTotal, dealerTotal = dealerTotal)