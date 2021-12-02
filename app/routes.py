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
        return redirect(url_for('result'), numThreads=numThreads, playerHand=playerHand, dealerHand=dealerHand)
    return render_template('index.html')


@app.route("/result", methods=['GET', 'POST'])
def result():
    num_threads_str = request.form.get('numThreads')
    player_hand_str = request.form.get('playerHand')
    dealer_hand_str = request.form.get('dealerHand')

    num_threads = int(num_threads_str)

    array = kernel.super_simple_example_runner(
        num_threads, player_hand_str, dealer_hand_str)
    player_hand_cards, player_hand_values, dealer_hand_cards, dealer_hand_values, player_total, dealer_total = kernel.formatInputForBlackJack(
        player_hand_str, dealer_hand_str)

    return render_template("result.html",
                           numThreads=num_threads_str,
                           array=array,
                           player_hand_cards=player_hand_cards,
                           player_hand_values=player_hand_values,
                           dealer_hand_cards=dealer_hand_cards,
                           dealer_hand_values=dealer_hand_values,
                           playerTotal=player_total,
                           dealerTotal=dealer_total,
                           )
