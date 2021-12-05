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
    games_per_thread_str = request.form.get('gamesPerThread')
    player_hand_str = request.form.get('playerHand')
    dealer_hand_str = request.form.get('dealerHand')


    num_threads = int(num_threads_str)
    games_per_thread = int(games_per_thread_str)
    total_sims = num_threads * games_per_thread

    # reformat output strings with commas
    total_sims_str = f"{total_sims:,d}"
    num_threads_str = f"{num_threads:,d}"
    games_per_thread_str = f"{games_per_thread:,d}"

    standing_winrate, hitting_winrate, time_taken_str = kernel.core_handler(
        num_threads, games_per_thread, player_hand_str, dealer_hand_str)
    player_hand_cards, player_hand_values, dealer_hand_cards, dealer_hand_values, player_total, dealer_total = kernel.formatInputForBlackJack(
        player_hand_str, dealer_hand_str)

    if time_taken_str == "N/A": # no sims run
        recomendation_str = "N/A"
    elif standing_winrate > hitting_winrate:
        recomendation_str = "Stand."
    else:
        recomendation_str = "Hit!"

    return render_template("result.html",
                           totalSims=total_sims_str,
                           numThreads=num_threads_str,
                           gamesPerThread=games_per_thread_str,
                           standingWinRatio=standing_winrate,
                           hittingWinRatio=hitting_winrate,
                           recomendationStr=recomendation_str,
                           player_hand_cards=player_hand_cards,
                           player_hand_values=player_hand_values,
                           dealer_hand_cards=dealer_hand_cards,
                           dealer_hand_values=dealer_hand_values,
                           playerTotal=player_total,
                           dealerTotal=dealer_total,
                           timeTaken=time_taken_str,
                           )
