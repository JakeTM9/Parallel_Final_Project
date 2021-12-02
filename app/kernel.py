import sys
import numba 
import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

@cuda.jit
def blackjack_kernel(wins_standing_array,
                     wins_hitting_array,
                     remaining_deck_array,
                     player_hand_array,
                     dealer_hand_array,
                     simulations_to_run,
                     rng_states):

    # TODO: something about gamestate data input
    thread_position = cuda.threadIdx.x # probably make this a different call
    wins_standing_count = 0
    wins_hitting_count = 0

    for _ in range(simulations_to_run):
        c = numba.cuda.local.array(10, numba.uint8)
        # this section will be the "core", real game logic will go here

        # begin basic example:
        # get random number from 0 to 1
        random_float_0_to_1 = xoroshiro128p_uniform_float32(rng_states,
                                                            thread_position)
        if random_float_0_to_1 > .5:
            wins_standing_count += 1

        random_float_0_to_1 = xoroshiro128p_uniform_float32(rng_states,
                                                            thread_position)
        if random_float_0_to_1 > .5:
            wins_hitting_count += 1
        # end basic example:

    wins_standing_array[thread_position] = wins_standing_count
    wins_hitting_array[thread_position] = wins_hitting_count


def get_card_values_from_hand_str(hand):
    card_str_list = hand.split(",")
    value_list = []
    for card in card_str_list:
        value_list.append(card_str_to_value(card))
    return card_str_list, value_list

def card_str_to_value (card):
    """ Takes a human-readable card string and gets the underlying card value.
        Ace is 11, Face cards are 10.
    """
    if "a" in card:
        return 11
    elif "2" in card:
        return 2
    elif "3" in card:
        return 3
    elif "4" in card:
        return 4
    elif "5" in card:
        return 5
    elif "6" in card:
        return 6
    elif "7" in card:
        return 7
    elif "8" in card:
        return 8
    elif "9" in card:
        return 9
    elif "10" in card:
        return 10
    elif "j" in card:
        return 10
    elif "q" in card:
        return 10
    elif "k" in card:
        return 10

def get_total_from_value_list(value_list):
    total = 0
    numAces = 0
    for value in value_list:
        total += value
        if value == 11:
            numAces += 1
    while total > 21 and numAces > 0:
        total -= 10
        numAces -= 1
    return total

## Called by Front end for
def formatInputForBlackJack (player_hand_string, dealer_hand_string): 
    player_hand_cards_list, player_hand_values_list = get_card_values_from_hand_str(player_hand_string)
    dealer_hand_cards_list, dealer_hand_values_list = get_card_values_from_hand_str(dealer_hand_string)
    playerTotal = get_total_from_value_list(player_hand_values_list)
    dealerTotal = get_total_from_value_list(dealer_hand_values_list)
    return player_hand_cards_list, player_hand_values_list, dealer_hand_cards_list, dealer_hand_values_list, playerTotal, dealerTotal

## Checks if Player has busted or has 21
def is_game_over(playerTotal):
    return playerTotal >= 21

def get_full_deck():
    """ returns an np array of all blackjack values in a deck, with aces as 11s """
    deck_value_list = list()

    # 2-11, includes one set of "10s" (the actual 10 cards) and one set of "11s" (aces)
    for card_value in range(2, 12):
        for _ in range(4):
            deck_value_list.append(card_value)
    
    # the other face cards, of which there are 12 (4 of each)
    for _ in range(12):
        deck_value_list.append(10)

    deck = np.array(deck_value_list)
    return deck

## Returns Deck with 0s where no card size 52
def initializeDeck(player_hand_values, dealer_hand_values): 
    deck = get_full_deck()
    for value in player_hand_values:
        deck[np.where(deck==value)[0][0]] = 0
    for value in dealer_hand_values:
        deck[np.where(deck==value)[0][0]] = 0
    return deck

## Returns Hand with 0s where no card size 12 (not the best implementation but I was spending too much time on this)
def normalizeHand(hand):
    hand = np.array(hand)
    zerosToAdd = 12 - hand.size
    for i in range(0,zerosToAdd):
        hand = np.insert(hand,hand.size, 0)
    return hand

def format_input_for_kernel(playerHand, dealerHand):
    player_hand_cards, player_hand_values, dealer_hand_cards, dealer_hand_values, playerTotal, dealerTotal = formatInputForBlackJack (playerHand, dealerHand)

    ## Checks if Player has busted or has 21
    game_over = is_game_over(playerTotal)

    ## Deck values without player values or dealer known values
    deck_without_hand_values = initializeDeck(player_hand_values, dealer_hand_values)

    ##player and dealer have a fixed array size of 12, cards are non-zero (unsure how handling dealer's uknown card so for now he is treated like a player)
    playerHandNormalized = normalizeHand(player_hand_values)
    dealerHandNormalized = normalizeHand(dealer_hand_values)

    ## printing these in console so you can see (hit submit on input)
    print(deck_without_hand_values, file=sys.stderr)
    print(playerHandNormalized, file=sys.stderr)
    print(dealerHandNormalized, file=sys.stderr)

    # hitOnFirst = not sure if call on kernel launch
    return game_over, deck_without_hand_values, playerHandNormalized, dealerHandNormalized

def core_handler(num_threads_to_run, games_per_thread, player_hand_str, dealer_hand_str):
    """ Takes input directly from "routes", returns win ratios back. Handles kernel execution. """

    # get data ready for kernel
    game_over, remaining_deck_array, player_hand_array, dealer_hand_array = format_input_for_kernel(player_hand_str, dealer_hand_str)

    if game_over:
        # TODO:do something else, no need to call kernel

        # These are made-up numbers, need to figure out if blackjack or bust
        standing_winrate = 1
        hitting_winrate = 0
        pass
    else:
        # intial state data needed for the RNG
        rng_states = create_xoroshiro128p_states(num_threads_to_run, seed=777)

        # all zeroes, allocate arrays
        wins_standing = np.zeros(num_threads_to_run)
        wins_hitting = np.zeros(num_threads_to_run) 

        # execute kernel instances, both arrays will be updated
        blackjack_kernel[1, num_threads_to_run](wins_standing,
                                                wins_hitting,
                                                remaining_deck_array,
                                                player_hand_array,
                                                dealer_hand_array,
                                                games_per_thread,
                                                rng_states)

        standing_win_average = np.average(wins_standing)
        hitting_win_average = np.average(wins_hitting)

        standing_winrate = standing_win_average / games_per_thread
        hitting_winrate = hitting_win_average / games_per_thread

    return standing_winrate, hitting_winrate
