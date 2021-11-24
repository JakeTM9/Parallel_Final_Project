import sys
import numba 
import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

@cuda.jit
def blackjack_kernel(wins_array, simulations_to_run, rng_states):

    # TODO: something about gamestate data input
    thread_position = cuda.threadIdx.x # probably make this a different call
    wins = 0

    for _ in range(simulations_to_run):
        # this section will be the "core", real game logic will go here

        # begin basic example:
        # get random number from 0 to 1
        random_float_0_to_1 = xoroshiro128p_uniform_float32(rng_states,
                                                            thread_position)
        if random_float_0_to_1 > .5:
            wins += 1
        # end basic example:

    wins_array[thread_position] = wins

# This function will eventually be removed, it's here for testing/reference
def super_simple_example_runner(threads_to_run):
    #threads to run = total threads/kernels, make this user-configurable
    games_per_thread = 1000 # total simulations per thread, also configurable

    # intial state data needed for the RNG
    rng_states = create_xoroshiro128p_states(threads_to_run, seed=777)

    # all zeroes, the real point here is to allocate the entire array
    wins_array = np.zeros(threads_to_run) 

    # wins_array gets updated during execution
    blackjack_kernel[1, threads_to_run](wins_array, games_per_thread, rng_states)

    # print out array for reference
    return wins_array

def processHand(hand):
    hand_list = hand.split(",")
    value_list = []
    for card in hand_list:
        value_list.append(theDisgustingFunction(card))
    return hand_list, value_list

def theDisgustingFunction (card):
    if "a" in card:
        return 1
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

def getTotal(values):
    total = 0
    for value in values:
        total += value
    return total   

def formatInputForBlackJack (playerHand, dealerHand):
    
    player_hand_cards, player_hand_values = processHand(playerHand)
    dealer_hand_cards, dealer_hand_values = processHand(dealerHand)
    playerTotal = getTotal(player_hand_values)
    dealerTotal = getTotal(dealer_hand_values)
    return player_hand_cards, player_hand_values, dealer_hand_cards, dealer_hand_values, playerTotal, dealerTotal
    #print(player_hand_cards)
    #print(player_hand_values)
    #print(dealer_hand_cards)
    #print(dealer_hand_values)
    #print(playerTotal)


playerHand = "as,5h,10s,kh"
dealerHand = "as,5h,10s,kh"         
        
#formatInputForBlackJack(playerHand,dealerHand) #TESTS FUNCTION


#super_simple_example_runner() # call runner, remove this in final product