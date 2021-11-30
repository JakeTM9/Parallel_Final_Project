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


def processHand(hand):
    hand_list = hand.split(",")
    value_list = []
    for card in hand_list:
        value_list.append(theDisgustingFunction(card))
    return hand_list, value_list

def theDisgustingFunction (card):
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

def getTotal(values):
    total = 0
    numAces = 0
    for value in values:
        total += value
        if value == 11:
            numAces += 1
        if total > 21 and numAces > 0:
            total -= 10
            numAces -= 1
    return total   

##Called by Front end for
def formatInputForBlackJack (playerHand, dealerHand): 
    player_hand_cards, player_hand_values = processHand(playerHand)
    dealer_hand_cards, dealer_hand_values = processHand(dealerHand)
    playerTotal = getTotal(player_hand_values)
    dealerTotal = getTotal(dealer_hand_values)
    return player_hand_cards, player_hand_values, dealer_hand_cards, dealer_hand_values, playerTotal, dealerTotal

## Checks if Player has busted or has 21
def initializeIsGameOver(playerTotal):
    if(playerTotal < 21):
        isGameOver = False
    else:
        isGameOver = True
    return isGameOver

 ## Retunrs New Deck with changing size PROB NOT GONNA USE BUT ITS HERE
def initializeDeckBad(player_hand_values, dealer_hand_values):
    Deck = np.array([11,11,11,11,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,9,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10])
    for value in player_hand_values:
        Deck = np.delete(Deck, np.where(Deck==value)[0][0])
    for value in dealer_hand_values:
        Deck = np.delete(Deck, np.where(Deck==value)[0][0])
    return Deck

## Returns Deck with 0s where no card size 52
def initializeDeck(player_hand_values, dealer_hand_values): 
    Deck = np.array([11,11,11,11,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,9,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10])
    for value in player_hand_values:
        Deck[np.where(Deck==value)[0][0]] = 0
    for value in dealer_hand_values:
        Deck[np.where(Deck==value)[0][0]] = 0
    return Deck

## Returns Hand with 0s where no card size 12 (not the best implementation but I was spending too much time on this)
def normalizeHand(hand):
    hand = np.array(hand)
    zerosToAdd = 12 - hand.size
    for i in range(0,zerosToAdd):
        hand = np.insert(hand,hand.size, 0)
    return hand

def formatInputForKernel (playerHand, dealerHand):
    player_hand_cards, player_hand_values, dealer_hand_cards, dealer_hand_values, playerTotal, dealerTotal = formatInputForBlackJack (playerHand, dealerHand)
    
    ## Checks if Player has busted or has 21
    isGameOver = initializeIsGameOver(playerTotal)
    ## Deck values without player values or dealer known values
    Deck = initializeDeck(player_hand_values, dealer_hand_values)
    ##player and dealer have a fixed array size of 12, cards are non-zero (unsure how handling dealer's uknown card so for now he is treated like a player)
    playerHandNormalized = normalizeHand(player_hand_values)
    dealerHandNormalized = normalizeHand(dealer_hand_values)

    ##printing these in console so you can see (hit submit on input)
    print(Deck, file=sys.stderr)
    print(playerHandNormalized, file=sys.stderr)
    print(dealerHandNormalized, file=sys.stderr)

    #hitOnFirst = not sure if call on kernel launch
    return isGameOver, Deck, playerHandNormalized, dealerHandNormalized
# This function will eventually be removed, it's here for testing/reference
def super_simple_example_runner(threads_to_run, playerHand, dealerHand):

    isGameover, Deck, playerHand, dealerHand = formatInputForKernel(playerHand, dealerHand)
    
    #threads to run = total threads/kernels, make this user-configurable
    games_per_thread = 1000 # total simulations per thread, also configurable

    # intial state data needed for the RNG
    rng_states = create_xoroshiro128p_states(threads_to_run, seed=777)

    # all zeroes, the real point here is to allocate the entire array
    wins_array = np.zeros(threads_to_run) 

    # wins_array gets updated during execution
    blackjack_kernel[1, threads_to_run](wins_array, games_per_thread, rng_states)

    ##NEW KERNEL ARGS?
    #blackjack_kernel[1, threads_to_run](wins_array, games_per_thread, rng_states, isGameover, Deck, playerHand, dealerHand)

    # print out array for reference
    return wins_array

#playerHand = "as,5h,10s,kh"
#dealerHand = "as,5h,10s,kh"         
        
#formatInputForBlackJack(playerHand,dealerHand) #TESTS FUNCTION


#super_simple_example_runner() # call runner, remove this in final product