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
def super_simple_example_runner():
    threads_to_run = 10 # total threads/kernels, make this user-configurable
    games_per_thread = 1000 # total simulations per thread, also configurable

    # intial state data needed for the RNG
    rng_states = create_xoroshiro128p_states(threads_to_run, seed=777)

    # all zeroes, the real point here is to allocate the entire array
    wins_array = np.zeros(threads_to_run) 

    # wins_array gets updated during execution
    blackjack_kernel[1, 10](wins_array, games_per_thread, rng_states)

    # print out array for reference
    print(wins_array)

super_simple_example_runner() # call runner, remove this in final product