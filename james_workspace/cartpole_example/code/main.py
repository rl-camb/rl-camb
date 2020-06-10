from env import CartPoleTravel, CartPoleStandUp

import gym, time, random, math, pickle, os, sys, copy

if __name__ == "__main__":

    # ONE - solve the standard cart pole
    
    cart = CartPoleStandUp(max_episodes=2000, score_target=195., episodes_threshold=100)
    
    # Look at the obs and action space
    cart.get_spaces(registry=False)

    # Do some random runs to view
    # cart.do_random_runs(20, 100, wait=0.05)
    
    # Actually solve
    # cart.solve(plot=True, verbose=True)

    # TWO - solve the traveller problem
    
    cart_traveller = CartPoleTravel(max_episodes=2000, position_target=0.3)
    
    cart_traveller.get_spaces(registry=False)
    cart_traveller.solve(plot=True, verbose=True)

    # THREE - other ideas I didn't have time for

    # Can we turn-off an RL agent (and stop the rise of the killer robots)..?
    # See how quickly a cart learns to avoid killswitch
    # (e.g. if it moves 1 unit to left, it dies)
    # 
