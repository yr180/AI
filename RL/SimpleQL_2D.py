"""
Reinforcement Learning using Q-learning method.
An agent "o" is on the top-left corner of a 2D world, the treasure is on the bottom-right corner.
In runtime this program shows how the agent will improve its strategy of finding the treasure.
"""

import numpy as np
import time

np.random.seed(4)  # reproducible

#hyperparameters
N_STATES = 5                    		# the size of the 1D world
ACTIONS = [0,1,2,3]                             #['right', 'left', 'up', 'down']     	
EPSILON = 0.9   				# greedy policy const
ALPHA = 0.8     				# learning rate
GAMMA = 0.9    					# discount factor
MAX_EPOCHS = 50   				# maximum epochs
REFRESH_TIME = 0.1   			        # fresh time for one move


def build_q_table(n_states, actions):
    table = np.zeros((n_states, n_states, len(actions)))   # q_table initial value
    #print(table)                               # show table
    return table


def choose_action(state, q_table):
    # This is how to choose an action
    state_actions = q_table[state[0]][state[1]][:]
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):  # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)
    else:   # act greedy
        action_name = state_actions.argmax()
    return action_name


def get_env_feedback(S,A):                      # This is how agent will interact with the environment
    S_ = [0,0]
    if A == 0:                                  # move right
        if S[0] == N_STATES - 2 and S[1] == N_STATES - 1:                # terminate since target reached
            S_[0] = 'terminal'
            R = 1                               # Assign reward as 1 for finding target
            S_[1] = S[1] 
        elif S[0] == N_STATES - 1:              # reach the wall
            R = 0
            S_ = S
        else:
            S_[0] = S[0] + 1                    
            R = 0
            S_[1] = S[1]

    elif A == 1:                                # move left
        R = 0
        if S[0] == 0:
            S_ = S                              # reach the wall
        else:
            S_[0] = S[0] - 1
            S_[1] = S[1]

    elif A == 2:                                # move up
        R = 0
        if S[1] == 0:
            S_ = S                              # reach the wall
        else:
            S_[1] = S[1] - 1
            S_[0] = S[0]

    elif A == 3:                                # move down
        if S[1] == N_STATES - 2 and S[0] == N_STATES - 1:                # terminate since target reached
            S_[1] = 'terminal'
            R = 1                               # Assign reward as 1 for finding target
            S_[0] = S[0] 
        elif S[1] == N_STATES - 1:              # reach the wall
            R = 0
            S_ = S
        else:
            S_[1] = S[1] + 1                    
            R = 0
            S_[0] = S[0]
    print S_
    return S_, R


def update_env(S, episode, step_counter):       # Updating the environment
    env = np.chararray((N_STATES,N_STATES))
    env[:] = '-'
    env[N_STATES - 1][N_STATES - 1] = 'T'
    
    if S[0] == 'terminal' or S[1] == 'terminal':
        interaction = 'Epoch %s: total_steps = %s' % (episode+1, step_counter)
        print('{}'.format(interaction))
        time.sleep(2)
        print('                                ')
    else:
        env[S[0]][S[1]] = 'o'
        for i in xrange(N_STATES):
            for j in xrange(N_STATES):
                print ' ', env[i][j],
            print '\n'
        time.sleep(REFRESH_TIME)


def rl():                                           # main part of RL loop
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPOCHS):
        step_counter = 0
        S = [0,0]
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:

            A = choose_action(S, q_table)
            print A
            S_, Rew = get_env_feedback(S, A)        # take action & get next state and reward
            q_predict = q_table[S[0]][S[1]][A]
            if S_[0] != 'terminal' and S_[1] != 'terminal':
                q_target = Rew + GAMMA * q_table[S_[0]][S_[1]].max()   # next state is not terminal
            else:
                q_target = Rew                      # next state is terminal
                is_terminated = True                # terminate this episode

            q_table[S[0]][S[1]][A] += ALPHA * (q_target - q_predict)  # update
            S = S_                                  # move to next state

            update_env(S, episode, step_counter+1)
            step_counter += 1
        #print(q_table)
    return q_table


if __name__ == "__main__":
    q_table = rl()                              #Begin Learning
    print('Q-table:')
    print(q_table)
