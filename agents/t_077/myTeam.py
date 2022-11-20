from copy import deepcopy
from turtle import st
from template import Agent
from Reversi.reversi_model import ReversiGameRule
from collections import deque
import random
import time
TIME_THINK = 0.95
CORNER = [(0,0),(0,7),(7,0),(7,7)]
CORNER_77= [(1,1),(1,6),(6,1),(6,6)]
EDGE_EXPECT = [(0,1),(0,3),(0,5),(7,1),(7,3),(7,5)]
GAMMA = 0.8
EPSILON = 0.8 
class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
        self.rule = ReversiGameRule(2)
    
    # similar to bfs
    # concern self aciton and use deterministic opponent action
    # define reward
    # step of mcts, select, expand,simulation, backpropagate
    # vt  = reward + grmma**vt +1 

    #select: e-greedy . choose max Q(s,a), with probability 1-e

    # choose action for opponents: 1.random,2.using a fixed policy
    def SelectAction(self,actions,game_state):
        self.rule.agent_colors = game_state.agent_colors
        time_start = time.time()
        action_corner = list(set(actions).intersection(set(CORNER)))
        if len(action_corner) == 1:
            return action_corner[0]
        if len(action_corner) > 0:
            actions = action_corner
        action_exp_77 = list(set(actions).difference(set(CORNER_77)))
        if len(action_exp_77) >0:
            actions = action_exp_77
        action_final = random.choice(actions)
        action_random = action_final

        v_s = dict()
        n_s = dict()
        action_be_s = dict()
        action_expand_s = dict()
        tree_root = 'r'

        def expand_f(state_cr, actions):
            if state_cr in action_expand_s:
                action_expan = action_expand_s[state_cr]
                return list(set(actions).difference(set(action_expan)))
            else:
                return actions
        
        while time.time() - time_start < TIME_THINK:
            curr_state = deepcopy(game_state)
            next_action = actions
            state_curr_tree = tree_root
            mcts_queue = deque([]) # for backpropagate
            reward = 0
            
            #select stage
            while len(expand_f(state_curr_tree,next_action)) == 0 and not self.Game_end(curr_state):
                if time.time() - time_start >= TIME_THINK:
                    print("select")
                    print(action_random)
                    print(action_final)
                    print(time.time() - time_start)
                    print(v_s)
                    return action_final
                    #select: e-greedy . choose max Q(s,a), with probability 1-e
                if (random.uniform(0,1)< (1-EPSILON)) and (state_curr_tree in action_be_s):
                    action_curr = action_be_s[state_curr_tree]
                else:
                    action_curr = random.choice(next_action)
                state_next, score_next = self.Action_run(curr_state ,action_curr)
                mcts_queue.append((state_curr_tree,action_curr))


                action_next_opp = self.Action_list_opp(state_next)
                state_be_opp = state_next
                score_max_opp = -1 
                for action_opp in action_next_opp:
                    state_next_opp, score_next_opp = self.Action_run_opp(state_next,action_opp )
                    if score_max_opp < score_next_opp :
                        score_max_opp = score_next_opp
                        state_be_opp = state_next_opp
                        action_be_opp = action_opp
                state_curr_tree = state_curr_tree+ str(action_curr[0])+str(action_curr[1])+str(action_be_opp[0])+str(action_be_opp[1])
                next_action = self.Action_list(state_be_opp)
                curr_state = state_be_opp

            #expand stage
            action_valid = expand_f(state_curr_tree,next_action)
            if len(action_valid) == 0:
                action_ex = random.choice(next_action)
            else:
                action_ex = random.choice(action_valid)
            if state_curr_tree in action_expand_s:
                action_expand_s[state_curr_tree].append(action_ex)
            else:
                action_expand_s[state_curr_tree] = [action_ex]
            mcts_queue.append((state_curr_tree,action_ex))
            state_next, score_next = self.Action_run(curr_state ,action_ex)

            action_next_opp = self.Action_list_opp(state_next)
            state_be_opp = state_next
            score_max_opp = 0
            for action_opp in action_next_opp:
                state_next_opp, score_next_opp = self.Action_run_opp(state_next,action_opp)
                if score_max_opp < score_next_opp :
                    score_max_opp = score_next_opp
                    state_be_opp = state_next_opp
                    action_be_opp = action_opp
            state_curr_tree = state_curr_tree+ str(action_ex[0])+str(action_ex[1])+str(action_be_opp[0])+str(action_be_opp[1])
            next_action = self.Action_list(state_be_opp)
            curr_state = state_be_opp

            length = 0
            #simulation stage

            while not self.Game_end(curr_state):
                length +=1
                if time.time() - time_start >= TIME_THINK:
                    print("simulation")
                    print(time.time() - time_start)
                    print(action_random)
                    print(action_final)
                    print(v_s)
                    return action_final
                
                action_si = random.choice(next_action)
                state_next, score_next = self.Action_run(curr_state ,action_si)

                action_next_opp = self.Action_list_opp(state_next)
                state_be_opp = state_next
                score_max_opp = -1 
                for action_opp in action_next_opp:
                    state_next_opp, score_next_opp = self.Action_run_opp(state_next,action_opp)
                    if score_max_opp < score_next_opp :
                        score_max_opp = score_next_opp
                        state_be_opp = state_next_opp
                        action_be_opp = action_opp
                #state_curr_tree = state_curr_tree+ str(action_curr[0])+str(action_curr[1])+str(action_be_opp[0])+str(action_be_opp[1])
                next_action = self.Action_list(state_be_opp)
                curr_state = state_be_opp
            reward = self.cal_reward(curr_state)
            # backpropagate
            vt_rew = reward * (GAMMA ** length)
            while len(mcts_queue) and time.time() - time_start < TIME_THINK:
                m_state, curr_action = mcts_queue.pop()
                if m_state in v_s:
                    if vt_rew > v_s[m_state]:
                        v_s[m_state] = vt_rew
                        action_be_s[m_state] = curr_action
                    n_s[m_state] +=1
                else:
                    v_s[m_state] = vt_rew
                    n_s[m_state] = 1
                    action_be_s[m_state] = curr_action
                vt_rew *= GAMMA
            if tree_root in action_be_s:
                action_final = action_be_s[tree_root]
                
        return action_final

    def Action_list(self,game_state):
        return self.rule.getLegalActions(game_state,self.id)
    
    def Action_run(self,game_state,action):
        curr_state = self.rule.generateSuccessor(game_state,action,self.id)
        curr_score = self.rule.calScore(curr_state,self.id)
        return (curr_state,curr_score)

    def Game_end(self,game_state):
        return (self.Action_list(game_state) == ["Pass"] and self.Action_list_opp(game_state) == ["Pass"])

    def Action_run_opp(self,game_state,action):
        curr_state = self.rule.generateSuccessor(game_state,action, 1 - self.id)
        curr_score = self.rule.calScore(curr_state,1- self.id)
        return (curr_state,curr_score)

    def Action_list_opp(self,game_state):
        return self.rule.getLegalActions(game_state,1-self.id)
    
    def cal_reward(self,game_state):
        return self.rule.calScore(game_state,self.id) - self.rule.calScore(game_state,1-self.id)
