from template import Agent
from Reversi.reversi_model import ReversiGameRule
import random
import time
from collections import deque

TIME_LIMIT = 0.92
CORNER = [(0, 0), (0, 7), (7, 0), (7, 7)] #Preferred part of the selection
INNER_CORNER = [(1, 1), (1, 6), (6, 1), (6, 6)]# Parts to avoid selecting


class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.game_rule = ReversiGameRule(2)

    def GetActions(self, state, agent_id):
        return self.game_rule.getLegalActions(state, agent_id)

    def AfterAction(self, state, action, agent_id):
        new_state = self.game_rule.generateSuccessor(state, action, agent_id)
        new_score = self.game_rule.calScore(new_state, agent_id)
        return new_state, new_score

    def SelectAction(self, actions, game_state):
        self.game_rule.agent_colors = game_state.agent_colors
        start_point = time.time()
        solution = random.choice(actions)
        queue = deque([(game_state, [])])
        max_score = 0

        while time.time() - start_point < TIME_LIMIT and len(queue):

            state, path = queue.popleft()
            available_actions = self.GetActions(state, self.id)

            strategy_action = list(set(available_actions).difference(set(INNER_CORNER)))
            if len(strategy_action) > 0:
                available_actions = strategy_action
            strategy_action = list(set(available_actions).intersection(set(CORNER)))
            if len(strategy_action) > 0:
                available_actions = strategy_action

            for action in available_actions:
                next_path = path + [action]
                next_state, next_score = self.AfterAction(state, action, self.id)

                if self.game_rule.gameEnds and next_score > max_score:
                    max_score = next_score
                    solution = next_path[0]
                if time.time() - start_point > TIME_LIMIT:
                    break

                opponent_max_score = 0
                opponent_next_action = self.GetActions(next_state, 1 - self.id)

                for opponent_action in opponent_next_action:
                    opponent_next_state, opponent_next_score = self.AfterAction(next_state, opponent_action, 1 - self.id)
                    if opponent_max_score < opponent_next_score:
                        opponent_max_score = opponent_next_score
                        next_state = opponent_next_state

                queue.append((next_state, next_path))

        return solution
