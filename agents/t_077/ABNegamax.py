import random
import time
from template import Agent
from Reversi.reversi_model import ReversiGameRule

MAX_SCORE = 99999
TIME_OUT = 0.95
CORNER = [(0, 0), (0, 7), (7, 0), (7, 7)]
INNER_CORNER = [(1, 1), (1, 6), (6, 1), (6, 6)]

class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.id = _id
        self.game_rule = ReversiGameRule(2)

    def SelectAction(self, actions, game_state):
        self.game_rule.agent_colors = game_state.agent_colors
        best_action = random.choice(actions)

        start_time = time.time()
        while time.time() - start_time <= TIME_OUT:
            legal_actions = self.game_rule.getLegalActions(game_state, self.id)

            cheat_actions = list(set(legal_actions).difference(set(INNER_CORNER)))
            if len(cheat_actions) > 0:
                legal_actions = cheat_actions

            cheat_actions = list(set(legal_actions).intersection(set(CORNER)))
            if len(cheat_actions) > 0:
                legal_actions = cheat_actions

            _, best_action = self.ABNegamax(game_state, legal_actions, self.id, -MAX_SCORE, MAX_SCORE, 2)

            return best_action

        return best_action

    def ABNegamax(self, game_state, actions, agent_id, alpha, beta, depth):
        if self.game_rule.gameEnds() or depth == 0:
            return [self.game_rule.calScore(game_state, agent_id), None]

        max_score = -MAX_SCORE
        for action in actions:
            next_state = self.game_rule.generateSuccessor(game_state, action, agent_id)
            opp_actions = self.game_rule.getLegalActions(next_state, 1 - agent_id)

            score = -self.ABNegamax(next_state, opp_actions, 1 - agent_id, -beta, -alpha, depth - 1)[0]

            if score > max_score:
                max_score = score
                best_action = action

            alpha = max(alpha, score)
            if alpha >= beta:
                return [alpha, best_action]

        return [max_score, best_action]
