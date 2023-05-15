import numpy as np

class Node:
    def __init__(self, parent=None, action_prob=1.0):
        self.parent = parent
        self.children = {}  # dictionary of {action: node}
        self.visit_count = 0
        self.total_reward = 0.0
        self.action_prob = action_prob  # probability of the action that led to this node

    def expand(self, action_probs):
        for action, prob in enumerate(action_probs):
            if action not in self.children:
                self.children[action] = Node(parent=self, action_prob=prob)

    def select_child(self):
        """
        Select the child node with the highest UCT value.
        UCT value is a combination of the action's expected reward and exploration potential.
        """
        uct_values = [(action, child.uct_value()) for action, child in self.children.items()]
        action, _ = max(uct_values, key=lambda x: x[1])
        return action, self.children[action]

    def update(self, reward):
        self.visit_count += 1
        self.total_reward += reward

    def uct_value(self):
        """
        Calculate the UCT value.
        This value balances exploitation (choosing actions with high expected reward)
        and exploration (choosing less-visited actions).
        """
        exploit = self.total_reward / self.visit_count
        explore = np.sqrt(np.log(self.parent.visit_count) / self.visit_count)
        return self.action_prob * (exploit + explore)

      
def MCTS(root_state, model, env, iterations):
    root_node = Node(state=root_state)

    for _ in range(iterations):
        node = root_node
        state = root_state.copy()

        # Selection
        while node.children:
            action, node = node.select_child()
            state, _, _, _ = env.step(action)

        # Expansion
        if not env.done:  # check if the game is over
            action_probs = model.predict(np.array([state]))[0]
            node.expand(action_probs)

        # Simulation
        while not env.done:  # simulate until the game ends
            action = np.random.choice(env.action_space.n)
            state, reward, _, _ = env.step(action)

        # Backpropagation
        while node is not None:
            node.update(reward)
            node = node.parent

    return np.argmax([child.visit_count for child in root_node.children])
