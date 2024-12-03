import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog, minimize

class BestResponseCalculator:
    def __init__(self, player_utility, strategy_space, num_firms=2, repeated=False):
        """
        Parameters:
        - player_utility (function): A function representing the utility of the player. It should accept the player's
          strategy and the opponent's strategy as arguments.
        - strategy_space (array-like): A list or array representing the available strategies for the player.
        - num_firms (int): Number of firms in the market (default is 2).
        - repeated (bool): Whether the game is infinitely repeated (default is False).
        """
        self.player_utility = player_utility
        self.strategy_space = strategy_space
        self.num_firms = num_firms
        self.repeated = repeated

    def calculate_best_response(self, opponent_strategy):
        best_strategy = None
        max_utility = -np.inf

        
        for strategy in self.strategy_space:
            utility = self.player_utility(strategy, opponent_strategy)
            if utility > max_utility:
                max_utility = utility
                best_strategy = strategy

        return best_strategy

    def cournot_best_response(self, opponent_quantity, price_intercept=100, cost=20):
        def utility(q1, q2):
            price = price_intercept - (q1 + q2)
            profit = (price - cost) * q1
            return profit

        self.player_utility = utility
        return self.calculate_best_response(opponent_quantity)

    def multi_firm_cournot(self, num_firms, cost=20, price_intercept=100):
        def total_profit(q):
            total = 0
            for i in range(num_firms):
                q_i = q[i]
                q_others = np.delete(q, i)
                total_quantity = q_i + sum(q_others)
                price = price_intercept - total_quantity
                profit = (price - cost) * q_i
                total -= profit  
            return total

        initial_guess = [10] * num_firms
        bounds = [(0, 50) for _ in range(num_firms)]
        result = minimize(total_profit, initial_guess, bounds=bounds, method='SLSQP')

        if result.success:
            return result.x
        else:
            return None

    def nash_equilibrium(self, utility_matrix):
        """
        Calculate the Nash equilibrium for a two-player game using linear programming.

        Parameters:
        - utility_matrix (2D array): A matrix representing the payoffs for Player 1 for each combination of strategies.

        Returns:
        - nash_strategy (array): The mixed strategy Nash equilibrium for Player 1.
        """
        num_strategies = len(utility_matrix)
        c = [-1] * num_strategies
        A_ub = -np.transpose(utility_matrix)
        b_ub = [-1] * len(utility_matrix[0])
        bounds = [(0, None) for _ in range(num_strategies)]

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

        if res.success:
            return res.x / sum(res.x)
        else:
            return None

    def mixed_strategy_nash(self, player_payoffs, opponent_payoffs):
        combined_payoffs = player_payoffs - opponent_payoffs
        
        num_strategies = len(player_payoffs)
        c = [-1] * num_strategies
        A_ub = -combined_payoffs
        b_ub = [-1] * len(combined_payoffs[0])
        bounds = [(0, None) for _ in range(num_strategies)]

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

        if res.success:
            return res.x / sum(res.x)
        else:
            return None

    def bayesian_nash_equilibrium(self, player_types, opponent_types, utility_function):
      
        bayesian_nash = {}

        for player_type in player_types:
            best_responses = []
            for opponent_type in opponent_types:
                def type_specific_utility(player_strategy, opponent_strategy):
                    return utility_function(player_type, opponent_type, player_strategy, opponent_strategy)

                self.player_utility = type_specific_utility
                best_response = self.calculate_best_response(opponent_strategy=0) 
                best_responses.append(best_response)

            bayesian_nash[player_type] = np.mean(best_responses) 

        return bayesian_nash

    def discounted_payoff(self, payoff_function, strategy, discount_factor=0.9, num_periods=100):
        total_payoff = 0
        for t in range(num_periods):
            discounted_value = (discount_factor ** t) * payoff_function(strategy)
            total_payoff += discounted_value
        return total_payoff

    def plot_best_response(self, opponent_quantity, price_intercept=100, cost=20):
        best_responses = []
        opponent_quantities = np.linspace(0, 50, 100)
        for q2 in opponent_quantities:
            best_responses.append(self.cournot_best_response(q2, price_intercept, cost))

        plt.plot(opponent_quantities, best_responses, label='Best Response Curve')
        plt.xlabel('Opponent Quantity')
        plt.ylabel('Best Response Quantity')
        plt.title('Best Response Curve in Cournot Competition')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_cournot_3d(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        q1 = np.linspace(0, 50, 50)
        q2 = np.linspace(0, 50, 50)
        q1, q2 = np.meshgrid(q1, q2)
        price = 100 - q1 - q2
        profit1 = (price - 20) * q1

        ax.plot_surface(q1, q2, profit1, cmap='viridis')
        ax.set_xlabel('Firm 1 Quantity')
        ax.set_ylabel('Firm 2 Quantity')
        ax.set_zlabel('Profit for Firm 1')
        plt.title('3D Surface Plot for Firm 1 Profit')
        plt.show()

# Example:

strategy_space = np.linspace(0, 50, 500)


calculator = BestResponseCalculator(player_utility=None, strategy_space=strategy_space, num_firms=2, repeated=False)

opponent_quantity = 30


best_response_quantity = calculator.cournot_best_response(opponent_quantity)

print(f"The best response for Firm 1, given that Firm 2 produces {opponent_quantity} units, is to produce {best_response_quantity:.2f} units.")


calculator.plot_best_response(opponent_quantity)


num_firms = 3
equilibrium_quantities = calculator.multi_firm_cournot(num_firms)
print(f"Equilibrium quantities for {num_firms} firms: {equilibrium_quantities}")


calculator.plot_cournot_3d()


utility_matrix = [[3, 1], [0, 2]]
nash_strategy = calculator.nash_equilibrium(utility_matrix)
print(f"The mixed strategy Nash equilibrium for Player 1 is: {nash_strategy}")

def bayesian_utility(player_type, opponent_type, player_strategy, opponent_strategy):
    return (100 - player_strategy - opponent_strategy) * player_strategy - 20 * player_strategy

player_types = [1, 2]
opponent_types = [1, 2]

bayesian_nash = calculator.bayesian_nash_equilibrium(player_types, opponent_types, bayesian_utility)
print(f"The Bayesian Nash equilibrium strategies are: {bayesian_nash}")
