import sys
import os
import pytest
import gymnasium as gym

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.utils import BlackjackStrategy

@pytest.fixture
def env():
    """Maakt een nieuwe Blackjack-omgeving aan."""
    env = gym.make("Blackjack-v1", natural=True, sab=False)
    yield env
    env.close()

@pytest.fixture
def strategy():
    """Maakt een nieuwe instantie van BlackjackStrategy aan."""
    return BlackjackStrategy()

def test_integration(env, strategy):
    """Test de strategie over meerdere episodes."""
    num_episodes = 1000
    total_reward = 0

    for episode in range(num_episodes):
        observation, _ = env.reset()
        done = False

        while not done:
            player_total, dealer_card, usable_ace = observation
            action = strategy.get_optimal_move(dealer_card, player_total, usable_ace)

            # Converteer actie naar Gymnasium formaat (0=stand, 1=hit)
            gym_action = 1 if action == "hit" else 0
            observation, reward, done, _, _ = env.step(gym_action)

        total_reward += reward

    # Controleer of total_reward een nummer is
    assert isinstance(total_reward, (int, float)), "Totaal reward moet een nummer zijn."
    # Controleer of rewards binnen het verwachte bereik liggen
    assert -num_episodes <= total_reward <= num_episodes, "Rewards moeten binnen het verwachte bereik liggen."
