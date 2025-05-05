import gymnasium as gym
from utils import BlackjackStrategy

def play_blackjack(num_episodes=10):
    """Speelt een aantal episodes van Blackjack met een bepaalde strategie.
    
    Dit is de hoofdfunctie die de interactie met de Blackjack-omgeving beheert.
    De functie speelt een opgegeven aantal episodes waarbij de optimale zet van de 
    speler wordt berekend op basis van de dealerkaart en het totaal van de speler.
    Na elke episode wordt de behaalde beloning weergegeven.
    
    Parameters:
    num_episodes (int): Het aantal episodes om te spelen (standaard 10).
    """
    env = gym.make("Blackjack-v1", natural=True, sab=False)  # Maak de Blackjack-omgeving
    strategy = BlackjackStrategy()  # Maak een nieuwe Blackjack-strategie
    
    for episode in range(num_episodes):
        observation, info = env.reset()  # Reset de omgeving voor elke episode
        done = False
        
        while not done:
            player_total, dealer_card, usable_ace = observation  # Verkrijg de spelstatus
            
            # Bepaal de optimale actie volgens de strategie
            action = strategy.get_optimal_move(dealer_card, player_total, usable_ace)
            
            # Zet de actie om naar de Gymnasium-acties (0=stand, 1=hit)
            gym_action = 1 if action == "hit" else 0
            
            # Voer de actie uit in de omgeving en krijg de nieuwe status
            observation, reward, done, truncated, info = env.step(gym_action)
        
        print(f"Episode {episode+1}: Reward = {reward}")

if __name__ == "__main__":
    play_blackjack(num_episodes=100)


def test_strategy():
    """Test de basisbeslissingen van de Blackjack-strategie.
    
    Deze functie test of de basisstrategie van Blackjack correct werkt door 
    een reeks testgevallen te simuleren. Het vergelijkt de verwachte actie met 
    de werkelijke actie die door de strategie wordt genomen. Het print de resultaten
    van elke test om te bevestigen of de strategie correct is.
    """
    strategy = BlackjackStrategy()  # Maak een nieuwe Blackjack-strategie
    
    # Test 1: Basis hit/stand beslissingen
    print("\n=== Test 1: Basis strategie ===")
    test_cases = [
        ("5", 16, False, "hit"),      # Hard 16 vs 5 → hit
        ("A", 12, True, "hit"),       # Soft 12 (A+2) vs A → hit
        ("6", 18, False, "stand"),    # Hard 18 vs 6 → stand
    ]
    
    for dealer_card, player_total, usable_ace, expected_action in test_cases:
        action = strategy.get_optimal_move(dealer_card, player_total, usable_ace)
        print(f"Speler {player_total} (Ace: {usable_ace}) vs Dealer {dealer_card}:")
        print(f"  Verwachte: {expected_action}, Kreeg: {action} → {'goed' if action == expected_action else 'fout'}")

def simulate_episodes(num_episodes=100):
    """Simuleer meerdere episodes van Blackjack en bereken de gemiddelde beloning.
    
    Deze functie simuleert een opgegeven aantal episodes in de Blackjack-omgeving.
    Na elke episode wordt de behaalde beloning toegevoegd aan de totale beloning.
    Na afloop van alle episodes wordt de gemiddelde beloning per episode weergegeven.
    
    Parameters:
    num_episodes (int): Het aantal episodes om te simuleren (standaard 100).
    """
    env = gym.make("Blackjack-v1", natural=True, sab=False)  # Maak de Blackjack-omgeving
    strategy = BlackjackStrategy()  # Maak een nieuwe Blackjack-strategie
    total_reward = 0

    for episode in range(num_episodes):
        observation, _ = env.reset()  # Reset de omgeving voor elke episode
        done = False
        
        while not done:
            player_total, dealer_card, usable_ace = observation  # Verkrijg de spelstatus
            action = strategy.get_optimal_move(dealer_card, player_total, usable_ace)
            gym_action = 1 if action == "hit" else 0
            observation, reward, done, _, _ = env.step(gym_action)
        
        total_reward += reward
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}: Reward = {reward}, Running Total = {total_reward}")

    print(f"\nGemiddelde reward per episode: {total_reward / num_episodes:.2f}")
    env.close()

if __name__ == "__main__":
    test_strategy()       # Unit tests
    simulate_episodes(100)  # Simulatie met Gymnasium

