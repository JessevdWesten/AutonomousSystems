import pytest
from src.utils import BlackjackStrategy

@pytest.fixture
def strategy():
    """Maakt een nieuwe instantie van BlackjackStrategy aan."""
    return BlackjackStrategy()

def test_basic_strategy(strategy):
    """Test basisstrategie zonder kaarttelling of inzetlogica."""
    
    test_cases = [
        # (dealer_card, player_total, usable_ace, verwachte_actie)
        ('5', 16, False, 'stand'),  # Hard 16 vs 5 → stand
        ('7', 16, False, 'hit'),   # Hard 16 vs 7 → hit
        ('6', 17, False, 'stand'),  # Hard 17 vs 6 → stand
        ('A', 17, True, 'hit'),    # Soft 17 (A+6) vs A → hit
    ]
    
    for dealer_card, player_total, usable_ace, expected in test_cases:
        action = strategy.get_optimal_move(dealer_card, player_total, usable_ace)
        assert action == expected, f"Fout bij player_total={player_total}, dealer_card={dealer_card}, usable_ace={usable_ace}. Verwacht {expected}, kreeg {action}"
