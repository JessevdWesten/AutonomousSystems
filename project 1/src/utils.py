class BlackjackStrategy:
    def __init__(self):
        """Initialiseert een nieuwe instantie van de Blackjack-strategie.
        
        De HI_LO_COUNT wordt op 0 gezet bij de initialisatie. Deze teller houdt 
        het aantal bij voor de Hi-Lo kaarttelmethode, die gebruikt kan worden 
        om de strategie te verbeteren, hoewel deze momenteel niet in gebruik is.
        """
        self.HI_LO_COUNT = 0

    def count_cards(self, card):
        """Werk de HI_LO_COUNT bij op basis van de gedeelde kaart.
        
        We wilden experimenteren met hoe de strategie zou afwijken als er maar 
        één deck werd gebruikt, omdat er veel informatie over beschikbaar is.
        Voor nu zullen we dit echter niet gebruiken.
        
        Hi-Lo telmethode:
        - Lage kaarten (2-6) verhogen de count (+1), wat betekent dat er meer 
          hoge kaarten overblijven.
        - Hoge kaarten (10, J, Q, K, A) verlagen de count (-1), wat betekent 
          dat er minder hoge kaarten overblijven.
        - Neutrale kaarten (7-9) hebben geen invloed op de count.
        
        In een simulatie zonder kaarttelling kan deze methode genegeerd worden.
        """
        if card in ['2', '3', '4', '5', '6']:
            self.HI_LO_COUNT += 1
        elif card in ['10', 'J', 'Q', 'K', 'A']:
            self.HI_LO_COUNT -= 1

    def calculate_bet(self, current_bet):
        """Pas de inzet aan op basis van de Hi-Lo count.
        
        Zoals bij het tellen van kaarten, kunnen we proberen de inzetgrootte 
        aan te passen op basis van de count, maar we zullen dit voor nu niet gebruiken.
        
        - Als de count significant positief is (>2), verhoog de inzet (goede 
          kansen).
        - Als de count significant negatief is (<-2), verlaag de inzet (slechte 
          kansen).
        - Anders, houd de inzet gelijk.
        
        In een simulatie zonder inzetten kan deze functie genegeerd worden.
        """
        if self.HI_LO_COUNT > 2:
            return current_bet * 2
        elif self.HI_LO_COUNT < -2:
            return max(current_bet // 2, 1)
        return current_bet

    def get_optimal_move(self, dealer_card, player_total, usable_ace=False):
        """Bepaal de optimale zet op basis van de dealerkaart, het totaal van de speler en of er een bruikbare aas is.
        
        De optimale zet wordt bepaald door de volgende logica:
        - Bij harde handen (geen bruikbare aas) wordt altijd geanalyseerd of 
          de speler moet hitten of staan, afhankelijk van de waarde van de 
          dealerkaart en het totaal van de speler.
        - Bij zachte handen (met een bruikbare aas) wordt ook bepaald of de 
          speler moet hitten of staan, maar met extra aandacht voor de waarde 
          van de aas en de dealerkaart.
        
        Het doel van deze functie is om de beste zet te kiezen die het meeste 
        kans heeft om te winnen, afhankelijk van de situatie.
        """
        dealer_card = str(dealer_card)
        if dealer_card in ['J', 'Q', 'K']:  # Vervang face cards door 10
            dealer_card = '10'
        elif dealer_card.isdigit():
            dealer_card = int(dealer_card)

        # Hard handen (geen bruikbare aas)
        if not usable_ace:
            if player_total <= 11:
                return "hit"  # Altijd hitten bij 11 of lager
            elif player_total == 12:
                if dealer_card in [4, 5, 6]:
                    return "stand"  # Sta bij dealer 4-6
                else:
                    return "hit"
            elif 13 <= player_total <= 16:
                if dealer_card in [2, 3, 4, 5, 6]:
                    return "stand"  # Sta bij dealer 2-6
                else:
                    return "hit"
            else:  # 17+
                return "stand"  # Sta bij 17 of meer

        # Soft handen (bruikbare aas)
        else:
            if player_total <= 17:
                return "hit"  # Altijd hitten bij 17 of lager
            elif player_total == 18:
                if dealer_card in [9, 10, 'A']:
                    return "hit"  # Hit bij dealer 9, 10 of A
                else:
                    return "stand"
            else:  # 19+
                return "stand"  # Sta bij 19 of meer
