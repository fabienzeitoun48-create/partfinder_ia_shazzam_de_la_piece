async def agent_sourcer(description_finale: str):
    """
    Agent 3 : Sourcing et Disponibilité.
    Remplace l'agent documenteur.
    """
    # Ce prompt est envoyé à Perplexity via l'app principale
    prompt = f"""Cherche les fournisseurs et prix pour la pièce suivante : {description_finale}.
    
    Instructions :
    1. Trouve des liens directs chez Leroy Merlin, castorama, brico depot, ManoMano, Mr.Bricolage, Union Matériaux , Au Forum du Bâtiment , Sikkens Solutions, Cédéo ou Amazon.
    2. Précise le prix moyen constaté.
    3. Si la pièce est épuisée, propose une alternative compatible.
    
    Formatte avec des liens cliquables.
    """
    return prompt