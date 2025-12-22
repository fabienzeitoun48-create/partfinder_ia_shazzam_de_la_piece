from database_standards import STANDARDS_TECHNIQUES

def agent_standardiste(dimensions_estimees: str):
    """
    Agent 2 : Correspondance avec les standards du marché.
    Remplace l'agent spécialiste Somfy/Legrand.
    """
    prompt = f"""Tu es un expert en métrologie et standards de construction.
    Dimensions analysées : {dimensions_estimees}
    Référentiel : {STANDARDS_TECHNIQUES}
    
    Ta mission :
    1. **Identification du Standard** : Détermine le pas (ex: Gaz 1/2", Métrique M8, etc.).
    2. **Équivalence** : Si la pièce est ancienne, donne la référence standard moderne correspondante.
    3. **Précision** : Indique si une mesure au pied à coulisse est nécessaire pour confirmer entre deux standards proches (ex: 12/17 vs 15/21).
    """
    return prompt