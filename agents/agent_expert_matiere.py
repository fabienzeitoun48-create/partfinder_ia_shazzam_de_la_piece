def agent_expert_matiere(analyse_visuelle: str):
    """
    Agent 1 : Analyse des matériaux et de l'usure.
    Remplace l'agent diagnostiqueur.
    """
    prompt = f"""Tu es un ingénieur expert en science des matériaux et quincaillerie industrielle.
    Basé sur cette description visuelle : {analyse_visuelle}
    
    Rédige un rapport sur :
    1. **Nature du Matériau** : Identifie s'il s'agit de laiton (jaune), inox, acier galva, ou zamak.
    2. **Analyse de l'état** : Repère les signes de fatigue (fissures, calcaire, corrosion, filetage émoussé).
    3. **Verdict technique** : La pièce est-elle réparable (ex: changement de joint) ou doit-elle être remplacée ?
    """
    return prompt