"""Base de données des standards Plomberie & Quincaillerie"""

STANDARDS_TECHNIQUES = {
    "filetages_plomberie": {
        "12/17": "3/8 Pouce - Courant pour robinets d'arrêt et flexibles",
        "15/21": "1/2 Pouce - Standard pour douches et robinetterie classique",
        "20/27": "3/4 Pouce - Arrivées d'eau machine à laver / compteurs",
        "26/34": "1 Pouce - Installations de chauffage ou pompage"
    },
    "types_tetes": [
        "Tête à clapet (ancienne génération, joint caoutchouc)",
        "Tête céramique 1/4 tour (moderne)",
        "Cartouche thermostatique (mitigeur douche)"
    ],
    "quincaillerie_charnieres": [
        "Charnière invisible à boîtier (standard 35mm)",
        "Fiche à visser (meuble ancien)",
        "Paumelle de porte standard"
    ]
}

def get_standards_summary():
    return str(STANDARDS_TECHNIQUES)