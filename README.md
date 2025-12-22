
---
title: PartFinder AI - Le Shazam de la Pièce Détachée
emoji: 🔍
colorFrom: orange
colorTo: gray
sdk: docker
app_file: app.py
pinned: false
---

# 🔍 PartFinder AI

**Le "Shazam" de la pièce détachée pour la plomberie et la quincaillerie.** PartFinder AI permet d'identifier instantanément des pièces anciennes ou inconnues (joints, têtes de robinet, charnières) à partir d'une simple photo et de trouver immédiatement où les acheter.

## 🎯 Objectif

Résoudre le cauchemar des techniciens de maintenance et des bricoleurs : identifier une pièce cassée sans référence visible et obtenir sa correspondance moderne ou son équivalent standard en moins de 30 secondes.

## 🧠 Architecture Multi-Agents

L'application repose sur une orchestration de trois intelligences spécialisées :

1.  **L'Expert Vision (Groq Llama 3.2)** : Analyse la géométrie, les filetages, les matériaux et l'état d'usure sur la photo.
2.  **Le Standardiste (Python Logic)** : Compare les mesures visuelles aux standards industriels (diamètres 12/17, 15/21, pas métriques, types de cartouches).
3.  **Le Sourcer Marchand (Perplexity AI)** : Recherche en temps réel la disponibilité des stocks chez les fournisseurs pro et grand public (ManoMano, Leroy Merlin, Cédéo, Amazon).

## 📊 Stack Technique

- [cite_start]**Backend** : FastAPI (Python 3.11+) [cite: 1]
- **Analyse Visuelle** : Groq Vision (Modèle Llama 3.2)
- **Intelligence Sourcing** : Perplexity Sonar API (Web Search temps réel)
- **Base de Connaissances** : Dictionnaire local des standards de quincaillerie et plomberie.
- **Frontend** : PWA (Progressive Web App) en HTML/JS Vanilla pour un usage fluide sur smartphone en intervention.

## 🚀 Fonctionnalités Clés

- 📷 **Capture Photo** : Prise de vue directe depuis le smartphone sur le lieu de l'intervention.
- 📏 **Identification des Standards** : Détection automatique des filetages et dimensions probables.
- 🛒 **Liens d'Achat** : Boutons directs vers les fiches produits des marchands disponibles.
- 🛠️ **Conseils de Remplacement** : Suggestions de pièces modernes compatibles avec les installations anciennes.

## 🛠️ Installation

1. Clonez le dépôt.
2. [cite_start]Installez les dépendances : `pip install -r requirements.txt`[cite: 1].
3. Configurez vos clés API dans un fichier `.env` (`GROQ_API_KEY`, `PERPLEXITY_API_KEY`).
4. Lancez le serveur : `uvicorn app:app --reload`.
