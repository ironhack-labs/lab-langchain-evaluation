# Exécuter le lab

1. Installer `python -m pip install -r requirements.txt` dans un environnement isolé.
2. Ouvrir le notebook et exécuter toutes les cellules. Le mode local utilise une recherche TF-IDF et une réponse extractive réelle.
3. Pour les modèles, renseigner `OPENAI_API_KEY` dans `.env`, passer `MODE='openai'`, puis activer selon besoin `RUN_GENERATED_EXAMPLES`, `RUN_JUDGE` et `RUN_RAGAS`.
4. Tests : `python -m unittest -v test_evaluation.py`.

Les sorties locales se trouvent dans `results/local_evaluation.json`. Les scores lexicaux ne sont pas des scores Ragas. Les contrôles avec réponse ou contexte volontairement incorrects sont identifiés comme tels.
