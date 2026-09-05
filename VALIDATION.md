# Validation locale

- Notebook exécuté intégralement en mode local, sans appel OpenAI.
- Cinq tests passent, dont la compatibilité du schéma Ragas et l’absence d’injection des références dans la chaîne de réponse.
- Source pertinente retrouvée : catalogue 4/4, NYC 5/5, corpus fictif de bibliothèque 2/3.
- Token F1 moyen : catalogue 0,040 ; NYC 0,114 ; bibliothèque 0,075.

Les faibles F1 rappellent qu’une recherche correcte ne suffit pas : la réponse extractive locale est souvent trop longue ou ne répond pas précisément à la question. L’échec sur la durée de prêt révèle une limite lexicale (question et source n’utilisent pas les mêmes mots). Ces résultats décrivent la baseline locale, pas GPT ni Ragas.

La génération de questions, le juge LLM et les scores sémantiques Ragas sont implémentés mais non exécutés faute de configuration API. Les sorties complètes sont dans le notebook et results/local_evaluation.json.
