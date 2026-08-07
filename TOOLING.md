# CodeRabbit vs Qodo Merge — recommendation for this repo

Rédigé par l'agent de nuit (tâche 3, 2026-08-07). Sert à décider si l'on
installe un reviewer automatique en plus de Claude Code, et lequel.
Installation = action humaine (GitHub App), je ne peux pas le faire
moi-même — voir `TODO.md` `[+review-bot]`.

## Contexte pour ce repo

`dstft` est un **repo public**, mainteneur unique (Maxime), package
PyPI scientifique (PyTorch). Pas d'équipe, pas de flux élevé de PR
externes pour l'instant. Ça change ce qui compte : le prix par siège
et les fonctionnalités "équipe" pèsent peu ; ce qui compte, c'est le
tier gratuit pour repo public et la qualité de la review sur du code
scientifique/numérique.

## Comparaison (2026)

| | CodeRabbit | Qodo Merge |
|---|---|---|
| **Gratuit pour repo public** | Oui, sans limite de temps ni carte bancaire — les fonctionnalités Pro sont automatiquement actives sur les repos publics | Oui, listing dédié *"Qodo Merge Pro for open source"* sur le GitHub Marketplace |
| **Plateformes** | GitHub uniquement (SaaS) | GitHub, GitLab, Bitbucket, Azure DevOps + option self-hosted (PR-Agent, avec ta propre clé LLM) |
| **Installation** | GitHub App, ~5 min, aucune config CI/CD | GitHub App, ~5 min, aucune config CI/CD |
| **Style de review** | Résumé de PR + commentaires + 40+ linters déterministes intégrés + config en langage naturel + préférences apprises dans le temps | Architecture multi-agents (Qodo 2.0, février 2026) ; meilleur score F1 mesuré parmi 8 outils comparés dans un benchmark indépendant |
| **Tests manquants** | Signale la zone non testée en commentaire (tu écris le test) | Génère directement le test unitaire manquant |
| **Rate limits (tier gratuit)** | 4 reviews/heure, 200 fichiers/heure | 30 reviews/mois par org sur le plan gratuit hébergé (pas de limite connue sur le listing open-source dédié) |

## Recommandation

**CodeRabbit** est le choix le plus adapté pour ce repo précis :

- Le rate limit (4 reviews/heure) est largement suffisant pour un
  mainteneur solo qui ouvre quelques PR par jour au maximum — jamais
  un facteur limitant en pratique ici.
- Les 40+ linters déterministes intégrés font double emploi partiel
  avec `ruff`/`mypy` déjà en place, mais ajoutent une couche
  sémantique (résumé de PR, détection de régressions logiques) que
  `pre-commit`/CI ne fait pas.
- Pas de gestion de clé API à faire soi-même (contrairement à
  RepoAgent) : c'est une GitHub App, zéro secret à créer.

Qodo Merge reste un choix raisonnable si un jour la génération
automatique de tests devient prioritaire (`dstft` a une bonne
couverture — 86% — donc ce n'est pas le cas aujourd'hui), ou si le
repo migre un jour vers une autre forge que GitHub.

## Ce que tu dois faire toi-même (je ne peux pas installer de GitHub App)

**Pour CodeRabbit** :
1. Va sur [coderabbit.ai](https://www.coderabbit.ai) → "Get Started
   Free".
2. Connecte-toi avec ton compte GitHub, autorise l'app CodeRabbit.
3. Sélectionne le repo `maxime-leiber/dstft` (ou "All repositories").
4. Vérifie les permissions demandées (accès repo + PR) puis valide.
5. Rien d'autre à configurer : la prochaine PR ouverte déclenche une
   review automatique. Un fichier `.coderabbit.yaml` optionnel permet
   de personnaliser le comportement plus tard si besoin.

**Pour Qodo Merge** (si tu préfères cette option à la place) :
1. Va sur le listing [Qodo Merge Pro for open
   source](https://github.com/marketplace/qodo-merge-pro-for-open-source)
   sur le GitHub Marketplace.
2. Clique "Set up a plan" / "Configure", connecte ton compte GitHub.
3. Sélectionne `maxime-leiber/dstft` ("Only select repositories").
4. Valide les permissions demandées.

Les deux sont mutuellement exclusifs en pratique (pas besoin des
deux) ; installe l'un des deux, ou aucun si tu préfères t'appuyer
uniquement sur Claude Code + la CI existante pour l'instant.

## Sources

- [The 3 best CodeRabbit alternatives for AI code review in 2026 — cubic.dev](https://www.cubic.dev/blog/the-3-best-coderabbit-alternatives-for-ai-code-review-in-2025)
- [Qodo vs CodeRabbit: AI Code Review Tools Compared (2026) — DEV Community](https://dev.to/rahulxsingh/qodo-vs-coderabbit-ai-code-review-tools-compared-2026-kdp)
- [How to Setup CodeRabbit: Complete Step-by-Step Guide (2026) — DEV Community](https://dev.to/rahulxsingh/how-to-setup-coderabbit-complete-step-by-step-guide-2026-4115)
- [CodeRabbit FAQ](https://www.coderabbit.ai/faq)
- [CodeRabbit · GitHub Marketplace](https://github.com/marketplace/coderabbitai)
- [Qodo - Free for Open Source Projects - GitHub Marketplace](https://github.com/marketplace/qodo-merge-pro-for-open-source)
- [Setup and Installation — Qodo Documentation](https://docs.qodo.ai/qodo-documentation/qodo-merge/getting-started/setup-and-installation)
