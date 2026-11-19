# Eon — assistant IA local multi-modèles

Eon est un assistant conversationnel en ligne de commande pour Linux. Il s’appuie sur [Ollama](https://ollama.com/) pour exécuter des modèles localement et router les demandes vers des modèles spécialisés. Le projet comprend aussi des outils web, fichiers et terminal, une mémoire locale, un optimiseur et des fonctions optionnelles de perception caméra/micro.

> **Attention :** les outils terminal et fichiers ont des effets réels sur la machine. Le mode `shell` est réglé sur `auto` dans la configuration fournie. Vérifie les permissions dans `config_ia.json` avant le premier lancement et ne donne pas accès à l’assistant à des demandes ou contenus non fiables.

## Fonctionnalités

- Routage des demandes par règles et embeddings Ollama (`bge-m3`), avec rechargement à chaud de `config_ia.json`.
- Modèles spécialisés pour la conversation, la culture générale, le code et l’administration Linux.
- Outils web, fichiers et terminal, chacun contrôlable par permission.
- Mémoire de conversation SQLite, faits durables et résumés épisodiques, conservés localement.
- Optimisation du routage et consolidation de mémoire via `--optimize`.
- Perception caméra et micro facultative (OpenCV, SoundDevice et Faster Whisper).

## Prérequis

- Fedora Linux ou une distribution Linux équivalente.
- Python 3.10 ou plus récent et `pip`.
- [Ollama](https://ollama.com/download) installé et démarré.
- Espace disque pour les modèles Ollama. La mémoire vidéo requise dépend du modèle utilisé ; les modèles 14B et 12B peuvent nécessiter davantage de mémoire ou solliciter le CPU.
- Pour la caméra et le micro : webcam/micro accessibles au système et dépendances système associées.

## Installation sous Fedora

Installe les outils système et les dépendances audio nécessaires :

```bash
sudo dnf install -y python3 python3-pip python3-devel gcc gcc-c++ portaudio-devel
```

Installe Ollama en suivant les instructions officielles. Sur Fedora, le script d’installation Linux peut être utilisé ainsi :

```bash
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable --now ollama
```

Depuis le dossier du projet, crée un environnement Python et installe les paquets listés dans `requirements.txt` :

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Télécharger les modèles IA

Ollama doit être démarré avant de télécharger les modèles. Les noms ci-dessous correspondent aux routes de `config_ia.json` :

```bash
ollama pull gemma3:1b
ollama pull huihui_ai/qwen3-abliterated:8b
ollama pull qwen2.5:7b
ollama pull qwen2.5-coder:7b
ollama pull qwen2.5-coder:14b
ollama pull gemma3:12b-it-qat
ollama pull bge-m3
```

Les modèles peuvent occuper plusieurs dizaines de gigaoctets au total. Pour une installation réduite, télécharge uniquement les modèles que tu gardes dans `config_ia.json`, ainsi que `bge-m3` si le routage par embeddings est activé (`routage.utiliser_embeddings`).

## Lancement

Dans le dossier du projet, avec l’environnement virtuel activé :

```bash
python main.py
```

Pour exécuter uniquement l’optimiseur :

```bash
python main.py --optimize
```

## Commandes interactives

| Commande | Action |
|---|---|
| `@<alias> <message>` | Force un modèle pour une demande, par exemple `@code vérifie ce script`. |
| `/model <alias\|auto>` | Verrouille un modèle ou réactive le routage automatique. |
| `/models` | Affiche les modèles, alias et permissions. |
| `/shell <ask\|auto\|off>` | Règle l’accès au terminal. |
| `/files <ask\|auto\|ro>` | Règle l’accès aux fichiers. |
| `/web <ask\|auto\|off>` | Règle l’accès au Web. |
| `/set <alias> <paramètre> <valeur>` | Modifie à chaud un réglage de modèle ou d’outil. |
| `/add-example <alias> <phrase>` | Ajoute un exemple au routage. |
| `/optimize` | Lance l’optimisation et la consolidation de mémoire. |
| `/history [n]` | Affiche les derniers échanges. |
| `/memory` ou `/forget <clé>` | Consulte ou supprime des faits mémorisés. |
| `/clear`, `/clear session`, `/clear all` | Efface l’historique du jour, de la session ou tout l’historique. |
| `/security <auto\|absent\|off>` | Règle le mode de sécurité caméra. |
| `/enroll [nom]`, `/enroll reset [nom]` | Enregistre ou réinitialise un profil de perception. |

Les alias des modèles sont définis dans `config_ia.json` (par exemple `1b`, `c`, `g`, `code`, `code+` et `12b`). Les réglages de routage, modèles par défaut et permissions se trouvent également dans ce fichier.

## Perception caméra et micro

Les dépendances Python de caméra et de reconnaissance vocale sont incluses dans `requirements.txt`. Les fonctions associées restent facultatives : Eon peut démarrer sans périphérique caméra ou micro. Les modèles ONNX présents dans `models_vision/` servent à certaines fonctions de vision.

## Fichiers et données locales

La configuration active est `config_ia.json`. Les historiques, la base SQLite, le profil et les fichiers de mémoire contiennent potentiellement des données personnelles et sont suivis dans ce dépôt privé. Vérifie leur contenu avant de partager le dépôt ou d’en modifier la visibilité. La clé `.history.key`, les journaux, le cache de routage, les captures, l’environnement virtuel et les fichiers IDE sont exclus par `.gitignore` et restent locaux.

## Structure du projet

| Fichier | Rôle |
|---|---|
| `main.py` | Boucle interactive, routage des requêtes et appels aux outils. |
| `router.py` | Routage par règles/embeddings et rechargement de configuration. |
| `config_ia.json` | Modèles, règles, paramètres et permissions. |
| `file_tools.py`, `shell_tools.py`, `web_tools.py` | Outils fichiers, terminal et Web. |
| `memory_manager.py` | Historique, mémoire et stockage SQLite. |
| `optimizer.py` | Optimisation du routage et consolidation de mémoire. |
| `perception_eon.py` | Fonctions de perception caméra et micro. |
| `terminal_panels.py` | Affichage du terminal et des événements. |
| `models_vision/` | Modèles ONNX utilisés par la perception visuelle. |
