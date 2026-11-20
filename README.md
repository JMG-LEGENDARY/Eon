# Eon — Assistant IA local multimodal et multi-modèles

Eon est un orchestrateur d'IA local pour Linux doté d'une interface terminal interactive (TUI), d'un routage sémantique hybride, d'une mémoire hiérarchique à trois niveaux, d'une synthèse vocale clonée et d'un pipeline de perception caméra/micro en temps réel.

> **Attention :** les outils terminal, fichiers et web agissent directement sur la machine. Par défaut dans `config_ia.json`, les modes `web_mode`, `file_mode` et `shell_mode` sont réglés sur `ask` (confirmation explicite requise avant chaque exécution sensible). Vérifie tes permissions avant le premier lancement et ne soumets pas de contenus non fiables à l'assistant.

---

## Fonctionnalités principales

- **Routage sémantique hybride (`router.py`)** : sélection automatique du modèle optimal via règles regex, embeddings locaux (`bge-m3`), superviseur léger (`qwen3:0.6b`) et rechargement à chaud de `config_ia.json`.
- **Mémoire hiérarchique à 3 niveaux (`context_compressor.py` & `memory_manager.py`)** : fenêtre active en VRAM, compression glissante en RAM et stockage persistant chiffré sous SQLite (`historique_eon.db`).
- **Pré-chauffage prédictif au regard** : détection du regard utilisateur via webcam pour pré-charger le modèle cible en VRAM avant même la fin de la requête.
- **Synthèse vocale locale (`tts_client.py` & `IA_voice_lab/tts_daemon.py`)** : démon IPC sur socket Unix (`/tmp/eon_tts.sock`) propulsé par Chatterbox TTS avec clonage vocal (`ma_voix_ref.wav`), streaming phrase par phrase et interruption instantanée (*barge-in* ou touche `Échap`).
- **Perception audio et vision (`perception_eon.py`)** : transcription vocale continue via Faster-Whisper, reconnaissance faciale biométrique (YuNet / SFace), détection d'objets (YOLO11n) et mode surveillance avec alertes.
- **Auto-optimisation (`optimizer.py`)** : affinage des seuils de routage et consolidation de la mémoire en arrière-plan sous contrôle thermique (`/optimize`).

---

## Prérequis

- **Système** : Fedora Linux (ou distribution Linux équivalente avec PipeWire/PulseAudio).
- **Python** :
  - Python 3.13 ou 3.14 pour le cœur d'Eon.
  - Python 3.12 dédié pour l'environnement vocal Chatterbox (`IA_voice_lab/.venv`).
- **Moteur LLM** : [Ollama](https://ollama.com/download) installé et actif.
- **Matériel** : GPU NVIDIA (CUDA) ou AMD (ROCm) recommandé (8 Go de VRAM minimum conseillés pour les modèles 7B/8B ; déchargement automatique entre modèles lourds 12B/14B intégré).
- **Périphériques (optionnels)** : webcam et microphone accessibles au système pour la perception multimodale.

---

## Installation sous Fedora

### 1. Dépendances système et Ollama

```bash
sudo dnf install -y python3 python3.12 python3-pip python3-devel gcc gcc-c++ portaudio-devel
curl -fsSL [https://ollama.com/install.sh](https://ollama.com/install.sh) | sh
sudo systemctl enable --now ollama
```

### 2. Environnement principal d'Eon

```bash
cd ~/Documents/MOI/Eon
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
deactivate
```

### 3. Environnement vocal Chatterbox (`IA_voice_lab`)

Le démon vocal tourne dans un environnement virtuel Python 3.12 isolé à l'intérieur du dépôt :

```bash
cd ~/Documents/MOI/Eon/IA_voice_lab
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
bash setup_voice_lab.sh
deactivate
```

*(Optionnel)* Place un échantillon audio propre de 8 à 15 secondes nommé `ma_voix_ref.wav` dans `IA_voice_lab/` pour activer le clonage vocal.

### 4. Téléchargement des modèles Ollama

Ollama doit être démarré avant de télécharger les modèles correspondant aux routes de `config_ia.json` :

```bash
ollama pull gemma3:1b
ollama pull qwen3:0.6b
ollama pull huihui_ai/qwen3-abliterated:8b
ollama pull qwen2.5:7b
ollama pull qwen2.5-coder:7b
ollama pull qwen2.5-coder:14b
ollama pull gemma3:12b-it-qat
ollama pull bge-m3
```

---

## Compatibilité GPU AMD (ROCm)

Eon peut tourner sur les cartes graphiques **AMD Radeon (architectures RDNA 2 / RDNA 3 — séries RX 6000, RX 7000 et iGPU Radeon 780M/890M)** sous Linux via la pile **ROCm**. Voici la configuration des trois composants :

### 1. Ollama (LLM et embeddings `bge-m3`)

Installe les utilitaires et bibliothèques ROCm sous Fedora :

```bash
sudo dnf install -y rocm-smi rocminfo rocm-hip
```

Si ton GPU AMD grand public n'est pas supporté nativement par défaut (ex. RX 6700 XT, RX 7600 ou iGPU intégré), force la variable `HSA_OVERRIDE_GFX_VERSION` dans le service `systemd` d'Ollama :

```bash
sudo systemctl edit ollama
```

Ajoute le bloc suivant selon la génération de ta puce graphique :

```ini
[Service]
# RDNA 2 (ex: RX 6600 / 6700 / 6800) -> 10.3.0
# RDNA 3 (ex: RX 7600 / 7700 / 7800 / 7900 / Radeon 780M) -> 11.0.0
Environment="HSA_OVERRIDE_GFX_VERSION=11.0.0"
```

Recharge et redémarre le service :

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

### 2. Synthèse vocale Chatterbox (`IA_voice_lab`)

PyTorch expose ROCm via l'interface HIP (`torch.cuda.is_available()` renvoie `True` sur GPU AMD). Dans l'environnement virtuel de `IA_voice_lab`, remplace PyTorch CUDA par la version ROCm :

```bash
cd ~/Documents/MOI/Eon/IA_voice_lab
source .venv/bin/activate
pip uninstall -y torch torchaudio torchvision
pip install torch torchaudio --index-url [https://download.pytorch.org/whl/rocm6.2](https://download.pytorch.org/whl/rocm6.2)
deactivate
```

*Alternative économie de VRAM :* sur un GPU AMD disposant de 8 Go de VRAM ou moins, définis `"tts_device": "cpu"` dans la section `"tts"` de `config_ia.json` afin de réserver 100 % de la mémoire vidéo aux modèles Ollama.

### 3. Perception audio et vision (`Faster-Whisper` & `ONNX`)

- **Faster-Whisper** s'exécute sur CPU en quantification `int8` (`compute_type="int8"`), rapide sur les processeurs AMD Ryzen sans consommer de VRAM.
- **Vision OpenCV (YuNet, SFace, YOLO11n)** fonctionne sur CPU via `onnxruntime` standard sans réglage supplémentaire.

---

## Lancement

Active l'environnement principal et lance Eon (le démon vocal `IA_voice_lab/tts_daemon.py` démarre automatiquement en arrière-plan dès l'ouverture) :

```bash
cd ~/Documents/MOI/Eon
source .venv/bin/activate
python main.py
```

Pour démarrer ou diagnostiquer manuellement le démon vocal dans un terminal séparé :

```bash
~/Documents/MOI/Eon/IA_voice_lab/.venv/bin/python ~/Documents/MOI/Eon/IA_voice_lab/tts_daemon.py
```

---

## Commandes interactives

| Commande                          | Action                                                                                                     |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `@<alias> <message>`            | Force un modèle sur un seul message (ex. `@code vérifie ce script`, `@c`, `@12b`).                 |
| `/model <alias\|auto>`           | Verrouille un modèle pour la session ou réactive le routage automatique.                                 |
| `/models`                       | Liste les modèles configurés, alias, rôles et permissions d'outils.                                     |
| `/tts [status\|test\|stop]`       | Vérifie l'état du démon Chatterbox, lance un test vocal ou coupe la parole en cours.                    |
| `/web <auto\|ask\|off>`           | Règle la permission d'accès aux outils de recherche Web.                                                 |
| `/file <auto\|ask\|off>`          | Règle la permission d'écriture/modification de fichiers locaux.                                          |
| `/shell <auto\|ask\|off>`         | Règle la permission d'exécution de commandes terminal.                                                   |
| `/history [N]`                  | Affiche les `N` derniers échanges et les appels d'outils de la session.                                 |
| `/export [chemin]`              | Exporte l'historique réel vers un fichier texte (défaut : `~/Téléchargements/conversation_eon.txt`). |
| `/enroll [nom\|reset\|del <nom>]` | Enregistre un visage autorisé ou réinitialise la base biométrique.                                      |
| `/secu [auto\|absent\|off]`       | Règle le mode de surveillance caméra.                                                                    |
| `/capteurs`                     | Affiche l'état temps réel de la caméra, du regard et du microphone.                                     |
| `/forget` ou `/reset`         | Purge le contexte actif en RAM.                                                                            |
| `/reload`                       | Recharge à chaud `config_ia.json` et `entrainement_routage.json`.                                     |
| `/optimize [min]`               | Lance l'optimisation du routage en arrière-plan sous contrôle thermique (défaut : 55 min).              |
| `/help`                         | Affiche l'aide complète des commandes.                                                                    |
| `/quit` ou `/exit`            | Arrête proprement le démon vocal, les capteurs et Eon.                                                   |

Les alias des modèles sont définis dans `config_ia.json` (`1b`, `c`, `g`, `code`, `code+`, `12b`).

---

## Perception caméra et micro

Les dépendances Python de caméra et de reconnaissance vocale sont incluses dans `requirements.txt`. Ces fonctions restent facultatives : Eon démarre normalement même sans périphérique caméra ou microphone connecté. Les modèles ONNX présents dans `models_vision/` assurent la détection faciale (YuNet), la reconnaissance biométrique (SFace) et la détection d'objets (YOLO11n).

---

## Fichiers et données locales

La configuration active est `config_ia.json`. Les environnements virtuels (`.venv/`, `IA_voice_lab/.venv/`), la base SQLite chiffrée (`historique_eon.db`), la clé `.history.key`, les profils biométriques (`profil_eon.json`), les journaux (`logs/`), le cache de routage (`routes_cache.json`), les captures (`captures/`) et les échantillons audio personnels (`*.wav`) sont exclus du suivi Git via `.gitignore` afin de garder le dépôt propre et confidentiel.

---

## Structure du projet

| Chemin                                                  | Rôle                                                                                                            |
| ------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `main.py`                                             | Orchestrateur principal, boucle autonome d'outils, gestion VRAM et commandes slash.                              |
| `router.py`                                           | Routage sémantique (regex + embeddings `bge-m3` + superviseur IA) et analyse d'intentions.                    |
| `config_ia.json`                                      | Configuration des routes, seuils, alias, règles regex, paramètres TTS et permissions.                          |
| `context_compressor.py`                               | Gestionnaire de contexte à 3 niveaux (VRAM / RAM / Disque).                                                     |
| `memory_manager.py`                                   | Base SQLite chiffrée, faits durables et recherche dans l'historique.                                            |
| `perception_eon.py`                                   | Pipeline temps réel webcam (YuNet, SFace, YOLO11n) et audio (Faster-Whisper, anti-écho).                       |
| `tts_client.py`                                       | Client IPC Unix vers Chatterbox avec nettoyage Markdown et auto-démarrage du démon.                            |
| `IA_voice_lab/`                                       | Environnement Python 3.12 dédié au démon vocal Chatterbox (`tts_daemon.py`) et échantillon de référence. |
| `terminal_panels.py`                                  | Interface TUI multi-panneaux (chat, logs routeur, vision, audio).                                                |
| `file_tools.py`, `shell_tools.py`, `web_tools.py` | Modules d'outils système, fichiers et recherche Internet.                                                       |
| `optimizer.py`                                        | Optimisation autonome des seuils et consolidation mémoire.                                                      |
| `models_vision/`                                      | Modèles ONNX pour la détection et reconnaissance faciale / objets.                                             |
