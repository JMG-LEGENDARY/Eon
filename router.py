import copy
import functools
import hashlib
import json
import math
import os
import re
import shutil
import ollama
from memory_manager import BASE_DIR

CONFIG_FILE = BASE_DIR / "config_ia.json"
CONFIG_BACKUP_FILE = BASE_DIR / "config_ia.bak.json"
TRAINING_FILE = BASE_DIR / "entrainement_routage.json"
CACHE_FILE = BASE_DIR / "routes_cache.json"

ENTRAINEMENT_DEFAUT: dict = {
    "routes": {
        "phatique": [
            "saluer quelqu'un de manière brève et informelle",
            "dire bonjour, bonsoir ou coucou",
            "remercier brièvement pour une réponse",
            "acquiescer simplement sans poser de nouvelle question",
            "prendre congé ou dire au revoir",
        ],
        "vision_multimodale": [
            "demander ce que la caméra ou la webcam voit en ce moment",
            "demander de décrire la scène, les personnes ou les objets devant l'écran",
            "demander la couleur d'un vêtement ou un détail physique visible",
            "demander de lire un texte ou un objet montré à la caméra",
            "vérifier si les capteurs visuels perçoivent quelque chose",
        ],
        "general_et_web": [
            "poser une question de culture générale ou demander une explication théorique",
            "demander les actualités récentes ou ce qu'il se passe en ce moment",
            "demander quel est le dernier modèle sorti d'un produit ou d'une technologie",
            "demander où joue un sportif ou quelle est la situation actuelle d'une personnalité",
            "demander de vérifier une information factuelle et datée sur des sources fiables",
            "demander ce que l'assistant sait sur l'utilisateur",
        ],
        "code_courant": [
            "demander d'exécuter une action ou une commande sur le système Linux",
            "demander de lire, créer, modifier ou copier un fichier sur le disque",
            "demander d'écrire ou corriger un script Python, Rust, C ou Bash",
            "demander de diagnostiquer une erreur système, un paquet ou un service",
            "demander de vérifier le contenu d'un fichier qui vient d'être créé",
        ],
        "architecture_lourde": [
            "concevoir l'architecture complète d'une application complexe",
            "refactoriser un projet entier sur plusieurs fichiers",
            "résoudre un problème avancé de concurrence, multithreading ou performance bas niveau",
            "auditer en profondeur la structure logicielle d'un système distribué",
        ],
        "creatif_discussion": [
            "discuter librement de tout et de rien sur un ton détendu et humain",
            "demander une blague, une anecdote ou une histoire immersive",
            "faire du brainstorming ouvert sans contrainte technique",
            "demander un avis personnel ou philosophique sur un sujet",
        ],
    },
    "intentions": {
        "web_obligatoire": [
            "demander des informations récentes ou d'actualité",
            "demander le dernier modèle ou la dernière version sortie d'un appareil",
            "demander le club actuel, le poste ou la situation présente d'une personne",
            "demander de vérifier des faits récents sur des sources externes",
            "contester une information datée et exiger une vérification à jour",
        ],
        "vision_camera": [
            "demander ce qui est visible devant la caméra",
            "demander de décrire les personnes ou objets présents dans la pièce",
            "accepter une proposition de description visuelle de la scène",
        ],
        "export_historique": [
            "demander une copie ou une sauvegarde de la conversation dans un fichier",
            "exporter l'historique des échanges vers un dossier personnel",
        ],
        "purge_contexte": [
            "demander d'oublier la conversation précédente et de repartir de zéro",
            "ignorer les réponses précédentes qui contenaient des erreurs",
        ],
        "insatisfaction": [
            "dire que le modèle choisi n'est pas le bon",
            "dire que la réponse est fausse, obsolète ou hors sujet",
            "reprocher à l'assistant de répéter la même chose sans vérifier",
            "signaler que le contenu d'un fichier ou d'une réponse est absurde ou vide",
        ],
    },
    "consentement": {
        "oui": [
            "accepter et donner son accord pour exécuter l'action",
            "dire oui, d'accord, vas-y ou fais-le",
            "confirmer et autoriser l'opération demandée",
        ],
        "non": [
            "refuser et annuler l'action proposée",
            "dire non, laisse tomber ou ne le fais pas",
            "interdire l'exécution de l'opération",
        ],
    },
}

CONFIG_DEFAUT: dict = {
    "web_mode": "auto",
    "file_mode": "ask",
    "shell_mode": "auto",
    "modele_optimiseur": "qwen2.5-coder:14B",
    "route_defaut": "qwen2.5:7B",
    "regles_apprises": [
        "Si un résultat de recherche web contredit tes connaissances internes ou tes réponses précédentes, le résultat web a toujours raison.",
        "Ne simule jamais l'exécution d'une commande système ou le contenu d'un fichier : utilise les outils réels.",
    ],
    "routage": {
        "utiliser_embeddings": True,
        "embed_model": "bge-m3",
        "superviseur_model": "qwen3:0.6b",
        "utiliser_superviseur_ia": True,
        "seuils": {
            "confiance_directe": 0.63,
            "marge_directe": 0.05,
            "intention_outil": 0.58,
            "insatisfaction": 0.58,
        },
    },
    "routes": {
        "gemma3:1b": {
            "alias": "1b",
            "role": "phatique",
            "supports_tools": False,
            "expertise": "Salutations courtes et accusés de réception.",
            "temperature": 0.6,
            "num_ctx": 1024,
            "num_predict": 60,
            "web_tools": False,
            "file_tools": False,
            "shell_tools": False,
        },
        "huihui_ai/qwen3-abliterated:8b": {
            "alias": "c",
            "role": "creatif_discussion",
            "expertise": "Discussion libre, humour, brainstorming, ton naturel sans filtre ni rigidité.",
            "temperature": 0.75,
            "num_ctx": 4096,
            "num_predict": 512,
            "web_tools": True,
            "file_tools": False,
            "shell_tools": False,
        },
        "qwen2.5:7B": {
            "alias": "g",
            "role": "general_et_web",
            "expertise": "Culture générale, actualités récentes via recherche web, explications théoriques, synthèse et mémoire utilisateur.",
            "temperature": 0.2,
            "num_ctx": 8192,
            "num_predict": 768,
            "web_tools": True,
            "file_tools": False,
            "shell_tools": False,
        },
        "qwen2.5-coder:7B": {
            "alias": "code",
            "role": "code_courant",
            "expertise": "Administration système Linux/Fedora, commandes terminal bash, lecture/édition de fichiers locaux, scripts Python/Rust/C.",
            "temperature": 0.15,
            "num_ctx": 8192,
            "num_predict": 1024,
            "web_tools": True,
            "file_tools": True,
            "shell_tools": True,
        },
        "qwen2.5-coder:14B": {
            "alias": "code+",
            "role": "architecture_lourde",
            "expertise": "Refactoring complexe multi-fichiers, architecture logicielle lourde et debugging bas niveau.",
            "temperature": 0.15,
            "num_ctx": 4096,
            "num_predict": 1536,
            "web_tools": True,
            "file_tools": True,
            "shell_tools": True,
        },
        "gemma3:12b-it-qat": {
            "alias": "12b",
            "role": "vision_multimodale",
            "expertise": "Analyse visuelle complexe de scènes caméra (intrusion, contexte, lecture d'image).",
            "temperature": 0.2,
            "num_ctx": 4096,
            "num_predict": 512,
            "web_tools": False,
            "file_tools": False,
            "shell_tools": False,
        },
    },
}


def _normaliser_l2(vec: list[float]) -> tuple[float, ...]:
    norme = math.hypot(*vec)
    if norme <= 0.0:
        return tuple(vec)
    return tuple(x / norme for x in vec)


def _produit_scalaire(v1: tuple[float, ...] | list[float], v2: tuple[float, ...] | list[float]) -> float:
    return sum(a * b for a, b in zip(v1, v2))


def _extraire_json_propre(texte: str) -> dict | None:
    """Extrait un objet JSON valide même si le mini-LLM entoure sa sortie de balises <think> ou markdown."""
    if "</think>" in texte:
        texte = texte.split("</think>", 1)[1]
    texte = texte.strip()
    try:
        obj = json.loads(texte)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", texte, flags=re.S)
        if match:
            try:
                obj = json.loads(match.group(0))
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                pass
    return None


@functools.lru_cache(maxsize=64)
def _embed_texte_cache(embed_model: str, texte: str) -> tuple[float, ...]:
    """Vectorise un texte sur CPU (0 Mo VRAM) avec mise en cache LRU."""
    raw_vec = ollama.embed(model=embed_model, input=texte, options={"num_gpu": 0}, keep_alive=-1)["embeddings"][0]
    return _normaliser_l2(raw_vec)


class RouteurAutonome:
    def __init__(self) -> None:
        self.config: dict = {}
        self.dataset: dict = {}
        self._config_mtime: float = 0.0
        self._training_mtime: float = 0.0
        self.embeddings_roles: dict[str, list[list[float]]] = {}
        self.embeddings_intentions: dict[str, list[list[float]]] = {}
        self.embeddings_consentement: dict[str, list[list[float]]] = {}
        self.charger(silencieux=False)

    def verifier_hot_reload(self) -> bool:
        try:
            mtime_cfg = CONFIG_FILE.stat().st_mtime if CONFIG_FILE.exists() else 0.0
            mtime_trn = TRAINING_FILE.stat().st_mtime if TRAINING_FILE.exists() else 0.0
        except OSError:
            return False
        if mtime_cfg != self._config_mtime or mtime_trn != self._training_mtime:
            self.charger(silencieux=True)
            return True
        return False

    def sauvegarder_config(self) -> None:
        tmp = CONFIG_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.config, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, CONFIG_FILE)
        self._config_mtime = CONFIG_FILE.stat().st_mtime

    def sauvegarder_dataset(self) -> None:
        tmp = TRAINING_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.dataset, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, TRAINING_FILE)
        self._training_mtime = TRAINING_FILE.stat().st_mtime

    def _normaliser_config(self, cfg: dict) -> tuple[dict, bool]:
        modifie = False
        defaut = copy.deepcopy(CONFIG_DEFAUT)

        if not isinstance(cfg.get("routage"), dict):
            cfg["routage"] = {}
            modifie = True

        routage = cfg["routage"]
        if "embed_model" in cfg:
            routage.setdefault("embed_model", cfg.pop("embed_model"))
            modifie = True
        if "seuils" in cfg:
            routage.setdefault("seuils", cfg.pop("seuils"))
            modifie = True

        for cle, val in defaut.items():
            if cfg.get(cle) is None:
                cfg[cle] = val
                modifie = True

        for cle_r, val_r in defaut["routage"].items():
            if routage.get(cle_r) is None:
                routage[cle_r] = val_r
                modifie = True

        if isinstance(routage.get("seuils"), dict):
            for k_s, v_s in defaut["routage"]["seuils"].items():
                if routage["seuils"].get(k_s) is None:
                    routage["seuils"][k_s] = v_s
                    modifie = True

        for modele, r_cfg in cfg.get("routes", {}).items():
            ref = defaut["routes"].get(modele, {})
            if r_cfg.get("num_predict") is None:
                r_cfg["num_predict"] = ref.get("num_predict", 768)
                modifie = True
            if r_cfg.get("num_ctx") is None:
                r_cfg["num_ctx"] = ref.get("num_ctx", 4096)
                modifie = True
            if r_cfg.get("temperature") is None:
                r_cfg["temperature"] = ref.get("temperature", 0.2)
                modifie = True
            if "shell_tools" not in r_cfg:
                r_cfg["shell_tools"] = r_cfg.get("role") in ("code_courant", "architecture_lourde")
                modifie = True
            if "expertise" not in r_cfg and modele in defaut["routes"]:
                r_cfg["expertise"] = defaut["routes"][modele]["expertise"]
                modifie = True

        return cfg, modifie

    def _restaurer_ou_generer_config(self) -> dict:
        if CONFIG_BACKUP_FILE.exists():
            try:
                bak = json.loads(CONFIG_BACKUP_FILE.read_text(encoding="utf-8"))
                if isinstance(bak, dict) and bak.get("routes"):
                    shutil.copy2(CONFIG_BACKUP_FILE, CONFIG_FILE)
                    return bak
            except (json.JSONDecodeError, OSError):
                pass
        self.config = copy.deepcopy(CONFIG_DEFAUT)
        self.sauvegarder_config()
        return self.config

    def _vectoriser_liste(self, embed_model: str, textes: list[str]) -> list[list[float]]:
        if not textes:
            return []
        vecs = ollama.embed(model=embed_model, input=textes, options={"num_gpu": 0})["embeddings"]
        return [list(_normaliser_l2(v)) for v in vecs]

    def charger(self, silencieux: bool = False) -> None:
        if not CONFIG_FILE.exists() or CONFIG_FILE.stat().st_size == 0:
            nouvelle_cfg = self._restaurer_ou_generer_config() if not self.config else self.config
        else:
            try:
                nouvelle_cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                if not isinstance(nouvelle_cfg, dict) or not nouvelle_cfg.get("routes"):
                    raise ValueError("clé 'routes' absente")
            except (json.JSONDecodeError, ValueError, OSError):
                nouvelle_cfg = self.config if self.config else self._restaurer_ou_generer_config()

        nouvelle_cfg, a_ete_migree = self._normaliser_config(nouvelle_cfg)
        self.config = nouvelle_cfg
        if a_ete_migree:
            self.sauvegarder_config()
        else:
            self._config_mtime = CONFIG_FILE.stat().st_mtime

        dataset_modifie = False
        if not TRAINING_FILE.exists() or TRAINING_FILE.stat().st_size == 0:
            self.dataset = copy.deepcopy(ENTRAINEMENT_DEFAUT)
            dataset_modifie = True
        else:
            try:
                self.dataset = json.loads(TRAINING_FILE.read_text(encoding="utf-8"))
                if not isinstance(self.dataset, dict):
                    raise ValueError("dataset invalide")
            except (json.JSONDecodeError, ValueError, OSError):
                self.dataset = copy.deepcopy(ENTRAINEMENT_DEFAUT)
                dataset_modifie = True

        for section, contenu_defaut in ENTRAINEMENT_DEFAUT.items():
            sec_dict = self.dataset.setdefault(section, {})
            for cle, phrases in contenu_defaut.items():
                if cle not in sec_dict or not sec_dict[cle]:
                    sec_dict[cle] = list(phrases)
                    dataset_modifie = True

        for modele, r_cfg in self.config.get("routes", {}).items():
            role = r_cfg.get("role", "")
            anciens_ex = r_cfg.get("examples", [])
            if role and isinstance(anciens_ex, list):
                cible_liste = self.dataset.setdefault("routes", {}).setdefault(role, [])
                for ex in anciens_ex:
                    if isinstance(ex, str) and ex.strip() and ex not in cible_liste:
                        cible_liste.append(ex.strip())
                        dataset_modifie = True

        anciens_signaux = self.config.get("signaux", {})
        if isinstance(anciens_signaux, dict):
            for ex in anciens_signaux.get("insatisfaction", []):
                cible_ins = self.dataset.setdefault("intentions", {}).setdefault("insatisfaction", [])
                if isinstance(ex, str) and ex.strip() and ex not in cible_ins:
                    cible_ins.append(ex.strip())
                    dataset_modifie = True

        if dataset_modifie:
            self.sauvegarder_dataset()
        elif TRAINING_FILE.exists():
            self._training_mtime = TRAINING_FILE.stat().st_mtime

        routage_cfg = self.config.get("routage", {})
        if not routage_cfg.get("utiliser_embeddings", True):
            self.embeddings_roles = {}
            self.embeddings_intentions = {}
            self.embeddings_consentement = {}
            return

        embed_model = routage_cfg.get("embed_model", "bge-m3")
        signature = hashlib.sha256(
            json.dumps({"model": embed_model, "dataset": self.dataset}, sort_keys=True).encode("utf-8")
        ).hexdigest()

        if CACHE_FILE.exists():
            try:
                cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
                if cache.get("signature") == signature:
                    self.embeddings_roles = cache["embeddings_roles"]
                    self.embeddings_intentions = cache["embeddings_intentions"]
                    self.embeddings_consentement = cache["embeddings_consentement"]
                    return
            except (json.JSONDecodeError, KeyError, OSError):
                pass

        self.embeddings_roles = {
            role: self._vectoriser_liste(embed_model, phrases)
            for role, phrases in self.dataset.get("routes", {}).items()
            if phrases
        }
        self.embeddings_intentions = {
            intent: self._vectoriser_liste(embed_model, phrases)
            for intent, phrases in self.dataset.get("intentions", {}).items()
            if phrases
        }
        self.embeddings_consentement = {
            "OUI": self._vectoriser_liste(embed_model, self.dataset.get("consentement", {}).get("oui", [])),
            "NON": self._vectoriser_liste(embed_model, self.dataset.get("consentement", {}).get("non", [])),
        }

        tmp = CACHE_FILE.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({
                "signature": signature,
                "embeddings_roles": self.embeddings_roles,
                "embeddings_intentions": self.embeddings_intentions,
                "embeddings_consentement": self.embeddings_consentement,
            }),
            encoding="utf-8",
        )
        os.replace(tmp, CACHE_FILE)

    def resoudre_modele(self, identifiant: str) -> str | None:
        cible = identifiant.strip().lower()
        for nom_modele, cfg in self.config.get("routes", {}).items():
            if nom_modele.lower() == cible or str(cfg.get("alias", "")).lower() == cible:
                return nom_modele
        for nom_modele in self.config.get("routes", {}):
            if cible in nom_modele.lower():
                return nom_modele
        return None

    def modele_par_role(self, role: str) -> str:
        routes = self.config.get("routes", {})
        for modele, cfg in routes.items():
            if cfg.get("role") == role:
                return modele
        return self.config.get("route_defaut", next(iter(routes), "qwen2.5:7B"))

    @staticmethod
    def _score_top_k(vec_requete: tuple[float, ...], vecs_ref: list[list[float]], k: int = 3) -> float:
        if not vecs_ref:
            return 0.0
        sims = sorted((_produit_scalaire(vec_requete, v) for v in vecs_ref), reverse=True)
        top = sims[: min(k, len(sims))]
        return sum(top) / len(top)

    def _appeler_superviseur_cpu(
        self,
        prompt_systeme: str,
        prompt_utilisateur: str,
        max_tokens: int = 60,
        format_json: bool = False,
    ) -> str:
        """Exécute le mini-modèle superviseur sur CPU (0 Mo VRAM)."""
        superviseur = self.config.get("routage", {}).get("superviseur_model", "qwen3:0.6b")
        kwargs: dict = {
            "model": superviseur,
            "messages": [
                {"role": "system", "content": prompt_systeme},
                {"role": "user", "content": prompt_utilisateur},
            ],
            "options": {"num_gpu": 0, "temperature": 0.0, "num_predict": max_tokens, "num_ctx": 1536},
            "keep_alive": -1,
        }
        if format_json:
            kwargs["format"] = "json"

        try:
            rep = ollama.chat(**kwargs, think=False)
        except TypeError:
            rep = ollama.chat(**kwargs)

        contenu = rep.get("message", {}).get("content", "").strip()
        if "</think>" in contenu:
            contenu = contenu.split("</think>", 1)[1].strip()
        return contenu

    def normaliser_intention_vocale(
        self,
        texte_voix: str,
        dernier_prompt: str = "",
        derniere_reponse: str = "",
    ) -> str:
        """Corrige les homophones et déformations phonétiques Whisper avant routage ou recherche Web."""
        brut = texte_voix.strip()
        if not brut or len(brut.split()) <= 2:
            return brut
        if not self.config.get("routage", {}).get("utiliser_superviseur_ia", True):
            return brut

        prompt_sys = (
            "Tu es un correcteur orthographique et phonétique pour reconnaissance vocale française (Whisper). "
            "Corrige UNIQUEMENT les fautes d'homophones, noms propres ou termes techniques mal transcrits "
            "(ex: 'un test' transcrit 'arfest', 'docker' transcrit 'doquer', 'python' transcrit 'piton'). "
            "RÈGLES STRICTES : "
            "1. Ne réponds JAMAIS à la question. "
            "2. Ne change pas le sens ni la structure de la demande. "
            "3. Si la phrase est déjà compréhensible, renvoie-la exactement à l'identique. "
            "4. Renvoie UNIQUEMENT la phrase corrigée sur une seule ligne, sans guillemets ni commentaire."
        )
        contexte = ""
        if dernier_prompt:
            contexte += f"Contexte récent utilisateur : {dernier_prompt[:120]}\n"
        if derniere_reponse:
            contexte += f"Contexte récent assistant : {derniere_reponse[:140]}\n"
        prompt_usr = f"{contexte}Transcription vocale brute : {brut}"

        try:
            corrige = self._appeler_superviseur_cpu(prompt_sys, prompt_usr, max_tokens=90, format_json=False)
            corrige = corrige.strip().strip("\"'")
            # Garde-fou contre toute dérive du mini-modèle
            if (
                not corrige
                or "\n" in corrige
                or len(corrige) > int(len(brut) * 1.6) + 25
                or corrige.lower().startswith(("bonjour", "je ", "voici", "bien sûr"))
            ):
                return brut
            return corrige
        except Exception:
            return brut

    def _construire_requete_contextualisee(
        self,
        message: str,
        dernier_prompt: str | None = None,
        derniere_reponse: str | None = None,
    ) -> str:
        texte = message.strip()
        if texte.startswith(("Contexte en cours :", "Sujet repris (archive) :")):
            return texte
        if len(texte.split()) <= 7 and (dernier_prompt or derniere_reponse):
            ctx = []
            if dernier_prompt:
                ctx.append(f"Demande précédente : {dernier_prompt[:140]}")
            if derniere_reponse:
                ctx.append(f"Réponse précédente : {derniere_reponse[:160]}")
            return "\n".join(ctx) + f"\nNouvelle demande : {texte}"
        return texte

    def analyser_intentions(
        self,
        message: str,
        dernier_prompt: str | None = None,
        derniere_reponse: str | None = None,
    ) -> dict[str, float]:
        """Évalue sémantiquement les intentions latentes (web, vision, export, purge, insatisfaction)."""
        if not self.embeddings_intentions:
            return {}
        embed_model = self.config.get("routage", {}).get("embed_model", "bge-m3")
        requete_ctx = self._construire_requete_contextualisee(message, dernier_prompt, derniere_reponse)
        vec_ctx = _embed_texte_cache(embed_model, requete_ctx)

        scores = {
            intent: round(self._score_top_k(vec_ctx, vecs, k=2), 3)
            for intent, vecs in self.embeddings_intentions.items()
        }

        if requete_ctx != message.strip():
            vec_brut = _embed_texte_cache(embed_model, message.strip())
            for intent, vecs in self.embeddings_intentions.items():
                score_brut = round(self._score_top_k(vec_brut, vecs, k=2), 3)
                if score_brut > scores.get(intent, 0.0):
                    scores[intent] = score_brut

        return scores

    def interpreter_consentement(self, reponse_utilisateur: str) -> tuple[bool, str]:
        texte = reponse_utilisateur.strip()
        if not texte:
            return False, "vide -> NON"

        routage_cfg = self.config.get("routage", {})
        if routage_cfg.get("utiliser_embeddings", True) and self.embeddings_consentement:
            embed_model = routage_cfg.get("embed_model", "bge-m3")
            vec_rep = _embed_texte_cache(embed_model, texte)
            score_oui = self._score_top_k(vec_rep, self.embeddings_consentement.get("OUI", []), k=2)
            score_non = self._score_top_k(vec_rep, self.embeddings_consentement.get("NON", []), k=2)
            ecart = score_oui - score_non
            if ecart >= 0.05 and score_oui >= 0.56:
                return True, f"sémantique:OUI ({score_oui:.2f} vs {score_non:.2f})"
            if -ecart >= 0.05 and score_non >= 0.56:
                return False, f"sémantique:NON ({score_non:.2f} vs {score_oui:.2f})"

        try:
            verdict = self._appeler_superviseur_cpu(
                "Classifieur binaire. L'utilisateur autorise-t-il l'action ? Réponds UNIQUEMENT par OUI ou NON.",
                texte,
                max_tokens=4,
            ).upper()
            return ("OUI" in verdict and "NON" not in verdict), f"superviseur:{verdict}"
        except Exception:
            return False, "fallback:NON"

    def evaluer_insatisfaction(self, reaction: str, dernier_prompt: str, derniere_reponse: str) -> tuple[bool, str]:
        texte = reaction.strip()
        if not texte or len(texte) > 250 or not self.embeddings_intentions:
            return False, ""

        scores = self.analyser_intentions(texte, dernier_prompt, derniere_reponse)
        sim = scores.get("insatisfaction", 0.0)
        seuil = float(self.config.get("routage", {}).get("seuils", {}).get("insatisfaction") or 0.58)
        if sim < seuil:
            return False, ""

        try:
            verdict = self._appeler_superviseur_cpu(
                "Détermine si le NOUVEAU MESSAGE exprime un rejet, une correction ou une insatisfaction "
                "vis-à-vis de la RÉPONSE PRÉCÉDENTE. Réponds UNIQUEMENT par ECHEC ou OK.",
                f"DEMANDE : {dernier_prompt[:250]}\nRÉPONSE IA : {derniere_reponse[:350]}\nNOUVEAU MESSAGE : {texte}",
                max_tokens=4,
            ).upper()
            if "ECHEC" in verdict:
                return True, f"superviseur:ECHEC (sim={sim:.2f})"
        except Exception:
            pass
        return False, ""

    def _arbitrer_par_superviseur_ia(
        self,
        requete_ctx: str,
        candidats: list[tuple[str, float]],
        dernier_role: str | None,
        rejet_precedent: bool = False,
    ) -> tuple[str, str]:
        """Vérifie le contexte ultra-court et oriente vers le rôle le plus pertinent."""
        descriptions = [
            f"- {cfg.get('role')} : {cfg.get('expertise')}"
            for cfg in self.config.get("routes", {}).values()
        ]
        consigne_rejet = (
            f"\nATTENTION : L'utilisateur vient de rejeter la réponse du rôle `{dernier_role}`. Choisis un rôle mieux adapté."
            if rejet_precedent and dernier_role
            else ""
        )

        prompt_sys = (
            "Tu es le routeur intelligent d'Eon. Analyse le contexte en cours et la nouvelle demande pour choisir l'unique rôle adapté.\n"
            "Rôles disponibles :\n" + "\n".join(descriptions) + "\n"
            "Principes :\n"
            "1. Si l'utilisateur fait une courte relance ou demande d'exécuter l'action proposée dans le contexte en cours, conserve le rôle précédent.\n"
            "2. Si la demande porte sur des faits récents, l'actualité, le sport, un produit sorti ou de la culture générale, choisis `general_et_web`.\n"
            "3. Si la demande implique des fichiers locaux, le terminal Linux ou du code, choisis `code_courant` (ou `architecture_lourde` si refactoring complexe).\n"
            "4. Si la demande porte sur ce que voit la caméra, choisis `vision_multimodale`."
            f"{consigne_rejet}\n"
            'Réponds UNIQUEMENT en JSON valide : {"role": "nom_du_role", "raison": "explication courte"}'
        )
        prompt_usr = (
            f"Rôle précédent : {dernier_role or 'aucun'}\n"
            f"Scores sémantiques : {', '.join(f'{r}={s:.2f}' for r, s in candidats[:3])}\n"
            f"{requete_ctx}"
        )

        try:
            brut = self._appeler_superviseur_cpu(prompt_sys, prompt_usr, max_tokens=65, format_json=True)
            data = _extraire_json_propre(brut)
            if data:
                role_choisi = str(data.get("role", "")).strip()
                raison = str(data.get("raison", "arbitrage IA")).strip()
                for cfg in self.config.get("routes", {}).values():
                    if cfg.get("role") == role_choisi:
                        return role_choisi, f"superviseur ({raison})"
        except Exception:
            pass
        return candidats[0][0], "sémantique:fallback"

    def choisir_modele(
        self,
        message: str,
        dernier_modele: str | None = None,
        modele_force: str | None = None,
        modele_verrouille: str | None = None,
        dernier_prompt: str | None = None,
        derniere_reponse: str | None = None,
    ) -> tuple[str, float, str]:
        if modele_force:
            return modele_force, 1.0, "manuel:prompt (@)"
        if modele_verrouille:
            return modele_verrouille, 1.0, "manuel:verrouillé (/model)"

        routage_cfg = self.config.get("routage", {})
        seuils = routage_cfg.get("seuils", {})
        embed_model = routage_cfg.get("embed_model", "bge-m3")

        msg_brut = re.sub(r"(.)\1{2,}", r"\1\1", message.strip())
        if len(msg_brut.split()) <= 5 and "phatique" in self.embeddings_roles:
            vec_brut = _embed_texte_cache(embed_model, msg_brut)
            scores_bruts = sorted(
                ((r, self._score_top_k(vec_brut, v, k=2)) for r, v in self.embeddings_roles.items()),
                key=lambda x: x[1],
                reverse=True,
            )
            if scores_bruts and scores_bruts[0][0] == "phatique" and scores_bruts[0][1] >= 0.60:
                return self.modele_par_role("phatique"), scores_bruts[0][1], "sémantique:phatique_direct"

        requete_ctx = self._construire_requete_contextualisee(message, dernier_prompt, derniere_reponse)
        vec_ctx = _embed_texte_cache(embed_model, requete_ctx)

        scores_roles: list[tuple[str, float]] = [
            (role, self._score_top_k(vec_ctx, vecs, k=3))
            for role, vecs in self.embeddings_roles.items()
        ]
        scores_roles.sort(key=lambda x: x[1], reverse=True)

        if not scores_roles:
            return self.config.get("route_defaut", "qwen2.5:7B"), 0.5, "défaut"

        meilleur_role, meilleur_score = scores_roles[0]
        second_score = scores_roles[1][1] if len(scores_roles) > 1 else 0.0
        marge = meilleur_score - second_score

        seuil_direct = float(seuils.get("confiance_directe") or 0.63)
        marge_directe = float(seuils.get("marge_directe") or 0.05)

        seuil_insat = float(seuils.get("insatisfaction") or 0.58)
        rejet_precedent = self._score_top_k(vec_ctx, self.embeddings_intentions.get("insatisfaction", []), k=2) >= seuil_insat

        if meilleur_score >= seuil_direct and marge >= marge_directe and not rejet_precedent:
            return self.modele_par_role(meilleur_role), meilleur_score, f"sémantique (+{marge:.2f})"

        if routage_cfg.get("utiliser_superviseur_ia", True):
            dernier_role = self.config.get("routes", {}).get(dernier_modele or "", {}).get("role")
            role_valide, raison_ia = self._arbitrer_par_superviseur_ia(
                requete_ctx,
                scores_roles,
                dernier_role,
                rejet_precedent=rejet_precedent,
            )
            return self.modele_par_role(role_valide), meilleur_score, raison_ia

        return self.modele_par_role(meilleur_role), meilleur_score, f"sémantique:top1 (+{marge:.2f})"

    def ajouter_phrase_entrainement(self, categorie: str, cible: str, phrase_generaliste: str) -> None:
        """Enrichit le fichier externe avec une phrase généraliste et met à jour les embeddings."""
        propre = phrase_generaliste.strip()
        if not propre:
            return
        liste = self.dataset.setdefault(categorie, {}).setdefault(cible, [])
        if propre not in liste:
            liste.append(propre)
            self.sauvegarder_dataset()
            self.charger(silencieux=True)
