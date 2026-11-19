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
CACHE_FILE = BASE_DIR / "routes_cache.json"

CONFIG_DEFAUT: dict = {
    "web_mode": "ask",
    "file_mode": "ask",
    "shell_mode": "ask",
    "modele_optimiseur": "qwen2.5:7b",
    "route_defaut": "qwen2.5:7b",
    "regles_apprises": [],
    "routage": {
        "utiliser_embeddings": True,
        "embed_model": "bge-m3",
        "seuils": {
            "confiance": 0.54,
            "marge": 0.03,
            "insatisfaction": 0.62,
        },
        "mots_cles_lourds": [
            "architecture", "microservice", "deadlock", "race condition",
            "concurrence", "multithread", "refactore", "refactor", "distribué", "scalab",
        ],
        "regles_regex": [
            {
                "nom": "phatique",
                "cible_role": "phatique",
                "confiance": 1.0,
                "dedupliquer_lettres": True,
                "pattern": (
                    r"^(salut|slut|slt|bonjour|bjr|bonsoir|coucou|cc|helo|hi|hey|yo)"
                    r"(\s+(ma\s+go|mec|gros|frero|frérot|toi|la\s+team|tout\s+le\s+monde|ca\s+va|ça\s+va))?[!?.…\s]*$|"
                    r"^(merci(\s+beaucoup)?|ok(ay)?|dacord|d'accord|parfait|super|genial|génial|nickel|"
                    r"au\s+revoir|a\s+plus|à\s+plus|bye)[!?.…\s]*$"
                ),
            },
            {
                "nom": "code_ou_systeme",
                "cible_role": "code_courant",
                "cible_role_lourd": "architecture_lourde",
                "confiance": 0.95,
                "pattern": (
                    r"(```|Traceback|\b[\w./-]+\.(py|rs|c|cpp|h|js|ts|sh|json|jsonl|toml|lua|html|css|sql|md|txt|log)\b|"
                    r"\b(fichier|fichiers|dossier|dossiers|répertoire|arborescence|script|code\s+source|terminal|bash|shell|"
                    r"commande|ex[ée]cute|lance|pwd|ls|cat|grep|git|nvidia-smi|systemctl|journalctl|os-release|machine|"
                    r"dnf|rpm|flatpak|sudo|paquets?|syst[èe]me|\bsys\b|zenity|notify-send|fen[êe]tre|"
                    r"mettre\s+[àa]\s+j(ou)?r|mise\s+[àa]\s+j(ou)?r|\bmaj\b|update|upgrade)\b)"
                ),
            },
            {
                "nom": "web_ou_actualite",
                "cible_role": "general_et_web",
                "confiance": 0.90,
                "pattern": (
                    r"\b(internet|web|en\s+ligne|actualit[ée]s?|news|m[ée]t[ée]o|aujourd'?hui|hier|cette\s+semaine|"
                    r"derni[èe]re?\s+(vid[ée]o|version|sortie|album)|prix|cours|2025|2026|cherche|recherche|google|youtube)\b"
                ),
            },
        ],
        "pattern_continuation_action": r"^(fais[- ]le|fais[- ]la|ex[ée]cute([- ]le|-la)?|lance([- ]le|-la)?|vas[- ]y|essaie|teste|appliques?[- ]le)[!?.…\s]*$",
        "pattern_rupture_relance": r"^(lis|lire|affiche|montre|liste|cherche|modifie|crée|ex[ée]cute|lance|quel\s+mod[èe]le|quelle\s+ia|qui\s+peut|qu'est-ce\s+que\s+tu\s+peux)\b",
    },
    "signaux": {
        "consentement_oui": [
            "oui", "ouais", "yep", "yes", "ok", "okay", "d'accord", "dac",
            "vas-y", "fonce", "allez", "carrément", "pourquoi pas",
            "si tu veux", "volontiers", "avec plaisir", "fais-le",
            "cherche", "go", "bien sûr", "évidemment", "c'est bon",
            "je veux bien", "autorise", "tu peux", "grave",
        ],
        "consentement_non": [
            "non", "nan", "nope", "no", "pas besoin", "laisse tomber",
            "t'embête pas", "ne cherche pas", "n'y va pas", "refuse", "interdit",
            "stop", "annule", "pas la peine", "inutile", "bof", "non merci",
            "je ne veux pas", "reste hors ligne", "tant pis", "oublie",
        ],
        "insatisfaction": [
            "non pas du tout c'est n'importe quoi",
            "tu es complètement à côté de la plaque",
            "tu racontes vraiment n'importe quoi",
            "ce n'est pas le bon modèle tu te trompes",
            "tu ne réponds pas à ma question",
            "arrête de parler comme un robot tu es lourd",
            "tu comprends rien à ce que je dis",
            "c'est faux et hors sujet",
            "tu es en totale hallucination",
            "pfff laisse tomber t'es nul",
        ],
    },
    "routes": {
        "gemma3:1b": {
            "alias": "1b",
            "role": "phatique",
            "expertise": "Salutations courtes et accusés de réception.",
            "temperature": 0.6,
            "num_ctx": 2048,
            "num_predict": 60,
            "web_tools": False,
            "file_tools": False,
            "shell_tools": False,
            "examples": [],
        },
        "huihui_ai/qwen3-abliterated:8b": {
            "alias": "c",
            "role": "creatif_discussion",
            "expertise": "Discussion libre, humour, brainstorming, ton naturel sans filtre ni rigidité.",
            "temperature": 0.8,
            "num_ctx": 4096,
            "num_predict": None,
            "web_tools": True,
            "file_tools": False,
            "shell_tools": False,
            "examples": [
                "on discute un peu de tout et de rien",
                "destresse un peu soit plus humain et naturel",
                "raconte moi une histoire immersive",
                "tu penses quoi de tout ça donne moi ton avis",
                "fais moi une blague ou un truc drôle",
            ],
        },
        "qwen2.5:7b": {
            "alias": "g",
            "role": "general_et_web",
            "expertise": "Culture générale, explications théoriques, synthèse, recherche Internet et mémoire utilisateur.",
            "temperature": 0.25,
            "num_ctx": 8192,
            "num_predict": None,
            "web_tools": True,
            "file_tools": False,
            "shell_tools": False,
            "examples": [
                "qu'est-ce que tu sais sur moi quelles infos tu as",
                "explique moi théoriquement comment ça fonctionne",
                "donne moi une définition simple et claire",
                "résume moi ce sujet en quelques points clés",
            ],
        },
        "qwen2.5-coder:7b": {
            "alias": "code",
            "role": "code_courant",
            "expertise": "Administration système Linux/Fedora, commandes terminal bash, lecture/édition de fichiers locaux, scripts Python/Rust/C.",
            "temperature": 0.15,
            "num_ctx": 8192,
            "num_predict": None,
            "web_tools": True,
            "file_tools": True,
            "shell_tools": True,
            "examples": [
                "exécute cette commande dans le terminal",
                "mets à jour mon système fedora avec dnf",
                "lis ce fichier python et modifie-le pour corriger le bug",
            ],
        },
        "qwen2.5-coder:14b": {
            "alias": "code+",
            "role": "architecture_lourde",
            "expertise": "Refactoring complexe multi-fichiers, architecture logicielle lourde et debugging bas niveau.",
            "temperature": 0.15,
            "num_ctx": 4096,
            "num_predict": None,
            "web_tools": True,
            "file_tools": True,
            "shell_tools": True,
            "examples": [
                "refactore entièrement mon projet vers une architecture propre",
                "conçois l'architecture complète d'un système distribué",
            ],
        },
        "gemma3:12b-it-qat": {
            "alias": "12b",
            "role": "vision_multimodale",
            "expertise": "Analyse visuelle complexe de scènes caméra (intrusion, contexte, lecture d'image).",
            "temperature": 0.15,
            "num_ctx": 4096,
            "num_predict": None,
            "web_tools": False,
            "file_tools": False,
            "shell_tools": False,
            "examples": [],
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


def _dedupli_lettres(texte: str) -> str:
    return re.sub(r"(.)\1+", r"\1", texte.strip().lower())


@functools.lru_cache(maxsize=32)
def _embed_texte_cache(embed_model: str, texte: str) -> tuple[float, ...]:
    """Vectorise un texte sur CPU (0 Mo VRAM) avec mise en cache LRU intra-tour."""
    raw_vec = ollama.embed(model=embed_model, input=texte, options={"num_gpu": 0}, keep_alive=-1)["embeddings"][0]
    return _normaliser_l2(raw_vec)


class RouteurAutonome:
    def __init__(self) -> None:
        self.config: dict = {}
        self._config_mtime: float = 0.0
        self.regles_compilees: list[dict] = []
        self.regex_action: re.Pattern | None = None
        self.regex_rupture: re.Pattern | None = None
        self.embeddings_ref: dict[str, list[list[float]]] = {}
        self.embeddings_consentement: dict[str, list[list[float]]] = {}
        self.embeddings_insatisfaction: list[list[float]] = []
        self.charger(silencieux=False)

    def verifier_hot_reload(self) -> bool:
        try:
            mtime = CONFIG_FILE.stat().st_mtime
        except OSError:
            return False
        if mtime != self._config_mtime:
            self.charger(silencieux=True)
            return True
        return False

    def sauvegarder_config(self) -> None:
        tmp = CONFIG_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.config, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, CONFIG_FILE)
        self._config_mtime = CONFIG_FILE.stat().st_mtime

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
            if cle not in cfg:
                cfg[cle] = val
                modifie = True

        for cle_r, val_r in defaut["routage"].items():
            if cle_r not in routage:
                routage[cle_r] = val_r
                modifie = True

        for modele, r_cfg in cfg.get("routes", {}).items():
            if "shell_tools" not in r_cfg:
                r_cfg["shell_tools"] = r_cfg.get("role") in ("code_courant", "architecture_lourde")
                modifie = True
            if "expertise" not in r_cfg and modele in defaut["routes"]:
                r_cfg["expertise"] = defaut["routes"][modele]["expertise"]
                modifie = True

        return cfg, modifie

    def _restaurer_ou_generer_config(self, motif: str) -> dict:
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
        nouvelle_cfg = None

        if not CONFIG_FILE.exists() or CONFIG_FILE.stat().st_size == 0:
            if not self.config:
                nouvelle_cfg = self._restaurer_ou_generer_config("config_ia.json absent ou vide")
            else:
                return
        else:
            try:
                nouvelle_cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                if not isinstance(nouvelle_cfg, dict) or not nouvelle_cfg.get("routes"):
                    raise ValueError("clé 'routes' absente ou vide")
            except (json.JSONDecodeError, ValueError, OSError) as err:
                if self.config:
                    self._config_mtime = CONFIG_FILE.stat().st_mtime
                    return
                nouvelle_cfg = self._restaurer_ou_generer_config(f"config_ia.json invalide ({err})")

        nouvelle_cfg, a_ete_migree = self._normaliser_config(nouvelle_cfg)
        self.config = nouvelle_cfg
        if a_ete_migree:
            self.sauvegarder_config()
        else:
            self._config_mtime = CONFIG_FILE.stat().st_mtime

        routage_cfg = self.config.get("routage", {})

        self.regles_compilees = []
        for r in routage_cfg.get("regles_regex", []):
            try:
                self.regles_compilees.append({
                    **r,
                    "_compiled": re.compile(r["pattern"], re.IGNORECASE),
                })
            except re.error:
                pass

        pat_action = routage_cfg.get("pattern_continuation_action", "")
        self.regex_action = re.compile(pat_action, re.IGNORECASE) if pat_action else None

        pat_rupture = routage_cfg.get("pattern_rupture_relance", "")
        self.regex_rupture = re.compile(pat_rupture, re.IGNORECASE) if pat_rupture else None

        if not routage_cfg.get("utiliser_embeddings", True):
            self.embeddings_ref = {}
            self.embeddings_consentement = {}
            self.embeddings_insatisfaction = []
            return

        embed_model = routage_cfg.get("embed_model", "bge-m3")
        routes_brutes = {
            m: cfg["examples"]
            for m, cfg in self.config.get("routes", {}).items()
            if cfg.get("role") not in ("phatique", "vision_multimodale") and cfg.get("examples")
        }
        signaux = self.config.get("signaux", {})
        consent_oui = signaux.get("consentement_oui", ["oui"])
        consent_non = signaux.get("consentement_non", ["non"])
        insatisfaction = signaux.get("insatisfaction", [])

        signature = hashlib.sha256(
            json.dumps(
                {
                    "model": embed_model,
                    "routes": routes_brutes,
                    "oui": consent_oui,
                    "non": consent_non,
                    "insatisfaction": insatisfaction,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

        if CACHE_FILE.exists():
            try:
                data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
                if data.get("signature") == signature:
                    self.embeddings_ref = data["embeddings"]
                    self.embeddings_consentement = data["consent_embeddings"]
                    self.embeddings_insatisfaction = data["insatisfaction_embeddings"]
                    return
            except (json.JSONDecodeError, KeyError):
                pass

        self.embeddings_ref = {
            model: self._vectoriser_liste(embed_model, exemples)
            for model, exemples in routes_brutes.items()
        }
        self.embeddings_consentement = {
            "OUI": self._vectoriser_liste(embed_model, consent_oui),
            "NON": self._vectoriser_liste(embed_model, consent_non),
        }
        self.embeddings_insatisfaction = self._vectoriser_liste(embed_model, insatisfaction)

        tmp = CACHE_FILE.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({
                "signature": signature,
                "embeddings": self.embeddings_ref,
                "consent_embeddings": self.embeddings_consentement,
                "insatisfaction_embeddings": self.embeddings_insatisfaction,
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
        if "route_defaut" in self.config:
            return self.config["route_defaut"]
        return next(iter(routes), "qwen2.5:7b")

    def interpreter_consentement(self, reponse_utilisateur: str) -> tuple[bool, str]:
        texte = reponse_utilisateur.strip()
        if not texte:
            return False, "vide -> NON par défaut"

        texte_min = texte.lower().strip(" !?.,…")
        signaux = self.config.get("signaux", {})
        if texte_min in [s.lower() for s in signaux.get("consentement_oui", ["oui"])] or texte_min in ("o", "y", "1"):
            return True, "liste:OUI"
        if texte_min in [s.lower() for s in signaux.get("consentement_non", ["non"])] or texte_min in ("n", "0"):
            return False, "liste:NON"

        routage_cfg = self.config.get("routage", {})
        if routage_cfg.get("utiliser_embeddings", True) and self.embeddings_consentement:
            embed_model = routage_cfg.get("embed_model", "bge-m3")
            vec_rep = _embed_texte_cache(embed_model, texte)

            def score_top2(vecs: list[list[float]]) -> float:
                if not vecs:
                    return 0.0
                sims = sorted((_produit_scalaire(vec_rep, v) for v in vecs), reverse=True)
                top = sims[: min(2, len(sims))]
                return sum(top) / len(top)

            score_oui = score_top2(self.embeddings_consentement["OUI"])
            score_non = score_top2(self.embeddings_consentement["NON"])
            ecart = score_oui - score_non
            if ecart >= 0.045 and score_oui >= 0.56:
                return True, f"embedding:OUI ({score_oui:.2f} vs {score_non:.2f})"
            if -ecart >= 0.045 and score_non >= 0.56:
                return False, f"embedding:NON ({score_non:.2f} vs {score_oui:.2f})"

        mini_modele = self.modele_par_role("phatique")
        try:
            res = ollama.chat(
                model=mini_modele,
                messages=[
                    {
                        "role": "system",
                        "content": "Classifieur binaire. L'utilisateur autorise-t-il l'action ? Réponds UNIQUEMENT OUI ou NON.",
                    },
                    {"role": "user", "content": texte},
                ],
                options={"num_gpu": 0, "temperature": 0.0, "num_predict": 4},
                keep_alive="2m",
            )
            verdict = res["message"]["content"].strip().upper()
            return ("OUI" in verdict and "NON" not in verdict), f"mini-IA ({mini_modele}):{verdict}"
        except Exception:
            return False, "fallback:NON"

    def evaluer_insatisfaction(self, reaction: str, dernier_prompt: str, derniere_reponse: str) -> tuple[bool, str]:
        texte = reaction.strip()
        if not texte or len(texte) > 250 or not self.embeddings_insatisfaction:
            return False, ""

        routage_cfg = self.config.get("routage", {})
        embed_model = routage_cfg.get("embed_model", "bge-m3")
        vec_rep = _embed_texte_cache(embed_model, texte)
        sims = sorted((_produit_scalaire(vec_rep, v) for v in self.embeddings_insatisfaction), reverse=True)
        score_top2 = sum(sims[:2]) / min(2, len(sims))

        seuil = routage_cfg.get("seuils", {}).get("insatisfaction", 0.62)
        if score_top2 < seuil:
            return False, ""

        mini_modele = self.modele_par_role("phatique")
        try:
            res = ollama.chat(
                model=mini_modele,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Détermine si le NOUVEAU MESSAGE exprime un rejet, une correction ou une insatisfaction "
                            "vis-à-vis de la RÉPONSE PRÉCÉDENTE. Réponds UNIQUEMENT par ECHEC ou OK."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"DEMANDE : {dernier_prompt}\nRÉPONSE IA : {derniere_reponse[:350]}\nNOUVEAU MESSAGE : {texte}",
                    },
                ],
                options={"num_gpu": 0, "temperature": 0.0, "num_predict": 4},
                keep_alive="2m",
            )
            if "ECHEC" in res["message"]["content"].strip().upper():
                return True, f"IA-critique (sim={score_top2:.2f})"
        except Exception:
            pass
        return False, ""

    def choisir_modele(
        self,
        message: str,
        dernier_modele: str | None = None,
        modele_force: str | None = None,
        modele_verrouille: str | None = None,
    ) -> tuple[str, float, str]:
        if modele_force:
            return modele_force, 1.0, "manuel:prompt (@)"
        if modele_verrouille:
            return modele_verrouille, 1.0, "manuel:verrouillé (/model)"

        texte = message.strip()
        texte_dedup = _dedupli_lettres(texte)
        mots = texte.split()
        routage_cfg = self.config.get("routage", {})
        seuils = routage_cfg.get("seuils", {"confiance": 0.54, "marge": 0.03})
        mots_lourds = routage_cfg.get("mots_cles_lourds", [])

        modele_phatique = self.modele_par_role("phatique")
        modele_defaut = self.config.get("route_defaut", self.modele_par_role("general_et_web"))
        modele_chat = self.modele_par_role("creatif_discussion")

        for regle in self.regles_compilees:
            rx = regle["_compiled"]
            cible_test = texte_dedup if regle.get("dedupliquer_lettres") else texte
            if rx.search(cible_test):
                role_cible = regle["cible_role"]
                if regle.get("cible_role_lourd") and (len(texte) > 700 or any(k in texte.lower() for k in mots_lourds)):
                    role_cible = regle["cible_role_lourd"]
                return self.modele_par_role(role_cible), float(regle.get("confiance", 0.95)), f"règle:{regle.get('nom', role_cible)}"

        if self.regex_action and self.regex_action.match(texte) and dernier_modele and dernier_modele != modele_phatique:
            if dernier_modele == modele_defaut:
                return self.modele_par_role("code_courant"), 0.90, "continuité:délégation_action"
            return dernier_modele, 0.90, "continuité:action"

        if routage_cfg.get("utiliser_embeddings", True) and self.embeddings_ref:
            embed_model = routage_cfg.get("embed_model", "bge-m3")
            vec_msg = _embed_texte_cache(embed_model, texte)

            scores = []
            for model, vecs in self.embeddings_ref.items():
                sims = sorted((_produit_scalaire(vec_msg, v) for v in vecs), reverse=True)
                top2 = sims[: min(2, len(sims))]
                scores.append((model, sum(top2) / len(top2)))
            scores.sort(key=lambda x: x[1], reverse=True)

            meilleur_modele, meilleur_score = scores[0]
            second_score = scores[1][1] if len(scores) > 1 else 0.0
            marge = meilleur_score - second_score

            rupture = bool(self.regex_rupture and self.regex_rupture.search(texte))
            if dernier_modele and dernier_modele != modele_phatique and len(mots) <= 5 and meilleur_score < 0.56 and not rupture:
                return dernier_modele, meilleur_score, "continuité:relance"

            if meilleur_score < seuils.get("confiance", 0.54) or (marge < seuils.get("marge", 0.03) and meilleur_score < 0.62):
                if "?" not in texte and len(mots) <= 8:
                    return modele_chat, meilleur_score, "fallback:discussion"
                return modele_defaut, meilleur_score, "fallback:défaut"

            return meilleur_modele, meilleur_score, f"sémantique (marge +{marge:.2f})"

        rupture = bool(self.regex_rupture and self.regex_rupture.search(texte))
        if dernier_modele and dernier_modele != modele_phatique and len(mots) <= 6 and not rupture:
            return dernier_modele, 0.80, "continuité:session"

        return modele_defaut, 0.75, "défaut (auto-délégation active)"