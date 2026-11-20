import copy
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import ollama

from memory_manager import (
    BASE_DIR,
    MEMOIRE_EPISODIQUE_FILE,
    charger_memoire_utile,
    charger_tous_echanges,
    compacter_base_donnees,
    extraire_cas_regression,
    memoriser_fait_utilisateur,
    sauvegarder_tous_echanges,
)

try:
    import terminal_panels as panels
except ImportError:
    panels = None

CONFIG_FILE = BASE_DIR / "config_ia.json"
CONFIG_BACKUP_FILE = BASE_DIR / "config_ia.bak.json"
TRAINING_FILE = BASE_DIR / "entrainement_routage.json"
REGRESSION_SUITE_FILE = BASE_DIR / "tests_regression_eon.json"
OPTIMISEUR_LOG_FILE = BASE_DIR / "optimiseur_journal.log"
RAPPORT_FINAL_FILE = BASE_DIR / "dernier_rapport_optimisation.md"
BACKUPS_DIR = BASE_DIR / "backups_config"
HISTORIQUE_OPTI_FILE = BACKUPS_DIR / "historique_optimisations.json"

SPECS_PARAMETRES: dict[str, dict[str, float]] = {
    "routage.seuils.confiance_directe": {"min": 0.48, "max": 0.78, "pas_max": 0.06, "defaut": 0.63},
    "routage.seuils.marge_directe": {"min": 0.015, "max": 0.14, "pas_max": 0.025, "defaut": 0.05},
    "routage.seuils.intention_outil": {"min": 0.48, "max": 0.74, "pas_max": 0.05, "defaut": 0.58},
    "routage.seuils.insatisfaction": {"min": 0.50, "max": 0.75, "pas_max": 0.05, "defaut": 0.58},
    "temperature": {"min": 0.05, "max": 0.90, "pas_max": 0.12, "defaut": 0.20},
}

REGEX_SECRET = re.compile(
    r"(sudo\s+-S|mot\s+de\s+passe|password|passwd|api[_-]?key|token\s*=|secret\s*=|-----BEGIN\s+[A-Z ]+PRIVATE\s+KEY-----)",
    re.IGNORECASE,
)

# Socle minimal de non-régression couvrant les rôles et intentions critiques d'Eon
CAS_REGRESSION_CANONIQUES: list[dict[str, str]] = [
    {"prompt": "Écris un script Python asynchrone pour scanner des ports TCP locaux.", "role_attendu": "code"},
    {"prompt": "Corrige l'erreur de segmentation dans ce code C avec Valgrind.", "role_attendu": "code"},
    {"prompt": "Vérifie l'espace disque disponible et la température du CPU sur Fedora.", "intention_attendue": "shell"},
    {"prompt": "Lis les 50 premières lignes du fichier main.py dans le projet.", "intention_attendue": "fichiers"},
    {"prompt": "Quelles sont les dernières vulnérabilités Linux publiées cette semaine ?", "intention_attendue": "web"},
    {"prompt": "Regarde par la webcam qui est assis devant l'ordinateur.", "intention_attendue": "vision"},
    {"prompt": "Explique la différence théorique entre le chiffrement symétrique AES-GCM et ChaCha20.", "role_attendu": "general"},
]


# ---------------- UTILITAIRES ATOMIQUES & LOGS ----------------

def _ecrire_atomique(chemin: Path, contenu: str) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(f"{chemin}.tmp")
    tmp.write_text(contenu, encoding="utf-8")
    os.replace(tmp, chemin)


def _log(message: str, journal: list[str], afficher_tui: bool = True) -> None:
    print(message, flush=True)
    journal.append(message)
    if afficher_tui and panels is not None:
        try:
            if hasattr(panels, "log_outil"):
                panels.log_outil(message)
        except Exception:
            pass


def _extraire_json_propre(texte: str) -> dict[str, Any] | None:
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


def _contient_secret_potentiel(texte: str) -> bool:
    return bool(REGEX_SECRET.search(texte))


def _cosinus(v1: list[float], v2: list[float]) -> float:
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    return sum(a * b for a, b in zip(v1, v2)) / (n1 * n2) if n1 > 0.0 and n2 > 0.0 else 0.0


# ---------------- GARDIEN THERMIQUE & ALIMENTATION LAPTOP ----------------

def _lire_temperatures() -> tuple[float | None, float | None]:
    """Lit la température GPU (nvidia-smi) et CPU (/sys/class/thermal) en °C."""
    temp_gpu: float | None = None
    temp_cpu: float | None = None

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        ligne = out.decode("utf-8").strip().splitlines()[0]
        temp_gpu = float(ligne)
    except Exception:
        pass

    try:
        zones = sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp"))
        vals = []
        for z in zones:
            raw = z.read_text(encoding="utf-8").strip()
            if raw.isdigit():
                c = int(raw) / 1000.0
                if 20.0 <= c <= 115.0:
                    vals.append(c)
        if vals:
            temp_cpu = max(vals)
    except Exception:
        pass

    return temp_gpu, temp_cpu


def _etat_alimentation_portable() -> tuple[bool, int]:
    """Retourne (sur_secteur, pourcentage_batterie) via /sys/class/power_supply."""
    sur_secteur = True
    niveau_bat = 100
    try:
        ps_dir = Path("/sys/class/power_supply")
        if not ps_dir.exists():
            return True, 100
        for bat in sorted(ps_dir.glob("BAT*")):
            statut_f = bat / "status"
            cap_f = bat / "capacity"
            if statut_f.exists():
                statut = statut_f.read_text(encoding="utf-8").strip().lower()
                if statut == "discharging":
                    sur_secteur = False
            if cap_f.exists():
                raw_cap = cap_f.read_text(encoding="utf-8").strip()
                if raw_cap.isdigit():
                    niveau_bat = int(raw_cap)
            break
    except Exception:
        pass
    return sur_secteur, niveau_bat


def _pause_thermique_adaptative(
    journal: list[str],
    pause_base_sec: float = 4.0,
    seuil_gpu_c: float = 74.0,
    seuil_cpu_c: float = 80.0,
) -> None:
    """Laisse respirer le refroidissement entre deux inférences lourdes."""
    time.sleep(pause_base_sec)
    t_gpu, t_cpu = _lire_temperatures()
    attentes = 0
    while (t_gpu is not None and t_gpu > seuil_gpu_c) or (t_cpu is not None and t_cpu > seuil_cpu_c):
        if attentes == 0:
            _log(
                f"  [Thermique] Température élevée (GPU={t_gpu or '?'}°C, CPU={t_cpu or '?'}°C) "
                f"-> pause refroidissement...",
                journal,
            )
        time.sleep(8.0)
        attentes += 1
        t_gpu, t_cpu = _lire_temperatures()
        if attentes >= 15:
            break


# ---------------- GESTION DES BACKUPS & TRAJECTOIRE DICHOTOMIQUE ----------------

def _charger_historique_optimisations() -> dict[str, Any]:
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    if HISTORIQUE_OPTI_FILE.exists():
        try:
            data = json.loads(HISTORIQUE_OPTI_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("sessions", [])
                data.setdefault("trajectoires_parametres", {})
                return data
        except Exception:
            pass
    return {"sessions": [], "trajectoires_parametres": {}}


def _sauvegarder_historique_optimisations(hist: dict[str, Any]) -> None:
    hist["sessions"] = hist.get("sessions", [])[-40:]
    _ecrire_atomique(HISTORIQUE_OPTI_FILE, json.dumps(hist, indent=2, ensure_ascii=False))


def _creer_snapshot_backup(horodatage_id: str) -> str:
    """Crée un backup horodaté de config_ia.json et entrainement_routage.json."""
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    if CONFIG_FILE.exists():
        shutil.copy2(CONFIG_FILE, CONFIG_BACKUP_FILE)
        shutil.copy2(CONFIG_FILE, BACKUPS_DIR / f"config_ia_{horodatage_id}.json")
    if TRAINING_FILE.exists():
        shutil.copy2(TRAINING_FILE, BACKUPS_DIR / f"entrainement_routage_{horodatage_id}.json")

    for prefixe in ("config_ia_*.json", "entrainement_routage_*.json"):
        anciens = sorted(BACKUPS_DIR.glob(prefixe))
        for f in anciens[:-25]:
            f.unlink(missing_ok=True)
    return f"backups_config/config_ia_{horodatage_id}.json"


def _restaurer_snapshot_backup(horodatage_id: str) -> bool:
    """Restaure immédiatement config_ia.json et entrainement_routage.json en cas de régression."""
    restaure = False
    bak_cfg = BACKUPS_DIR / f"config_ia_{horodatage_id}.json"
    bak_trn = BACKUPS_DIR / f"entrainement_routage_{horodatage_id}.json"
    if bak_cfg.exists():
        shutil.copy2(bak_cfg, CONFIG_FILE)
        restaure = True
    if bak_trn.exists():
        shutil.copy2(bak_trn, TRAINING_FILE)
        restaure = True
    return restaure


def _formater_contexte_backups(hist: dict[str, Any]) -> str:
    """Prépare un résumé structuré de l'historique des réglages pour éviter toute régression."""
    trajs = hist.get("trajectoires_parametres", {})
    lignes = ["HISTORIQUE DES TRAJECTOIRES DE PARAMÈTRES (DICHOTOMIE) :"]
    if not trajs:
        lignes.append("- Aucun paramètre encore ajusté (première calibration).")
    else:
        for param, info in trajs.items():
            valeurs = info.get("valeurs", [])[-6:]
            chaine_vals = " -> ".join(f"{v:.3f}" for v in valeurs)
            dernier_pas = info.get("dernier_pas", 0.0)
            lignes.append(
                f"- {param} : [{chaine_vals}] (dernier pas={dernier_pas:+.3f}, "
                f"nb_inversion={info.get('inversions', 0)})"
            )

    sessions = hist.get("sessions", [])[-4:]
    if sessions:
        lignes.append("\nDERNIÈRES SESSIONS D'OPTIMISATION :")
        for s in sessions:
            lignes.append(f"- [{s.get('date')}] {s.get('resume_court', '')}")
            for act in s.get("actions_cles", [])[:5]:
                lignes.append(f"    * {act}")

    return "\n".join(lignes)


def _calculer_valeur_dichotomique(
    cle_param: str,
    valeur_actuelle: float,
    valeur_souhaitee: float,
    spec: dict[str, float],
    hist: dict[str, Any],
) -> tuple[float, str]:
    trajs = hist.setdefault("trajectoires_parametres", {})
    etat = trajs.setdefault(
        cle_param,
        {"valeurs": [round(valeur_actuelle, 4)], "dernier_pas": 0.0, "inversions": 0},
    )

    ecart_brut = valeur_souhaitee - valeur_actuelle
    if abs(ecart_brut) < 0.004:
        return valeur_actuelle, "écart négligeable (< 0.004)"

    signe = 1.0 if ecart_brut > 0 else -1.0
    dernier_pas = float(etat.get("dernier_pas", 0.0))
    pas_autorise = float(spec["pas_max"])

    if dernier_pas != 0.0:
        if (dernier_pas > 0 and signe < 0) or (dernier_pas < 0 and signe > 0):
            pas_autorise = max(0.005, abs(dernier_pas) * 0.5)
            etat["inversions"] = int(etat.get("inversions", 0)) + 1
            motif_pas = f"dichotomie après inversion (pas réduit à {pas_autorise:.3f})"
        else:
            pas_autorise = min(pas_autorise, max(0.01, abs(dernier_pas) * 0.85))
            motif_pas = f"progression contrôlée (pas max {pas_autorise:.3f})"
    else:
        motif_pas = f"exploration initiale bornée (pas max {pas_autorise:.3f})"

    delta_effectif = signe * min(abs(ecart_brut), pas_autorise)
    nouvelle_val = round(max(spec["min"], min(spec["max"], valeur_actuelle + delta_effectif)), 3)

    if nouvelle_val == round(valeur_actuelle, 3):
        return valeur_actuelle, "butée de borne atteinte"

    etat["valeurs"].append(nouvelle_val)
    etat["valeurs"] = etat["valeurs"][-12:]
    etat["dernier_pas"] = round(nouvelle_val - valeur_actuelle, 4)
    return nouvelle_val, motif_pas


# ---------------- REPLAY DE RÉGRESSION BOUT-EN-BOUT ----------------

def _obtenir_embed_model(config: dict[str, Any]) -> str:
    return config.get("routage", {}).get("embed_model") or config.get("embed_model") or "bge-m3"


def _charger_suite_regression(config: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Fusionne la suite de régression persistante (tests_regression_eon.json),
    les cas canoniques et les vrais cas historiques validés en SQLite.
    """
    suite: list[dict[str, Any]] = []
    vus: set[str] = set()

    if REGRESSION_SUITE_FILE.exists():
        try:
            brut = json.loads(REGRESSION_SUITE_FILE.read_text(encoding="utf-8"))
            if isinstance(brut, list):
                for item in brut:
                    if isinstance(item, dict) and item.get("prompt"):
                        cle = str(item["prompt"]).strip().lower()
                        if cle not in vus:
                            vus.add(cle)
                            suite.append(item)
        except Exception:
            pass

    for canon in CAS_REGRESSION_CANONIQUES:
        cle = canon["prompt"].strip().lower()
        if cle not in vus:
            vus.add(cle)
            suite.append(dict(canon))

    routes_cfg = config.get("routes", {})
    role_par_modele = {m.lower(): cfg.get("role", "") for m, cfg in routes_cfg.items()}

    for cas_sql in extraire_cas_regression(limite=35):
        prompt = cas_sql["prompt"].strip()
        cle = prompt.lower()
        if cle in vus:
            continue
        mod_att = cas_sql["modele_attendu"].lower()
        role_att = role_par_modele.get(mod_att, "")
        if role_att:
            vus.add(cle)
            suite.append(
                {
                    "prompt": prompt,
                    "role_attendu": role_att,
                    "origine": cas_sql["origine"],
                }
            )

    suite = suite[-80:]
    _ecrire_atomique(REGRESSION_SUITE_FILE, json.dumps(suite, indent=2, ensure_ascii=False))
    return suite


def _simuler_routage_complet(
    suite: list[dict[str, Any]],
    config: dict[str, Any],
    dataset: dict[str, Any],
    embed_model: str,
) -> tuple[float, list[str]]:
    """
    Rejoue toute la suite de régression contre le moteur sémantique (rôles + intentions + seuils)
    et retourne (taux_reussite_pct, liste_regressions).
    """
    if not suite:
        return 100.0, []

    routes_data = dataset.get("routes", {})
    intentions_data = dataset.get("intentions", {})
    seuils = config.get("routage", {}).get("seuils", {})
    seuil_conf = float(seuils.get("confiance_directe", 0.63))
    seuil_marge = float(seuils.get("marge_directe", 0.05))
    seuil_int = float(seuils.get("intention_outil", 0.58))

    vecs_roles = {
        role: ollama.embed(model=embed_model, input=phrases, options={"num_gpu": 0}, keep_alive=-1)["embeddings"]
        for role, phrases in routes_data.items()
        if phrases
    }
    vecs_intentions = {
        intent: ollama.embed(model=embed_model, input=phrases, options={"num_gpu": 0}, keep_alive=-1)["embeddings"]
        for intent, phrases in intentions_data.items()
        if phrases
    }

    reussis = 0
    echecs_replay: list[str] = []

    for cas in suite:
        prompt = str(cas["prompt"])
        v_p = ollama.embed(model=embed_model, input=prompt, options={"num_gpu": 0}, keep_alive=-1)["embeddings"][0]

        # 1. Évaluation du rôle prédit (top-3 cosinus moyen)
        role_predit = "superviseur"
        if vecs_roles:
            scores_r = []
            for role, vecs in vecs_roles.items():
                sims = sorted((_cosinus(v_p, v) for v in vecs), reverse=True)
                top = sims[: min(3, len(sims))]
                scores_r.append((role, sum(top) / len(top)))
            scores_r.sort(key=lambda x: x[1], reverse=True)
            top1_r, top1_s = scores_r[0]
            top2_s = scores_r[1][1] if len(scores_r) > 1 else 0.0
            if top1_s >= seuil_conf and (top1_s - top2_s) >= seuil_marge:
                role_predit = top1_r
            else:
                # Même en escalade superviseur, le candidat principal reste top1_r
                role_predit = top1_r

        # 2. Évaluation des intentions outils activées
        intentions_predites: set[str] = set()
        for intent, vecs in vecs_intentions.items():
            sims = sorted((_cosinus(v_p, v) for v in vecs), reverse=True)
            top = sims[: min(3, len(sims))]
            if (sum(top) / len(top)) >= seuil_int:
                intentions_predites.add(intent)

        role_att = cas.get("role_attendu")
        int_att = cas.get("intention_attendue")

        ok_cas = True
        if role_att and role_predit != role_att:
            ok_cas = False
            echecs_replay.append(f"Rôle attendu `{role_att}` != `{role_predit}` sur « {prompt[:55]} »")
        if int_att and int_att not in intentions_predites:
            ok_cas = False
            echecs_replay.append(f"Intention attendue `{int_att}` manquée sur « {prompt[:55]} »")

        if ok_cas:
            reussis += 1

    taux = round((reussis / float(len(suite))) * 100.0, 2)
    return taux, echecs_replay


def _valider_phrase_generaliste(
    phrase: str,
    section: str,
    cible: str,
    dataset: dict[str, Any],
    embed_model: str,
) -> tuple[bool, str]:
    propre = phrase.strip()
    if len(propre.split()) < 4:
        return False, "rejetée : trop courte (< 4 mots)"
    if _contient_secret_potentiel(propre):
        return False, "rejetée : donnée sensible détectée"

    existants = dataset.get(section, {}).get(cible, [])
    if propre.lower() in [e.lower() for e in existants]:
        return False, "rejetée : déjà présente"

    vec_new = ollama.embed(model=embed_model, input=propre, options={"num_gpu": 0}, keep_alive=-1)["embeddings"][0]
    pire_concurrent = ("aucun", 0.0)
    sim_cible_max = 0.0

    for nom_cat, phrases_cat in dataset.get(section, {}).items():
        if not phrases_cat:
            continue
        vecs = ollama.embed(model=embed_model, input=phrases_cat, options={"num_gpu": 0}, keep_alive=-1)["embeddings"]
        sim_max = max(_cosinus(vec_new, v) for v in vecs)
        if nom_cat == cible:
            sim_cible_max = sim_max
            if sim_max > 0.90:
                return False, f"rejetée : redondante avec '{cible}' (sim={sim_max:.2f})"
        else:
            if sim_max > pire_concurrent[1]:
                pire_concurrent = (nom_cat, sim_max)
            if section == "routes" and sim_max > 0.81:
                return False, f"rejetée : collision sémantique avec le rôle '{nom_cat}' (sim={sim_max:.2f})"

    return True, f"validée (sim_cible={sim_cible_max:.2f}, plus proche='{pire_concurrent[0]}' à {pire_concurrent[1]:.2f})"


def _backtester_seuil_routage(
    echanges: list[dict[str, Any]],
    dataset: dict[str, Any],
    config: dict[str, Any],
    param_court: str,
    nouvelle_valeur: float,
    embed_model: str,
) -> tuple[bool, str]:
    if param_court not in ("confiance_directe", "marge_directe"):
        return True, "validation directe (paramètre hors matrice de marge)"

    routes_cfg = config.get("routes", {})
    role_par_modele = {m: cfg.get("role", "") for m, cfg in routes_cfg.items()}
    routes_data = dataset.get("routes", {})
    if not routes_data:
        return True, "dataset vide"

    vecs_roles = {
        role: ollama.embed(model=embed_model, input=phrases, options={"num_gpu": 0}, keep_alive=-1)["embeddings"]
        for role, phrases in routes_data.items()
        if phrases
    }

    echantillon = [
        e for e in echanges[-60:]
        if e.get("prompt") and e.get("modele") in role_par_modele and e.get("raison") != "règle:phatique"
    ]
    if len(echantillon) < 5:
        return True, "historique court -> validation acceptée"

    seuils_actuels = dict(config.get("routage", {}).get("seuils", {}))
    seuils_test = dict(seuils_actuels)
    seuils_test[param_court] = nouvelle_valeur

    def _evaluer_jeu(seuils_eval: dict[str, float]) -> tuple[int, int]:
        escalades_utiles = 0
        escalades_inutiles = 0
        s_conf = float(seuils_eval.get("confiance_directe", 0.63))
        s_marge = float(seuils_eval.get("marge_directe", 0.05))

        for e in echantillon:
            v_p = ollama.embed(model=embed_model, input=str(e["prompt"]), options={"num_gpu": 0}, keep_alive=-1)["embeddings"][0]
            scores = []
            for role, vecs in vecs_roles.items():
                sims = sorted((_cosinus(v_p, v) for v in vecs), reverse=True)
                top = sims[: min(3, len(sims))]
                scores.append((role, sum(top) / len(top)))
            scores.sort(key=lambda x: x[1], reverse=True)
            top1 = scores[0][1]
            marge = top1 - (scores[1][1] if len(scores) > 1 else 0.0)
            passe_superviseur = not (top1 >= s_conf and marge >= s_marge)

            if e.get("echec_detecte") and passe_superviseur:
                escalades_utiles += 1
            elif e.get("audite_ok") and passe_superviseur:
                escalades_inutiles += 1
        return escalades_utiles, escalades_inutiles

    u_avant, i_avant = _evaluer_jeu(seuils_actuels)
    u_apres, i_apres = _evaluer_jeu(seuils_test)

    score_avant = (u_avant * 2) - i_avant
    score_apres = (u_apres * 2) - i_apres
    if score_apres < score_avant:
        return False, f"backtest défavorable (score {score_avant} -> {score_apres}, escalades inutiles {i_avant} -> {i_apres})"
    return True, f"backtest validé (score {score_avant} -> {score_apres}, échecs interceptés {u_avant} -> {u_apres})"


# ---------------- CONSOLIDATION MÉMOIRE LONG TERME ----------------

def _consolider_memoire_long_terme(
    echanges: list[dict[str, Any]],
    juge: str,
    journal: list[str],
    deadline: float,
) -> tuple[int, int]:
    non_consolides = [
        (i, e) for i, e in enumerate(echanges)
        if not e.get("consolide_memoire") and e.get("raison") != "règle:phatique" and e.get("prompt")
    ]
    if len(non_consolides) < 2:
        return 0, 0

    par_jour: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for idx, e in non_consolides:
        jour = str(e.get("date", datetime.now().date().isoformat()))[:10]
        par_jour[jour].append((idx, e))

    episodes_existants: dict[str, str] = {}
    if MEMOIRE_EPISODIQUE_FILE.exists():
        for ligne in MEMOIRE_EPISODIQUE_FILE.read_text(encoding="utf-8").splitlines():
            try:
                obj = json.loads(ligne)
                if obj.get("date") and obj.get("resume"):
                    episodes_existants[obj["date"]] = obj["resume"]
            except json.JSONDecodeError:
                continue

    memoire_actuelle = charger_memoire_utile()
    cles_connues = list(memoire_actuelle.keys())
    nb_episodes_maj = 0
    nb_faits_ajoutes = 0

    _log("\n[Phase 3 — Consolidation Mémoire] Synthèse épisodique & faits durables...", journal)

    for jour, items in sorted(par_jour.items()):
        if time.monotonic() >= deadline:
            _log("  [Temps] Budget horaire atteint, arrêt propre de la consolidation.", journal)
            break

        extrait_conv = "\n".join(
            f"- Moi : {str(e.get('prompt', ''))[:220]}\n  IA : {str(e.get('reponse', ''))[:240]}"
            for _, e in items[-20:]
        )
        resume_precedent = episodes_existants.get(jour, "Aucun")

        prompt_memoire = f"""Tu gères la mémoire à long terme d'Eon, l'assistant IA de Bryan.
Analyse ces échanges du {jour} et produis un JSON :
1. `resume_journee` : résumé factuel en 1 ou 2 phrases des sujets traités et actions réalisées (fusionne avec le résumé existant s'il y en a un).
2. `faits_utilisateurs` : liste de faits durables sur Bryan, sa machine, son OS, ses outils ou ses projets (clé courte en snake_case + valeur précise).
   INTERDIT d'enregistrer des mots de passe, tokens ou informations temporaires.

RÉSUMÉ EXISTANT POUR LE {jour} : {resume_precedent}
CLÉS DÉJÀ CONNUES : {json.dumps(cles_connues, ensure_ascii=False)}

ÉCHANGES DU {jour} :
{extrait_conv}

Réponds UNIQUEMENT en JSON valide :
{{
  "resume_journee": "...",
  "faits_utilisateurs": [{{"cle": "exemple_cle", "valeur": "exemple_valeur"}}]
}}"""

        try:
            res = ollama.chat(
                model=juge,
                messages=[{"role": "user", "content": prompt_memoire}],
                format="json",
                options={"temperature": 0.1, "num_ctx": 4096, "num_thread": 6, "num_predict": 320},
                keep_alive="1m",
            )
            data = _extraire_json_propre(res["message"]["content"]) or {}
        except Exception as err:
            _log(f"  -> [Erreur synthèse mémoire {jour}] : {err}", journal)
            continue

        resume = str(data.get("resume_journee") or "").strip()
        if len(resume) >= 15 and not _contient_secret_potentiel(resume):
            episodes_existants[jour] = resume
            nb_episodes_maj += 1
            _log(f"  [+] Épisode [{jour}] : {resume}", journal)

        for fait in data.get("faits_utilisateurs", []):
            if not isinstance(fait, dict):
                continue
            cle = str(fait.get("cle", "")).strip()
            val = str(fait.get("valeur", "")).strip()
            if len(cle) >= 2 and len(val) >= 2 and not _contient_secret_potentiel(f"{cle} {val}"):
                msg_mem = memoriser_fait_utilisateur(cle, val)
                if not msg_mem.startswith("Refus"):
                    nb_faits_ajoutes += 1
                    _log(f"  [+] Mémoire utile -> {msg_mem}", journal)

        for idx, _ in items:
            echanges[idx]["consolide_memoire"] = True

        _pause_thermique_adaptative(journal, pause_base_sec=3.0)

    if nb_episodes_maj > 0:
        lignes_ep = [
            json.dumps({"date": d, "resume": r}, ensure_ascii=False)
            for d, r in sorted(episodes_existants.items())[-40:]
        ]
        _ecrire_atomique(MEMOIRE_EPISODIQUE_FILE, "\n".join(lignes_ep) + "\n")

    return nb_episodes_maj, nb_faits_ajoutes


# ---------------- PIPELINE PRINCIPAL D'OPTIMISATION ----------------

def executer_optimisation_nocturne(duree_max_minutes: int = 55) -> str:
    """
    Exécute une optimisation complète, thermiquement maîtrisée et protégée contre les régressions :
    1. Snapshot de backup + mesure baseline du replay de régression.
    2. Audit silencieux des échanges passés non vérifiés.
    3. Diagnostic approfondi + ajustements fins par dichotomie.
    4. Consolidation mémoire long terme.
    5. Replay de régression final : validation ou rollback automatique + rapport détaillé.
    """
    debut_ts = time.monotonic()
    deadline = debut_ts + max(5, int(duree_max_minutes)) * 60
    dernier_heartbeat = debut_ts
    horodatage_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    journal: list[str] = []
    explications_detaillees: list[str] = []
    actions_cles: list[str] = []

    echanges = charger_tous_echanges()
    if not echanges:
        return "Aucun historique à analyser."

    config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    dataset = (
        json.loads(TRAINING_FILE.read_text(encoding="utf-8"))
        if TRAINING_FILE.exists()
        else {"routes": {}, "intentions": {}, "consentement": {}}
    )
    hist_opti = _charger_historique_optimisations()
    chemin_backup = _creer_snapshot_backup(horodatage_id)

    juge = config.get("modele_optimiseur", "qwen2.5-coder:14b")
    embed_model = _obtenir_embed_model(config)
    t_gpu_init, t_cpu_init = _lire_temperatures()

    _log(
        f"\n[Optimiseur] Démarrage session {horodatage_id} | Modèle : {juge} (num_thread=6) | "
        f"Budget temps : {duree_max_minutes} min | Températures : GPU={t_gpu_init or '?'}°C, CPU={t_cpu_init or '?'}°C",
        journal,
    )
    _log(f"[Backup] Snapshot initial sauvegardé : {chemin_backup}", journal)

    # Mesure de référence avant toute modification (Replay de régression)
    suite_regression = _charger_suite_regression(config)
    score_replay_avant, echecs_replay_avant = _simuler_routage_complet(
        suite_regression, config, dataset, embed_model
    )
    _log(
        f"[Replay Baseline] Score suite de régression ({len(suite_regression)} cas) : {score_replay_avant:.2f} %",
        journal,
    )

    def _heartbeat_si_necessaire(etape_label: str) -> None:
        nonlocal dernier_heartbeat
        maintenant = time.monotonic()
        if (maintenant - dernier_heartbeat) >= 120.0:
            ecoule_min = (maintenant - debut_ts) / 60.0
            tg, tc = _lire_temperatures()
            _log(
                f"  [Progression {ecoule_min:.1f} min] {etape_label} | "
                f"Modifications validées : {len(actions_cles)} | GPU={tg or '?'}°C, CPU={tc or '?'}°C",
                journal,
            )
            dernier_heartbeat = maintenant

    catalogue_modeles = "\n".join(
        f"- {nom} (rôle: {cfg.get('role')}, temp: {cfg.get('temperature')}, "
        f"num_ctx: {cfg.get('num_ctx')}, num_predict: {cfg.get('num_predict')}, "
        f"web: {cfg.get('web_tools')}, fichiers: {cfg.get('file_tools')}, terminal: {cfg.get('shell_tools')})"
        for nom, cfg in config.get("routes", {}).items()
    )

    # =========================================================================
    # PHASE 1 : AUDIT SILENCIEUX DES ÉCHANGES RÉCENTS
    # =========================================================================
    a_auditer = [
        (i, e) for i, e in enumerate(echanges)
        if not e.get("echec_detecte") and not e.get("audite_ok") and e.get("raison") != "règle:phatique"
    ][-30:]

    if a_auditer:
        _log(f"\n[Phase 1 — Audit silencieux] Inspection approfondie de {len(a_auditer)} échange(s)...", journal)
        for debut_lot in range(0, len(a_auditer), 8):
            if time.monotonic() >= deadline:
                break
            lot = a_auditer[debut_lot : debut_lot + 8]
            bloc = "\n".join(
                f"[{i}] Modèle={e.get('modele')} | Prompt={e.get('prompt')} | Réponse={str(e.get('reponse'))[:300]}"
                for i, e in lot
            )
            prompt_audit = (
                f"Modèles disponibles :\n{catalogue_modeles}\n\n"
                "Analyse chaque échange ci-dessous. Liste UNIQUEMENT les index [i] où la réponse ou le routage "
                "présente un défaut réel (hallucination, info obsolète non vérifiée sur le web, commande simulée, "
                "refus injustifié ou mauvais modèle choisi).\n"
                f"{bloc}\n\n"
                'Réponds en JSON : {"echecs": [{"index": 0, "raison": "explication précise"}], "bilan_lot": "résumé court"}'
            )
            try:
                res = ollama.chat(
                    model=juge,
                    messages=[{"role": "user", "content": prompt_audit}],
                    format="json",
                    options={"temperature": 0.0, "num_ctx": 4096, "num_thread": 6, "num_predict": 400},
                    keep_alive="2m",
                )
                data_audit = _extraire_json_propre(res["message"]["content"]) or {}
                rates = {
                    int(item["index"]): str(item.get("raison", "Audit IA"))
                    for item in data_audit.get("echecs", [])
                    if isinstance(item, dict) and "index" in item
                }
                if data_audit.get("bilan_lot"):
                    _log(f"  -> Lot {debut_lot // 8 + 1} : {data_audit['bilan_lot']}", journal)

                for idx, e in lot:
                    if idx in rates:
                        echanges[idx]["echec_detecte"] = True
                        echanges[idx]["raison_echec"] = f"Audit IA : {rates[idx]}"
                        _log(f"  -> [Échec détecté #{idx}] « {e.get('prompt')} » : {rates[idx]}", journal)
                    else:
                        echanges[idx]["audite_ok"] = True
            except Exception as err:
                _log(f"  -> [Erreur audit lot] : {err}", journal)

            _heartbeat_si_necessaire("Phase 1 (Audit)")
            _pause_thermique_adaptative(journal, pause_base_sec=3.5)

    # =========================================================================
    # PHASE 2 : DIAGNOSTIC & AJUSTEMENTS FINS PAR DICHOTOMIE
    # =========================================================================
    echecs = [
        (i, e) for i, e in enumerate(echanges)
        if e.get("echec_detecte") and not e.get("traite_optimiseur") and e.get("raison") != "règle:phatique"
    ]

    config_modifiee = False
    dataset_modifie = False

    if echecs:
        contexte_backups = _formater_contexte_backups(hist_opti)
        _log(f"\n[Phase 2 — Optimisation fine & Dichotomie] {len(echecs)} échec(s) à traiter...", journal)

        for rang, (idx, echec) in enumerate(echecs[-18:], start=1):
            if time.monotonic() >= deadline:
                _log("  [Temps] Budget horaire atteint, passage au bilan final.", journal)
                break

            suite = echanges[idx + 1]["prompt"] if idx + 1 < len(echanges) else ""
            seuils_actuels = config.get("routage", {}).get("seuils", {})

            prompt_juge = f"""Tu es l'ingénieur d'optimisation d'Eon (assistant IA local sur Linux Fedora).
Ta mission est d'améliorer la précision du système par petits pas convergents (méthode dichotomique), JAMAIS par sauts brutaux.

{contexte_backups}

ÉTAT ACTUEL DE LA CONFIGURATION :
- Seuils de routage (`routage.seuils`) : {json.dumps(seuils_actuels)}
  (Bornes autorisées : confiance_directe [0.48..0.78], marge_directe [0.015..0.14], intention_outil [0.48..0.74], insatisfaction [0.50..0.75])
- Modèles & hyperparamètres :
{catalogue_modeles}
- Règles système (`regles_apprises`) : {json.dumps(config.get("regles_apprises", []), ensure_ascii=False)}
- Rôles disponibles dans `entrainement_routage.json` : {list(dataset.get("routes", {}).keys())}
- Intentions disponibles dans `entrainement_routage.json` : {list(dataset.get("intentions", {}).keys())}

ÉCHANGE EN ÉCHEC #{idx} :
- Prompt utilisateur : {echec.get('prompt')}
- Modèle choisi : {echec.get('modele')} (raison: {echec.get('raison')}, score: {echec.get('score')})
- Réponse IA : {str(echec.get('reponse'))[:500]}
- Symptôme / raison échec : {echec.get('raison_echec')}
- Réaction suivante de l'utilisateur : {suite}

CONSIGNES STRICTES :
1. `phrase_entrainement` doit être une PHRASE GÉNÉRALISTE décrivant le type de demande utilisateur (ex: "demander le club actuel d'un sportif", JAMAIS un nom propre spécifique comme "Bradley Barcola").
2. Si tu proposes une `valeur_cible` pour un seuil ou une `temperature` de modèle, propose une valeur mesurée proche de la valeur actuelle (le moteur appliquera en plus une dichotomie amortie).
3. Explique clairement ce que tu fais et pourquoi dans `explication_detaillee`.

Réponds UNIQUEMENT en JSON valide :
{{
  "diagnostic": "cause racine exacte",
  "explication_detaillee": "explication pédagogique complète de l'analyse, de la modification retenue et de l'effet attendu",
  "phrase_entrainement": {{
    "section": "routes ou intentions ou null",
    "cible": "nom du rôle ou de l'intention (ou null)",
    "phrase": "formulation généraliste à ajouter (ou null)"
  }},
  "ajustement_seuil": {{
    "param": "confiance_directe | marge_directe | intention_outil | insatisfaction | null",
    "valeur_cible": null
  }},
  "ajustement_modele": {{
    "modele": "nom exact du modèle dans routes (ou null)",
    "temperature_cible": null,
    "num_predict_cible": null
  }},
  "regle_systeme_ajouter": "consigne courte et générale (ou null)",
  "regle_systeme_retirer": "texte exact d'une règle obsolète à supprimer (ou null)"
}}"""

            try:
                res = ollama.chat(
                    model=juge,
                    messages=[{"role": "user", "content": prompt_juge}],
                    format="json",
                    options={"temperature": 0.1, "num_ctx": 4096, "num_thread": 6, "num_predict": 550},
                    keep_alive="2m",
                )
                verdict = _extraire_json_propre(res["message"]["content"]) or {}
            except Exception as err:
                _log(f"  [Erreur Juge #{idx}] : {err}", journal)
                _pause_thermique_adaptative(journal, pause_base_sec=4.0)
                continue

            diag = str(verdict.get("diagnostic", "")).strip()
            expl = str(verdict.get("explication_detaillee", "")).strip()
            _log(f"\n  --- [{rang}/{min(len(echecs), 18)}] Échange #{idx} : « {echec.get('prompt')} » ---", journal)
            if diag:
                _log(f"  Diagnostic  : {diag}", journal)
            if expl:
                _log(f"  Explication : {expl}", journal)

            actions_ce_tour: list[str] = []

            # A. Ajout d'une phrase généraliste dans entrainement_routage.json
            pe = verdict.get("phrase_entrainement")
            if isinstance(pe, dict):
                sec = str(pe.get("section") or "").strip()
                cib = str(pe.get("cible") or "").strip()
                phr = str(pe.get("phrase") or "").strip()
                if sec in ("routes", "intentions") and cib in dataset.get(sec, {}) and len(phr) >= 10:
                    ok_phr, detail_phr = _valider_phrase_generaliste(phr, sec, cib, dataset, embed_model)
                    if ok_phr:
                        dataset[sec][cib].append(phr)
                        dataset[sec][cib] = dataset[sec][cib][-35:]
                        dataset_modifie = True
                        act = f"Dataset [{sec}.{cib}] + « {phr} » ({detail_phr})"
                        actions_ce_tour.append(act)
                        _log(f"  [+] {act}", journal)
                    else:
                        _log(f"  [=] Phrase ignorée ({detail_phr}) : « {phr} »", journal)

            # B. Ajustement dichotomique d'un seuil de routage + backtest
            aj_s = verdict.get("ajustement_seuil")
            if isinstance(aj_s, dict) and aj_s.get("param") and aj_s.get("valeur_cible") is not None:
                p_court = str(aj_s["param"]).strip()
                cle_complete = f"routage.seuils.{p_court}"
                if cle_complete in SPECS_PARAMETRES:
                    try:
                        val_souhaitee = float(aj_s["valeur_cible"])
                        val_act = float(seuils_actuels.get(p_court, SPECS_PARAMETRES[cle_complete]["defaut"]))
                        val_dicho, motif_dicho = _calculer_valeur_dichotomique(
                            cle_complete, val_act, val_souhaitee, SPECS_PARAMETRES[cle_complete], hist_opti
                        )
                        if val_dicho != val_act:
                            ok_bt, detail_bt = _backtester_seuil_routage(
                                echanges, dataset, config, p_court, val_dicho, embed_model
                            )
                            if ok_bt:
                                config.setdefault("routage", {}).setdefault("seuils", {})[p_court] = val_dicho
                                config_modifiee = True
                                act = f"Seuil `{p_court}` : {val_act:.3f} -> {val_dicho:.3f} [{motif_dicho} | {detail_bt}]"
                                actions_ce_tour.append(act)
                                _log(f"  [+] {act}", journal)
                            else:
                                _log(f"  [!] Seuil `{p_court}` ({val_act:.3f} -> {val_dicho:.3f}) annulé : {detail_bt}", journal)
                    except (TypeError, ValueError):
                        pass

            # C. Ajustement dichotomique des hyperparamètres d'un modèle
            aj_m = verdict.get("ajustement_modele")
            if isinstance(aj_m, dict) and aj_m.get("modele") in config.get("routes", {}):
                nom_m = str(aj_m["modele"])
                cfg_m = config["routes"][nom_m]

                if aj_m.get("temperature_cible") is not None:
                    try:
                        t_souhaitee = float(aj_m["temperature_cible"])
                        t_act = float(cfg_m.get("temperature", 0.2))
                        cle_t = f"routes.{nom_m}.temperature"
                        t_dicho, motif_t = _calculer_valeur_dichotomique(
                            cle_t, t_act, t_souhaitee, SPECS_PARAMETRES["temperature"], hist_opti
                        )
                        if t_dicho != t_act:
                            cfg_m["temperature"] = t_dicho
                            config_modifiee = True
                            act = f"Température `{nom_m}` : {t_act:.2f} -> {t_dicho:.2f} [{motif_t}]"
                            actions_ce_tour.append(act)
                            _log(f"  [+] {act}", journal)
                    except (TypeError, ValueError):
                        pass

                if aj_m.get("num_predict_cible") is not None:
                    try:
                        np_souhaite = int(aj_m["num_predict_cible"])
                        np_act = int(cfg_m.get("num_predict", 512))
                        delta_np = max(-256, min(256, np_souhaite - np_act))
                        np_nouveau = max(60, min(2048, np_act + delta_np))
                        if abs(np_nouveau - np_act) >= 32:
                            cfg_m["num_predict"] = np_nouveau
                            config_modifiee = True
                            act = f"`num_predict` `{nom_m}` : {np_act} -> {np_nouveau}"
                            actions_ce_tour.append(act)
                            _log(f"  [+] {act}", journal)
                    except (TypeError, ValueError):
                        pass

            # D. Règles système apprises
            regles_actuelles = config.setdefault("regles_apprises", [])
            r_suppr = str(verdict.get("regle_systeme_retirer") or "").strip()
            if r_suppr and r_suppr in regles_actuelles:
                regles_actuelles.remove(r_suppr)
                config_modifiee = True
                act = f"Règle retirée : « {r_suppr} »"
                actions_ce_tour.append(act)
                _log(f"  [-] {act}", journal)

            r_ajout = str(verdict.get("regle_systeme_ajouter") or "").strip()
            if len(r_ajout) >= 12 and r_ajout.lower() != "null" and not _contient_secret_potentiel(r_ajout):
                if r_ajout not in regles_actuelles:
                    regles_actuelles.append(r_ajout)
                    config["regles_apprises"] = regles_actuelles[-12:]
                    config_modifiee = True
                    act = f"Règle ajoutée : « {r_ajout} »"
                    actions_ce_tour.append(act)
                    _log(f"  [+] {act}", journal)

            if actions_ce_tour:
                actions_cles.extend(actions_ce_tour)
                explications_detaillees.append(
                    f"### Échange #{idx} — « {echec.get('prompt')} »\n"
                    f"- **Diagnostic :** {diag}\n"
                    f"- **Explication :** {expl}\n"
                    + "\n".join(f"- **Action :** {a}" for a in actions_ce_tour)
                )
            else:
                _log("  [=] Aucune modification retenue après vérification.", journal)

            echanges[idx]["traite_optimiseur"] = True
            _heartbeat_si_necessaire(f"Phase 2 ({rang}/{min(len(echecs), 18)})")
            _pause_thermique_adaptative(journal, pause_base_sec=4.5)
    else:
        _log("\n[Phase 2 — Diagnostic] Aucun nouvel échec à corriger.", journal)

    # =========================================================================
    # PHASE 3 : CONSOLIDATION MÉMOIRE LONG TERME
    # =========================================================================
    nb_episodes, nb_faits = _consolider_memoire_long_terme(echanges, juge, journal, deadline)

    # =========================================================================
    # PHASE 4 : REPLAY DE RÉGRESSION FINAL, ROLLBACK SI DÉGRADATION & RAPPORT
    # =========================================================================
    statut_replay = f"Stable ({score_replay_avant:.2f} %)"
    rollback_effectue = False

    if config_modifiee or dataset_modifie:
        score_replay_apres, echecs_replay_apres = _simuler_routage_complet(
            suite_regression, config, dataset, embed_model
        )
        _log(
            f"\n[Phase 4 — Replay de Régression] Score avant : {score_replay_avant:.2f} % -> "
            f"Score après : {score_replay_apres:.2f} % ({len(suite_regression)} cas)",
            journal,
        )

        if score_replay_apres < (score_replay_avant - 0.5):
            rollback_effectue = _restaurer_snapshot_backup(horodatage_id)
            config_modifiee = False
            dataset_modifie = False
            actions_cles.clear()
            statut_replay = (
                f"RÉGRESSION DÉTECTÉE ({score_replay_avant:.2f} % -> {score_replay_apres:.2f} %) "
                f"-> Rollback automatique effectué"
            )
            _log(f"  [!] {statut_replay}", journal)
            for err_rep in echecs_replay_apres[:6]:
                _log(f"      - {err_rep}", journal)
        else:
            statut_replay = f"Validé ({score_replay_avant:.2f} % -> {score_replay_apres:.2f} %)"
            if config_modifiee:
                _ecrire_atomique(CONFIG_FILE, json.dumps(config, indent=2, ensure_ascii=False))
            if dataset_modifie:
                _ecrire_atomique(TRAINING_FILE, json.dumps(dataset, indent=2, ensure_ascii=False))

    sauvegarder_tous_echanges(echanges)
    compacter_base_donnees()

    # Décharge proprement le modèle juge de la RTX 5050
    try:
        ollama.generate(model=juge, prompt="", keep_alive=0)
    except Exception:
        pass

    duree_totale_min = (time.monotonic() - debut_ts) / 60.0
    t_gpu_fin, t_cpu_fin = _lire_temperatures()

    resume_court = (
        f"{len(echecs)} échec(s) analysé(s), {len(actions_cles)} ajustement(s) retenu(s), "
        f"replay={statut_replay}, mémoire +{nb_episodes} épisode(s) / +{nb_faits} fait(s) en {duree_totale_min:.1f} min."
    )
    hist_opti["sessions"].append(
        {
            "date": datetime.now().isoformat(timespec="seconds"),
            "backup": chemin_backup,
            "duree_min": round(duree_totale_min, 2),
            "rollback": rollback_effectue,
            "resume_court": resume_court,
            "actions_cles": actions_cles,
        }
    )
    _sauvegarder_historique_optimisations(hist_opti)

    lignes_rapport = [
        f"# Rapport d'Optimisation Eon — {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
        "",
        f"- **Durée totale :** {duree_totale_min:.1f} minute(s) | **Backup :** `{chemin_backup}`",
        f"- **Replay de régression :** {statut_replay}",
        f"- **Températures :** Début (GPU {t_gpu_init or '?'}°C / CPU {t_cpu_init or '?'}°C) -> Fin (GPU {t_gpu_fin or '?'}°C / CPU {t_cpu_fin or '?'}°C)",
        f"- **Bilan global :** {resume_court}",
        "",
        "## 1. Modifications appliquées et justifications",
        "",
    ]
    if rollback_effectue:
        lignes_rapport.append(
            "**Alerte Replay :** Les modifications proposées ont dégradé la précision sur la suite de régression "
            "et ont été automatiquement annulées."
        )
    elif explications_detaillees:
        lignes_rapport.extend(explications_detaillees)
    else:
        lignes_rapport.append("Aucune modification de paramètre nécessaire lors de cette passe (stabilité confirmée).")

    lignes_rapport.extend(
        [
            "",
            "## 2. Trajectoires actuelles des paramètres (Convergence dichotomique)",
            "",
        ]
    )
    for param, info in sorted(hist_opti.get("trajectoires_parametres", {}).items()):
        vals = " -> ".join(f"{v:.3f}" for v in info.get("valeurs", []))
        lignes_rapport.append(f"- **`{param}`** : `{vals}` (inversions : {info.get('inversions', 0)})")

    rapport_md = "\n".join(lignes_rapport) + "\n"
    _ecrire_atomique(RAPPORT_FINAL_FILE, rapport_md)

    _log("\n" + "=" * 70, journal)
    _log(f"[BILAN FINAL] {resume_court}", journal)
    for act in actions_cles:
        _log(f"  • {act}", journal)
    _log(f"[RAPPORT COMPLET] Écrit dans : {RAPPORT_FINAL_FILE}", journal)
    _log("=" * 70, journal)

    entete_log = f"\n{'=' * 70}\nSESSION D'OPTIMISATION — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'=' * 70}\n"
    existant = OPTIMISEUR_LOG_FILE.read_text(encoding="utf-8") if OPTIMISEUR_LOG_FILE.exists() else ""
    _ecrire_atomique(OPTIMISEUR_LOG_FILE, (existant + entete_log + "\n".join(journal) + "\n")[-120000:])

    return rapport_md


# ---------------- BOUCLE D'AUTO-ÉVALUATION NOCTURNE / VEILLE ----------------

class PlanificateurNocturneEon:
    """
    Surveille discrètement l'inactivité utilisateur et l'état matériel du portable
    pour lancer l'auto-évaluation la nuit (01h00–05h30) ou après une longue pause.
    """

    def __init__(
        self,
        heure_debut: int = 1,
        heure_fin: int = 5,
        inactivite_min_sec: float = 900.0,
        seuil_echanges_en_attente: int = 6,
    ) -> None:
        self.heure_debut = heure_debut
        self.heure_fin = heure_fin
        self.inactivite_min_sec = inactivite_min_sec
        self.seuil_echanges_en_attente = seuil_echanges_en_attente
        self._dernier_activite_ts: float = time.monotonic()
        self._derniere_execution_date: str = ""
        self._arret = threading.Event()
        self._en_cours = threading.Lock()

    def notifier_activite_utilisateur(self) -> None:
        self._dernier_activite_ts = time.monotonic()

    def _conditions_materielles_ok(self) -> tuple[bool, str]:
        sur_secteur, bat_pct = _etat_alimentation_portable()
        if not sur_secteur and bat_pct < 55:
            return False, f"sur batterie faible ({bat_pct}%)"
        t_gpu, t_cpu = _lire_temperatures()
        if (t_gpu is not None and t_gpu > 68.0) or (t_cpu is not None and t_cpu > 74.0):
            return False, f"température trop haute (GPU={t_gpu}°C, CPU={t_cpu}°C)"
        return True, "ok"

    def _doit_se_declencher(self) -> bool:
        maintenant_dt = datetime.now()
        aujourd_hui = maintenant_dt.date().isoformat()
        if self._derniere_execution_date == aujourd_hui:
            return False

        inactivite = time.monotonic() - self._dernier_activite_ts
        if inactivite < self.inactivite_min_sec:
            return False

        dans_fenetre_nuit = self.heure_debut <= maintenant_dt.hour <= self.heure_fin
        if not dans_fenetre_nuit:
            return False

        echanges = charger_tous_echanges()
        non_verifies = sum(
            1 for e in echanges
            if not e.get("audite_ok") and not e.get("traite_optimiseur") and e.get("raison") != "règle:phatique"
        )
        return non_verifies >= self.seuil_echanges_en_attente

    def boucle_surveillance(self) -> None:
        while not self._arret.is_set():
            try:
                if self._doit_se_declencher():
                    ok_mat, _ = self._conditions_materielles_ok()
                    if ok_mat and self._en_cours.acquire(blocking=False):
                        try:
                            self._derniere_execution_date = datetime.now().date().isoformat()
                            executer_optimisation_nocturne(duree_max_minutes=45)
                        finally:
                            self._en_cours.release()
            except Exception:
                pass
            self._arret.wait(timeout=180.0)

    def demarrer(self) -> threading.Thread:
        t = threading.Thread(target=self.boucle_surveillance, name="EonNightlyOptimizer", daemon=True)
        t.start()
        return t

    def arreter(self) -> None:
        self._arret.set()


if __name__ == "__main__":
    # Permet de lancer l'optimiseur manuellement ou via un timer systemd --user
    if "--daemon" in sys.argv:
        planif = PlanificateurNocturneEon()
        planif.boucle_surveillance()
    else:
        duree = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 55
        executer_optimisation_nocturne(duree_max_minutes=duree)