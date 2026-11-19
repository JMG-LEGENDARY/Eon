import json
import math
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import ollama

from memory_manager import (
    BASE_DIR,
    MEMOIRE_EPISODIQUE_FILE,
    charger_memoire_utile,
    charger_tous_echanges,
    memoriser_fait_utilisateur,
    sauvegarder_tous_echanges,
)

CONFIG_FILE = BASE_DIR / "config_ia.json"
CONFIG_BACKUP_FILE = BASE_DIR / "config_ia.bak.json"
OPTIMISEUR_LOG_FILE = BASE_DIR / "optimiseur_journal.log"

BORNES_SEUILS = {
    "confiance": (0.45, 0.68),
    "marge": (0.015, 0.10),
    "insatisfaction": (0.58, 0.72),
}

REGEX_SECRET = re.compile(
    r"(sudo\s+-S|mot\s+de\s+passe|password|passwd|api[_-]?key|token\s*=|secret\s*=)",
    re.IGNORECASE,
)


def _log(message: str, journal: list[str]) -> None:
    print(message, flush=True)
    journal.append(message)


def _sauvegarder_journal(journal: list[str]) -> None:
    if not journal:
        return
    horodatage = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entete = f"\n{'=' * 70}\nSESSION D'OPTIMISATION — {horodatage}\n{'=' * 70}\n"
    nouveau_bloc = entete + "\n".join(journal) + "\n"

    existant = OPTIMISEUR_LOG_FILE.read_text(encoding="utf-8") if OPTIMISEUR_LOG_FILE.exists() else ""
    contenu_total = (existant + nouveau_bloc)[-60000:]
    _ecrire_atomique(OPTIMISEUR_LOG_FILE, contenu_total)


def _lire_memoire_optimiseur(max_chars: int = 3000) -> str:
    if not OPTIMISEUR_LOG_FILE.exists():
        return "Aucune session d'optimisation précédente."
    contenu = OPTIMISEUR_LOG_FILE.read_text(encoding="utf-8").strip()
    return contenu[-max_chars:] if contenu else "Journal vide."


def _sauvegarder_backup_config() -> None:
    if CONFIG_FILE.exists():
        shutil.copy2(CONFIG_FILE, CONFIG_BACKUP_FILE)


def _contient_secret_potentiel(texte: str) -> bool:
    return bool(REGEX_SECRET.search(texte))


def _cosinus(v1: list[float], v2: list[float]) -> float:
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    return sum(a * b for a, b in zip(v1, v2)) / (n1 * n2) if n1 > 0.0 and n2 > 0.0 else 0.0


def _exemple_route_valide(phrase: str, modele_cible: str, config: dict) -> tuple[bool, str]:
    mots_interdits = ("répondre aux", "prompts phatiques", "modèle choisi", "routeur")
    if len(phrase.split()) < 4:
        return False, "Rejeté : formulation trop courte (< 4 mots)."
    if any(m in phrase for m in mots_interdits):
        return False, "Rejeté : ressemble à une consigne système et non à une requête utilisateur."
    if _contient_secret_potentiel(phrase):
        return False, "Rejeté : contient un mot-clé sensible (mot de passe / token)."

    embed_model = config["embed_model"]
    vec_new = ollama.embed(model=embed_model, input=phrase, options={"num_gpu": 0})["embeddings"][0]

    pire_concurrent = ("aucun", 0.0)
    max_cible = 0.0

    for modele, cfg in config["routes"].items():
        if cfg.get("role") == "phatique" or not cfg.get("examples"):
            continue
        vecs = ollama.embed(model=embed_model, input=cfg["examples"], options={"num_gpu": 0})["embeddings"]
        max_sim = max(_cosinus(vec_new, v) for v in vecs)

        if modele == modele_cible:
            max_cible = max_sim
            if max_sim > 0.88:
                return False, f"Rejeté : déjà couvert par '{modele_cible}' (similarité {max_sim:.2f} > 0.88)."
        else:
            if max_sim > pire_concurrent[1]:
                pire_concurrent = (modele, max_sim)
            if max_sim > 0.80:
                return False, f"Rejeté : collision sémantique avec '{modele}' (similarité {max_sim:.2f} > 0.80)."

    return (
        True,
        f"Validé (sim cible={max_cible:.2f}, plus proche concurrent='{pire_concurrent[0]}' à {pire_concurrent[1]:.2f}).",
    )


def _obtenir_embed_model(config: dict) -> str:
    return (
        config.get("embed_model")
        or config.get("routage", {}).get("embed_model")
        or "bge-m3"
    )


def _signal_insatisfaction_nouveau(signal: str, config: dict) -> tuple[bool, str]:
    """Vérifie qu'un nouveau signal d'insatisfaction n'est pas déjà couvert par les exemples existants."""
    signal_propre = signal.strip().lower()
    if len(signal_propre) < 4:
        return False, "signal trop court"

    existants = [
        s.strip().lower()
        for s in config.get("signaux_insatisfaction", [])
        if isinstance(s, str) and s.strip()
    ]
    if signal_propre in existants:
        return False, "déjà présent à l'identique"
    if not existants:
        return True, "premier signal enregistré"

    embed_model = _obtenir_embed_model(config)
    try:
        rep = ollama.embed(model=embed_model, input=[signal_propre, *existants])
        vecs = np.array(rep["embeddings"], dtype=np.float32)
        normes = np.linalg.norm(vecs, axis=1, keepdims=True)
        normes[normes == 0] = 1.0
        vecs = vecs / normes
        sims = vecs[1:] @ vecs[0]
        sim_max = float(np.max(sims))
        if sim_max >= 0.86:
            return False, f"trop proche d'un signal existant (sim={sim_max:.2f})"
        return True, f"nouveau signal validé (sim_max={sim_max:.2f})"
    except Exception as err:
        return True, f"validé sans embedding ({err})"

def _consolider_memoire_long_terme(
    echanges: list[dict],
    juge: str,
    journal: list[str],
) -> tuple[int, int]:
    """Résume les échanges récents dans memoire_episodique.jsonl et extrait les faits durables."""
    non_consolides = [
        (i, e) for i, e in enumerate(echanges)
        if not e.get("consolide_memoire") and e.get("raison") != "règle:phatique" and e.get("prompt")
    ]
    if len(non_consolides) < 2:
        return 0, 0

    par_jour: dict[str, list[tuple[int, dict]]] = defaultdict(list)
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

    _log("\n[Phase 3 — Consolidation Mémoire] Synthèse épisodique & extraction de faits...", journal)

    for jour, items in sorted(par_jour.items()):
        extrait_conv = "\n".join(
            f"- Moi : {str(e.get('prompt', ''))[:200]}\n  IA : {str(e.get('reponse', ''))[:220]}"
            for _, e in items[-15:]
        )
        resume_precedent = episodes_existants.get(jour, "Aucun")

        prompt_memoire = f"""Tu gères la mémoire à long terme d'un assistant IA local.
Analyse ces échanges du {jour} et produis un JSON :
1. `resume_journee` : résumé factuel en 1 ou 2 phrases des sujets traités et actions réalisées (fusionne avec le résumé existant s'il y en a un).
2. `faits_utilisateurs` : liste de faits durables sur l'utilisateur, sa machine, son OS, ses outils ou ses projets (clé courte en snake_case + valeur précise).
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
                options={"temperature": 0.1, "num_ctx": 4096},
                keep_alive="1m",
            )
            data = json.loads(res["message"]["content"])
        except Exception as err:
            _log(f"  -> [Erreur synthèse mémoire {jour}] : {err}", journal)
            continue

        resume = (data.get("resume_journee") or "").strip()
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
                nb_faits_ajoutes += 1
                _log(f"  [+] Mémoire utile -> {msg_mem}", journal)

        for idx, _ in items:
            echanges[idx]["consolide_memoire"] = True

    if nb_episodes_maj > 0:
        lignes_ep = [
            json.dumps({"date": d, "resume": r}, ensure_ascii=False)
            for d, r in sorted(episodes_existants.items())[-30:]
        ]
        _ecrire_atomique(MEMOIRE_EPISODIQUE_FILE, "\n".join(lignes_ep) + "\n")

    return nb_episodes_maj, nb_faits_ajoutes


def executer_optimisation_nocturne() -> str:
    journal: list[str] = []

    echanges = charger_tous_echanges()
    if not echanges:
        return "Aucun historique à analyser."

    config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    juge = config.get("modele_optimiseur", config["route_defaut"])

    catalogue_modeles = "\n".join(
        f"- {nom} (rôle: {cfg['role']}, web: {cfg.get('web_tools')}, fichiers: {cfg.get('file_tools')}, terminal: {cfg.get('shell_tools')})"
        for nom, cfg in config["routes"].items()
    )

    _log(f"\n[Optimiseur] Démarrage de l'analyse avec : {juge} ({len(echanges)} échange(s) en archive)", journal)

    # 1. Audit IA des échanges non phatiques sans plainte explicite
    a_auditer = [
        (i, e) for i, e in enumerate(echanges)
        if not e.get("echec_detecte") and not e.get("audite_ok") and e.get("raison") != "règle:phatique"
    ][-15:]

    if a_auditer:
        _log(f"[Phase 1 — Audit silencieux] Inspection de {len(a_auditer)} échange(s)...", journal)
        bloc = "\n".join(
            f"[{i}] Modèle={e.get('modele')} | Prompt={e.get('prompt')} | Réponse={str(e.get('reponse'))[:250]}"
            for i, e in a_auditer
        )
        prompt_audit = (
            f"Modèles disponibles :\n{catalogue_modeles}\n\n"
            f"Liste UNIQUEMENT les index [i] où la RÉPONSE de l'IA est réellement mauvaise "
            f"(hallucination, commande inventée, refus injustifié, ou mauvais modèle).\n{bloc}\n\n"
            'Réponds en JSON : {"echecs": [{"index": 0, "raison": "explication courte"}], "bilan_audit": "résumé court"}'
        )
        try:
            res = ollama.chat(
                model=juge,
                messages=[{"role": "user", "content": prompt_audit}],
                format="json",
                options={"temperature": 0.0, "num_ctx": 4096},
            )
            data_audit = json.loads(res["message"]["content"])
            rates = {
                item["index"]: item.get("raison", "Audit IA")
                for item in data_audit.get("echecs", [])
                if isinstance(item, dict) and "index" in item
            }
            if data_audit.get("bilan_audit"):
                _log(f"  -> Synthèse : {data_audit['bilan_audit']}", journal)

            for idx, e in a_auditer:
                if idx in rates:
                    echanges[idx]["echec_detecte"] = True
                    echanges[idx]["raison_echec"] = f"Audit IA : {rates[idx]}"
                    _log(f"  -> [Échec repéré #{idx}] « {e.get('prompt')} » : {rates[idx]}", journal)
                else:
                    echanges[idx]["audite_ok"] = True
            if not rates:
                _log("  -> Aucun échec caché détecté.", journal)
        except Exception as err:
            _log(f"  -> [Erreur audit] : {err}", journal)

    # 2. Diagnostic et mise à jour de config_ia.json
    echecs = [
        (i, e) for i, e in enumerate(echanges)
        if e.get("echec_detecte") and not e.get("traite_optimiseur") and e.get("raison") != "règle:phatique"
    ]

    ajouts_routes = 0
    ajouts_regles = 0
    retraits_regles = 0
    ajouts_signaux = 0
    ajustements_seuils = 0
    config_modifiee = False

    if echecs:
        historique_optimiseur = _lire_memoire_optimiseur()
        _log(f"\n[Phase 2 — Diagnostic & Auto-Patch] {len(echecs)} échec(s) à traiter :", journal)
        seuil_deja_ajuste = False
        for idx, echec in echecs[-10:]:
            suite = echanges[idx + 1]["prompt"] if idx + 1 < len(echanges) else ""
            _log(f"\n  --- Échange #{idx} : « {echec.get('prompt')} » ({echec.get('modele')}) ---", journal)
            _log(f"  Symptôme : {echec.get('raison_echec')}", journal)

            prompt_juge = f"""Tu es l'optimiseur autonome d'un routeur multi-modèles local.
Relis le journal précédent pour ne pas répéter un correctif inefficace.

JOURNAL PRÉCÉDENT :
{historique_optimiseur}

CONFIGURATION :
- Seuils : {json.dumps(config.get("seuils", {}))}
- Règles apprises : {json.dumps(config.get("regles_apprises", []), ensure_ascii=False)}
- Modèles :
{catalogue_modeles}

ÉCHANGE RATÉ :
- Prompt : {echec.get('prompt')}
- Modèle choisi : {echec.get('modele')} (raison: {echec.get('raison')}, score: {echec.get('score')})
- Réponse IA : {echec.get('reponse')}
- Raison échec : {echec.get('raison_echec')}
- Réaction suivante : {suite}

RÈGLES STRICTES POUR TON JSON :
- `exemple_routage` doit être une phrase que l'UTILISATEUR pourrait taper (jamais une consigne pour l'IA).
- Si le bon modèle a été choisi mais qu'il a halluciné ou mal répondu, mets `bon_modele: null` et propose une `regle_systeme` courte.
- Utilise `regle_a_supprimer` uniquement si une règle existante dans `regles_apprises` est contre-productive.

Réponds UNIQUEMENT en JSON valide :
{{
  "diagnostic": "cause exacte en 1 phrase",
  "justification_correctif": "pourquoi ce correctif règle le problème",
  "bon_modele": "nom exact du modèle qui aurait dû répondre (ou null)",
  "exemple_routage": "requête utilisateur type pour entraîner le routeur (ou null)",
  "nouveau_signal_insatisfaction": "expression de rejet utilisateur (ou null)",
  "regle_systeme": "consigne courte et impérative pour le prompt système (ou null)",
  "regle_a_supprimer": "texte exact d'une règle apprise à retirer (ou null)",
  "ajustement_seuil": {{"param": "confiance ou marge ou insatisfaction", "delta": 0.0}}
}}"""

            try:
                res = ollama.chat(
                    model=juge,
                    messages=[{"role": "user", "content": prompt_juge}],
                    format="json",
                    options={"temperature": 0.1, "num_ctx": 4096},
                    keep_alive="1m",
                )
                verdict = json.loads(res["message"]["content"])
            except Exception as err:
                _log(f"  [Erreur Juge] : {err}", journal)
                continue

            _log(f"  Diagnostic    : {verdict.get('diagnostic', '')}", journal)
            _log(f"  Justification : {verdict.get('justification_correctif', '')}", journal)

            bon_modele = verdict.get("bon_modele")
            if bon_modele == "null":
                bon_modele = None
            exemple = (verdict.get("exemple_routage") or "").strip().lower()
            signal = (verdict.get("nouveau_signal_insatisfaction") or "").strip().lower()
            regle = (verdict.get("regle_systeme") or "").strip()
            regle_suppr = (verdict.get("regle_a_supprimer") or "").strip()
            ajustement = verdict.get("ajustement_seuil")

            aucune_action = True

            if bon_modele in config["routes"] and config["routes"][bon_modele]["role"] != "phatique" and len(exemple) >= 8:
                aucune_action = False
                valide, explication_vec = _exemple_route_valide(exemple, bon_modele, config)
                if valide:
                    liste_ex = config["routes"][bon_modele].setdefault("examples", [])
                    liste_ex.append(exemple)
                    config["routes"][bon_modele]["examples"] = liste_ex[-25:]
                    ajouts_routes += 1
                    config_modifiee = True
                    _log(f"  [+] Routage ({bon_modele}) : « {exemple} » -> {explication_vec}", journal)
                else:
                    _log(f"  [=] Routage ({bon_modele}) : « {exemple} » ignoré -> {explication_vec}", journal)

            if len(signal) >= 6 and signal != "null":
                aucune_action = False
                valide_sig, explication_sig = _signal_insatisfaction_nouveau(signal, config)
                if valide_sig:
                    liste_sig = config.setdefault("signaux", {}).setdefault("insatisfaction", [])
                    liste_sig.append(signal)
                    config["signaux"]["insatisfaction"] = liste_sig[-30:]
                    ajouts_signaux += 1
                    config_modifiee = True
                    _log(f"  [+] Signal critique : « {signal} » -> {explication_sig}", journal)
                else:
                    _log(f"  [=] Signal critique : « {signal} » ignoré -> {explication_sig}", journal)

            regles_actuelles = config.setdefault("regles_apprises", [])
            if regle_suppr and regle_suppr in regles_actuelles:
                aucune_action = False
                regles_actuelles.remove(regle_suppr)
                retraits_regles += 1
                config_modifiee = True
                _log(f"  [-] Règle système retirée : « {regle_suppr} »", journal)

            if len(regle) >= 10 and regle.lower() != "null" and not _contient_secret_potentiel(regle):
                aucune_action = False
                if regle not in regles_actuelles:
                    regles_actuelles.append(regle)
                    config["regles_apprises"] = regles_actuelles[-10:]
                    ajouts_regles += 1
                    config_modifiee = True
                    _log(f"  [+] Règle système ajoutée : « {regle} »", journal)
                else:
                    _log(f"  [=] Règle ignorée (déjà présente) : « {regle} »", journal)

            if isinstance(ajustement, dict):
                param = str(ajustement.get("param", "")).strip()
                try:
                    delta = float(ajustement.get("delta", 0.0))
                except (TypeError, ValueError):
                    delta = 0.0

                if param in BORNES_SEUILS and abs(delta) >= 0.005:
                    aucune_action = False
                    delta_borne = max(-0.02, min(0.02, delta))
                    ancien = float(config.setdefault("seuils", {}).get(param, 0.54))
                    minimum, maximum = BORNES_SEUILS[param]
                    nouveau = round(max(minimum, min(maximum, ancien + delta_borne)), 3)
                    if nouveau != ancien:
                        config["seuils"][param] = nouveau
                        ajustements_seuils += 1
                        config_modifiee = True
                        _log(f"  [+] Seuil '{param}' ajusté : {ancien} -> {nouveau}", journal)

            if aucune_action:
                _log("  [=] Aucune modification de config nécessaire.", journal)

            echanges[idx]["traite_optimiseur"] = True
    else:
        _log("[Phase 2 — Diagnostic] Aucun nouvel échec à corriger.", journal)

    # 3. Consolidation épisodique & mémoire utile
    nb_episodes, nb_faits = _consolider_memoire_long_terme(echanges, juge, journal)

    if config_modifiee:
        _sauvegarder_backup_config()
        _ecrire_atomique(CONFIG_FILE, json.dumps(config, indent=2, ensure_ascii=False))

    sauvegarder_tous_echanges(echanges)
    _liberer_vram(juge)

    resume = (
        f"Optimisation terminée : {len(echecs)} échec(s) traité(s) | "
        f"+{ajouts_routes} route(s), +{ajouts_signaux} signal(aux), +{ajouts_regles}/-{retraits_regles} règle(s), "
        f"{ajustements_seuils} seuil(s) | Mémoire : {nb_episodes} épisode(s), +{nb_faits} fait(s)."
    )
    _log(f"\n[Bilan] {resume}", journal)
    _sauvegarder_journal(journal)
    return resume


def _liberer_vram(modele: str) -> None:
    try:
        ollama.generate(model=modele, keep_alive=0)
    except Exception:
        pass


def _ecrire_atomique(chemin: Path, contenu: str) -> None:
    tmp = chemin.with_suffix(".tmp")
    tmp.write_text(contenu, encoding="utf-8")
    os.replace(tmp, chemin)