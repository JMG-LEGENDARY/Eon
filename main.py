import base64
import json
import platform
import re
import sys
import time
import uuid
from datetime import datetime
import ollama

import file_tools
import shell_tools
import terminal_panels as panels
from file_tools import (
    chercher_dans_fichiers,
    ecrire_fichier,
    lire_fichier,
    lister_fichiers,
    modifier_fichier,
)
from memory_manager import (
    BASE_DIR,
    charger_memoire_rapide,
    charger_memoire_utile,
    consulter_historique,
    effacer_historique_jour,
    effacer_tout_historique,
    formater_contexte_memoire,
    journaliser_echange,
    marquer_dernier_echange_echec,
    memoriser_fait_utilisateur,
    supprimer_fait_utilisateur,
)
from optimizer import executer_optimisation_nocturne
from perception_eon import PerceptionEon
from router import RouteurAutonome
from shell_tools import (
    arreter_tache_en_cours,
    executer_commande_shell,
    obtenir_tache_en_cours,
    statut_tache_en_cours,
)
from web_tools import lire_page_web, rechercher_sur_internet

MAX_ETAPES_AGENT = 12
MAX_TRANSFERTS_TOUR = 2
NB_SORTIES_OUTILS_COMPLETES = 2

REGEX_BALISE_WEB_FALLBACK = re.compile(r"\[DEMANDE_WEB\s*:\s*(.+?)\]", re.IGNORECASE)
REGEX_BALISE_TRANSFERT_FALLBACK = re.compile(r"\[TRANSFERT\s*:\s*([^\|\]]+)\|\s*(.+?)\]", re.IGNORECASE)
REGEX_PROPOSITION_WEB_TEXTE = re.compile(
    r"(chercher|rechercher|vérifier|aller|regarder|consulter).{0,35}(internet|en ligne|web|google)",
    re.IGNORECASE,
)
# Questions visuelles fines nécessitant réellement gemma3:12b-it-qat (couleur, texte, sens d'une scène)
REGEX_VISION_COMPLEXE = re.compile(
    r"\b(quelle\s+couleur|couleur\s+(de|du|le|des)|t-?shirt|pull|sweat|v[êe]tement|tenue|"
    r"lis\s+(ce|la|le|ceci|ça)|[ée]crit\s+sur|analyse\s+(l'image|la\s+sc[èe]ne|ce\s+que\s+tu\s+vois)|"
    r"d[ée]cris\s+(pr[ée]cis[ée]ment|ce\s+que\s+tu\s+vois|l'image))\b",
    re.IGNORECASE,
)
REGEX_CAPTURE_PHOTO = re.compile(
    r"\b(capture\s+(de\s+)?(la\s+)?cam[ée]ra|prends?\s+une?\s+photo|capture\s+photo|enregistre\s+l'image)\b",
    re.IGNORECASE,
)

_PERCEPTION_ACTIVE: PerceptionEon | None = None


def observer_camera(question: str = "Décris précisément tout ce que tu vois devant la caméra.") -> str:
    """Analyse sémantique approfondie de l'image caméra via gemma3:12b-it-qat.

    Args:
        question: Ce qu'il faut analyser ou vérifier sur l'image de la caméra.
    """
    if _PERCEPTION_ACTIVE is None:
        return "Capteurs non initialisés."
    return _PERCEPTION_ACTIVE.observer_camera(question)


def capturer_photo_camera(nom_fichier: str = "") -> str:
    """Prend une capture photo avec la webcam et l'enregistre sur le disque.

    Args:
        nom_fichier: Nom optionnel du fichier image (ex: 'test.jpg').
    """
    if _PERCEPTION_ACTIVE is None:
        return "Capteurs non initialisés."
    return _PERCEPTION_ACTIVE.capturer_photo(nom_fichier)


def inspecter_systeme() -> str:
    """Retourne les informations réelles sur l'OS, la machine, le dossier courant et l'équipe de modèles."""
    cfg = json.loads((BASE_DIR / "config_ia.json").read_text(encoding="utf-8"))
    lignes = [
        f"OS réel : {platform.system()} {platform.release()} ({platform.platform()})",
        f"Architecture : {platform.machine()} | Python : {platform.python_version()}",
        f"Répertoire projet : {BASE_DIR}",
        "Équipe de modèles disponibles :",
    ]
    for nom, info in cfg.get("routes", {}).items():
        caps = []
        if info.get("web_tools"):
            caps.append("Internet")
        if info.get("file_tools"):
            caps.append("Fichiers locaux")
        if info.get("shell_tools"):
            caps.append("Terminal bash")
        lignes.append(
            f"- @{info.get('alias', '?')} ({nom}) : {info.get('expertise', info.get('role'))} "
            f"[Outils: {', '.join(caps) or 'aucun'}]"
        )
    return "\n".join(lignes)


def transferer_tache(modele_cible: str, raison: str, consigne_transmise: str) -> str:
    """Délègue la tâche en cours à un autre modèle de l'équipe plus qualifié ou disposant des outils requis."""
    return f"TRANSFERT_DEMANDE|{modele_cible}|{raison}|{consigne_transmise}"


OUTILS_WEB_LISTE = [rechercher_sur_internet, lire_page_web]
OUTILS_FICHIERS_LISTE = [lister_fichiers, lire_fichier, chercher_dans_fichiers, modifier_fichier, ecrire_fichier]
OUTILS_SHELL_LISTE = [executer_commande_shell, statut_tache_en_cours, arreter_tache_en_cours]
OUTILS_BASE_LISTE = [
    consulter_historique,
    memoriser_fait_utilisateur,
    inspecter_systeme,
    transferer_tache,
    observer_camera,
    capturer_photo_camera,
]
NOMS_OUTILS_WEB = {fn.__name__ for fn in OUTILS_WEB_LISTE}
MAP_TOUS_OUTILS = {
    fn.__name__: fn for fn in [*OUTILS_WEB_LISTE, *OUTILS_FICHIERS_LISTE, *OUTILS_SHELL_LISTE, *OUTILS_BASE_LISTE]
}


class OrchestrateurAgent:
    def __init__(self) -> None:
        self.routeur = RouteurAutonome()
        self.modele_verrouille: str | None = None
        self.web_preautorise_tour = False
        self.perception: PerceptionEon | None = None
        file_tools.CALLBACK_CONFIRMATION = self.demander_autorisation_fichier
        shell_tools.CALLBACK_CONFIRMATION = self.demander_autorisation_shell
        self._synchroniser_modes_outils()

    def _synchroniser_modes_outils(self) -> None:
        file_tools.MODE_FICHIERS = self.routeur.config.get("file_mode", "ask")
        shell_tools.MODE_SHELL = self.routeur.config.get("shell_mode", "ask")

    def verifier_hot_reload(self) -> None:
        if self.routeur.verifier_hot_reload():
            self._synchroniser_modes_outils()

    def _demander_saisie_bloquee(self, invite: str) -> str:
        etait_actif = bool(self.perception and self.perception.ia_en_cours.is_set())
        if etait_actif and self.perception is not None:
            self.perception.ia_en_cours.clear()
            time.sleep(0.05)
        try:
            return panels.saisir(invite).strip()
        except (KeyboardInterrupt, EOFError):
            return ""
        finally:
            if etait_actif and self.perception is not None:
                self.perception.ia_en_cours.set()

    def demander_autorisation_web(self, modele: str, action: str, cible: str) -> bool:
        mode = self.routeur.config.get("web_mode", "ask")
        if mode == "off":
            panels.ecrire_chat("\n  [Web bloqué] Accès Internet désactivé (/web off).")
            return False
        if mode == "auto" or self.web_preautorise_tour:
            panels.log_outil(f"[Web auto] {action} : {cible}")
            return True

        panels.ecrire_chat(f"\n  [Autorisation Web] {modele} souhaite {action} : « {cible} »")
        rep = self._demander_saisie_bloquee("Autoriser l'accès à Internet ? [o/N] : ")
        autorise, detail = self.routeur.interpreter_consentement(rep)
        panels.ecrire_chat(f"  [Verdict : {'AUTORISÉ' if autorise else 'REFUSÉ'} | {detail}]")
        return autorise

    def demander_autorisation_fichier(self, action: str, detail_diff: str) -> bool:
        panels.ecrire_chat(f"\n  [Autorisation Fichier] {action}\n{detail_diff}\n")
        rep = self._demander_saisie_bloquee("Appliquer cette modification sur le disque ? [o/N] : ")
        autorise, detail = self.routeur.interpreter_consentement(rep)
        panels.ecrire_chat(f"  [Verdict Fichier : {'AUTORISÉ' if autorise else 'REFUSÉ'} | {detail}]")
        return autorise

    def demander_autorisation_shell(self, commande_affichee: str) -> bool:
        panels.ecrire_chat(f"\n  [Autorisation Terminal] {commande_affichee}")
        rep = self._demander_saisie_bloquee("Exécuter cette commande ? [o/N] : ")
        autorise, detail = self.routeur.interpreter_consentement(rep)
        panels.ecrire_chat(f"  [Verdict Terminal : {'AUTORISÉ' if autorise else 'REFUSÉ'} | {detail}]")
        return autorise

    def _annuaire_equipe(self, modele_courant: str) -> str:
        lignes = []
        for nom, cfg in self.routeur.config.get("routes", {}).items():
            if nom == modele_courant or cfg.get("role") in ("phatique", "vision_multimodale"):
                continue
            caps = []
            if cfg.get("shell_tools"):
                caps.append("Terminal bash")
            if cfg.get("file_tools"):
                caps.append("Fichiers")
            if cfg.get("web_tools"):
                caps.append("Web")
            lignes.append(
                f"- @{cfg.get('alias')} ({nom}) : {cfg.get('expertise', cfg.get('role'))} "
                f"[Capacités : {', '.join(caps) or 'dialogue'}]"
            )
        return "\n".join(lignes)

    def _liste_outils_route(self, route_cfg: dict) -> list:
        if route_cfg.get("role") in ("phatique", "vision_multimodale"):
            return []
        outils = list(OUTILS_BASE_LISTE)
        if route_cfg.get("web_tools", False) and self.routeur.config.get("web_mode", "ask") != "off":
            outils.extend(OUTILS_WEB_LISTE)
        if route_cfg.get("file_tools", False):
            outils.extend(OUTILS_FICHIERS_LISTE)
        if route_cfg.get("shell_tools", False) and self.routeur.config.get("shell_mode", "ask") != "off":
            outils.extend(OUTILS_SHELL_LISTE)
        return outils

    def construire_system_prompt(self, modele: str, route_cfg: dict, mode_fallback_texte: bool = False) -> dict[str, str]:
        role = route_cfg.get("role", "")
        if role == "phatique":
            return {
                "role": "system",
                "content": (
                    "Tu es Eon. Tutoie l'utilisateur. Réponds de façon très courte, détendue et naturelle en français (1 phrase). "
                    "Ne dis jamais « Je suis là pour vous assister » ni « Quel est votre besoin ? »."
                ),
            }

        date_actuelle = datetime.now().strftime("%A %d %B %Y, %H:%M")
        a_fichiers = route_cfg.get("file_tools", False)
        a_shell = route_cfg.get("shell_tools", False) and self.routeur.config.get("shell_mode", "ask") != "off"
        a_web = route_cfg.get("web_tools", False) and self.routeur.config.get("web_mode", "ask") != "off"

        ton = (
            "Tu es Eon, un pote intelligent, détendu et naturel. Tutoie toujours l'utilisateur. Parle comme un vrai humain, jamais comme un centre d'appel."
            if role == "creatif_discussion"
            else "Tu es Eon, un assistant technique franc, naturel et autonome. Tutoie toujours l'utilisateur. Va droit au but sans blabla final."
        )

        regles = [
            f"Nom : Eon | Moteur actif : {modele} (@{route_cfg.get('alias')}) | Date : {date_actuelle} | Langue : français.",
            ton,
            "Règle d'honnêteté & Vision : N'invente JAMAIS un fait, une couleur, un vêtement, un objet ou le résultat d'une commande. "
            "Tu disposes déjà des détections temps réel (personnes identifiées, objets visibles) dans `ÉTAT DES CAPTEURS`. "
            "Si l'utilisateur demande une analyse visuelle fine (couleur, texte, sens d'une image), appelle `observer_camera`.",
        ]

        if a_fichiers or a_shell:
            regles.append(
                f"OUTILS LOCAUX ACTIFS (dossier : {BASE_DIR}, OS : Fedora Linux) : "
                "Tu as un accès réel aux outils locaux (`executer_commande_shell`, `statut_tache_en_cours`, `arreter_tache_en_cours`, `lister_fichiers`, `lire_fichier`, `modifier_fichier`, `ecrire_fichier`). "
                "Appelle-les VRAIMENT via `tool_calls`. INTERDIT ABSOLU de simuler une commande (`[simulé]`) ou de prétendre ne pas avoir accès au terminal. "
                "Pour une mise à jour Fedora, appelle `executer_commande_shell(commande='sudo dnf upgrade -y')` sans demander le mot de passe dans le chat."
            )
        else:
            regles.append(
                "LIMITES DE TES OUTILS : Tu n'as PAS d'accès au terminal bash ni aux fichiers locaux. "
                "Ne simule JAMAIS une commande et ne prétends jamais pouvoir l'exécuter toi-même."
            )

        if a_web:
            txt_web = "Tu disposes de `rechercher_sur_internet` et `lire_page_web`."
            if mode_fallback_texte:
                txt_web += " Pour chercher en ligne, réponds UNIQUEMENT : [DEMANDE_WEB: requête]."
            regles.append(txt_web)

        annuaire = self._annuaire_equipe(modele)
        if annuaire:
            instruction_transfert = (
                "AUTONOMIE & DÉLÉGATION : Si la demande sort de ton domaine d'expertise ou exige un outil que tu n'as pas "
                "(ex: exécuter une commande shell, éditer un fichier, architecture lourde, discussion créative), "
                "explique brièvement pourquoi tu ne peux pas le faire seul et appelle IMMÉDIATEMENT `transferer_tache` vers le spécialiste adapté :\n"
                f"{annuaire}"
            )
            if mode_fallback_texte:
                instruction_transfert += "\nSi tu n'as pas d'appels de fonctions, réponds UNIQUEMENT : [TRANSFERT: alias_cible | raison et consigne]."
            regles.append(instruction_transfert)

        regles_apprises = self.routeur.config.get("regles_apprises", [])
        if regles_apprises:
            regles.append("Consignes apprises :\n" + "\n".join(f"- {r}" for r in regles_apprises))

        if self.perception is not None:
            regles.append(f"ÉTAT DES CAPTEURS : {self.perception.decrire_etat_capteurs()}")

        bloc_memoire = formater_contexte_memoire()
        if bloc_memoire:
            regles.append(f"\n{bloc_memoire}")

        return {"role": "system", "content": "\n".join(regles)}

    @staticmethod
    def _compresser_historique_outils(messages: list[dict], garder_recents: int = NB_SORTIES_OUTILS_COMPLETES) -> None:
        indices_outils = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
        if len(indices_outils) <= garder_recents:
            return
        for idx in indices_outils[:-garder_recents]:
            contenu = str(messages[idx].get("content", ""))
            if len(contenu) > 350:
                messages[idx]["content"] = (
                    f"{contenu[:220].rstrip()}\n... [sortie outil condensée ({len(contenu)} car.)] ...\n{contenu[-100:].lstrip()}"
                )

    def _appel_stream_ollama(
        self,
        modele: str,
        messages: list,
        outils: list | None,
        options: dict,
        interruption_event=None,
        keep_alive: str | int = "4m",
    ):
        kwargs = {
            "model": modele,
            "messages": messages,
            "tools": outils,
            "stream": True,
            "options": options,
            "keep_alive": keep_alive,
        }
        if "qwen3" in modele.lower():
            try:
                stream = ollama.chat(**kwargs, think=False)
            except TypeError:
                stream = ollama.chat(**kwargs)
        else:
            stream = ollama.chat(**kwargs)

        for chunk in stream:
            if interruption_event is not None and interruption_event.is_set():
                try:
                    stream.close()
                except Exception:
                    pass
                break
            yield chunk

    def _generer_bilan_arret(
        self,
        modele: str,
        messages: list[dict],
        options: dict,
        motif: str,
        interruption_event=None,
    ) -> str:
        panels.ecrire_chat(f"\n  [Agent -> {motif}. Génération du bilan...]\n")
        self._compresser_historique_outils(messages)
        messages.append({
            "role": "user",
            "content": (
                f"[ARRÊT AUTOMATIQUE : {motif}]\n"
                "Tu n'as plus d'outils pour ce tour. Résume brièvement : 1) ce qui a été fait, 2) ce qui bloque, 3) la suite conseillée."
            ),
        })
        texte = ""
        for chunk in self._appel_stream_ollama(modele, messages, None, options, interruption_event=interruption_event):
            morceau = chunk.get("message", {}).get("content", "")
            if morceau:
                panels.ecrire_chat(morceau, fin="")
                texte += morceau
        panels.flush_chat()
        panels.ecrire_chat("")
        return texte

    def executer_tour_autonome(
        self,
        message: str,
        modele_initial: str,
        session_id: str,
        interruption_event=None,
        images_tour: list[bytes] | None = None,
    ) -> tuple[str, str, list[str], str | None]:
        modele_courant = modele_initial
        prompt_courant = message
        outils_utilises_tour: list[str] = []
        modele_delegue_final: str | None = None
        historique = charger_memoire_rapide(session_id=session_id, limite_echanges=6)

        for nb_transferts in range(MAX_TRANSFERTS_TOUR + 1):
            if interruption_event is not None and interruption_event.is_set():
                return "", modele_courant, outils_utilises_tour, modele_delegue_final

            route_cfg = self.routeur.config["routes"].get(modele_courant, {})
            outils_liste = self._liste_outils_route(route_cfg)
            a_shell = route_cfg.get("shell_tools", False) and self.routeur.config.get("shell_mode", "ask") != "off"
            supporte_tools_natifs = True
            est_modele_vlm = route_cfg.get("role") == "vision_multimodale"

            options = {
                "num_ctx": route_cfg.get("num_ctx", 4096),
                "temperature": route_cfg.get("temperature", 0.25),
            }
            if route_cfg.get("num_predict") is not None:
                options["num_predict"] = route_cfg["num_predict"]

            msg_utilisateur: dict = {"role": "user", "content": prompt_courant}
            if images_tour and nb_transferts == 0:
                msg_utilisateur["images"] = [base64.b64encode(img).decode("ascii") for img in images_tour]

            messages = [
                self.construire_system_prompt(modele_courant, route_cfg, False),
                *historique,
                msg_utilisateur,
            ]

            appels_compteur: dict[str, int] = {}
            derniere_signature_appel: str | None = None
            raison_interruption: str | None = None
            rattrapage_simulation_fait = False
            demande_transfert: tuple[str, str, str] | None = None

            for _ in range(MAX_ETAPES_AGENT):
                if interruption_event is not None and interruption_event.is_set():
                    return "", modele_courant, outils_utilises_tour, modele_delegue_final

                self._compresser_historique_outils(messages)
                texte_tour = ""
                tool_calls_recus = []
                mode_fallback_texte = False

                for tentative in (1, 2):
                    if tentative == 1 and supporte_tools_natifs and outils_liste:
                        outils_actifs = outils_liste
                        mode_fallback_texte = False
                        messages[0] = self.construire_system_prompt(modele_courant, route_cfg, False)
                    else:
                        outils_actifs = None
                        mode_fallback_texte = True
                        messages[0] = self.construire_system_prompt(modele_courant, route_cfg, True)

                    try:
                        stream = self._appel_stream_ollama(
                            modele_courant,
                            messages,
                            outils_actifs,
                            options,
                            interruption_event=interruption_event,
                            keep_alive=0 if est_modele_vlm else "4m",
                        )
                        texte_tour = ""
                        tool_calls_recus = []
                        for chunk in stream:
                            msg_chunk = chunk.get("message", {})
                            if msg_chunk.get("content"):
                                morceau = msg_chunk["content"]
                                if not mode_fallback_texte:
                                    panels.ecrire_chat(morceau, fin="")
                                texte_tour += morceau
                            if msg_chunk.get("tool_calls"):
                                tool_calls_recus.extend(msg_chunk["tool_calls"])
                        break
                    except ollama.ResponseError as err:
                        if tentative == 1 and "does not support tools" in str(err):
                            supporte_tools_natifs = False
                            continue
                        raise

                if interruption_event is not None and interruption_event.is_set():
                    panels.flush_chat()
                    return texte_tour, modele_courant, outils_utilises_tour, modele_delegue_final

                if mode_fallback_texte:
                    match_transfert = REGEX_BALISE_TRANSFERT_FALLBACK.search(texte_tour)
                    if match_transfert:
                        cible_raw = match_transfert.group(1).strip().lstrip("@")
                        consigne = match_transfert.group(2).strip()
                        demande_transfert = (cible_raw, "Délégation demandée par le modèle", consigne)
                        break

                    match_web = REGEX_BALISE_WEB_FALLBACK.search(texte_tour)
                    if match_web:
                        requete_web = match_web.group(1).strip()
                        if self.demander_autorisation_web(modele_courant, "rechercher sur Internet", requete_web):
                            res = rechercher_sur_internet(requete_web)
                            panels.log_outil(f"Résultat web :\n{res[:1200]}")
                            outils_utilises_tour.append("rechercher_sur_internet")
                            messages.append({"role": "assistant", "content": texte_tour})
                            messages.append({
                                "role": "user",
                                "content": f"[RÉSULTATS INTERNET POUR '{requete_web}'] :\n{res}\n\nRéponds brièvement en te basant sur ces résultats.",
                            })
                        else:
                            messages.append({"role": "assistant", "content": texte_tour})
                            messages.append({"role": "user", "content": "[ACCÈS INTERNET REFUSÉ] Dis-le brièvement."})
                        continue

                    panels.ecrire_chat(texte_tour)
                    return texte_tour, modele_courant, outils_utilises_tour, modele_delegue_final

                if not tool_calls_recus:
                    texte_min = texte_tour.lower()
                    tente_simulation = a_shell and not rattrapage_simulation_fait and (
                        "[simulé]" in texte_min
                        or "simuler l'exécution" in texte_min
                        or "je ne peux pas exécuter directement" in texte_min
                    )
                    if tente_simulation:
                        rattrapage_simulation_fait = True
                        panels.ecrire_chat("\n  [Rattrapage Agent -> simulation bloquée, exécution réelle forcée...]\n")
                        messages.append({"role": "assistant", "content": texte_tour})
                        messages.append({
                            "role": "user",
                            "content": (
                                "[ERREUR SYSTÈME : Tu as l'outil `executer_commande_shell` actif. "
                                "N'écris JAMAIS `[simulé]`. Appelle MAINTENANT `executer_commande_shell` avec la commande.]"
                            ),
                        })
                        continue

                    panels.flush_chat()
                    panels.ecrire_chat("")
                    return texte_tour, modele_courant, outils_utilises_tour, modele_delegue_final

                panels.flush_chat()
                messages.append({"role": "assistant", "content": texte_tour, "tool_calls": tool_calls_recus})
                for appel in tool_calls_recus:
                    if interruption_event is not None and interruption_event.is_set():
                        return texte_tour, modele_courant, outils_utilises_tour, modele_delegue_final

                    nom_fn = appel["function"]["name"]
                    args = appel["function"].get("arguments", {})

                    if nom_fn == "transferer_tache":
                        cible_raw = str(args.get("modele_cible", "")).strip().lstrip("@")
                        raison_tr = str(args.get("raison", "Expertise mieux adaptée")).strip()
                        consigne_tr = str(args.get("consigne_transmise", message)).strip()
                        demande_transfert = (cible_raw, raison_tr, consigne_tr)
                        break

                    signature_appel = f"{nom_fn}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
                    appels_compteur[signature_appel] = appels_compteur.get(signature_appel, 0) + 1
                    if signature_appel == derniere_signature_appel or appels_compteur[signature_appel] >= 3:
                        raison_interruption = f"Boucle infinie stoppée sur `{nom_fn}`"
                        break
                    derniere_signature_appel = signature_appel

                    if nom_fn in NOMS_OUTILS_WEB:
                        cible = args.get("requete") or args.get("url") or str(args)
                        action = "rechercher sur Internet" if nom_fn == "rechercher_sur_internet" else "lire la page web"
                        if not self.demander_autorisation_web(modele_courant, action, cible):
                            messages.append({"role": "tool", "content": "REFUS UTILISATEUR : Accès Internet refusé."})
                            continue

                    panels.log_outil(f"Appel outil : {nom_fn}({args})")
                    fn = MAP_TOUS_OUTILS.get(nom_fn)
                    if fn:
                        outils_utilises_tour.append(nom_fn)
                        try:
                            resultat_outil = str(fn(**args))
                        except TypeError as err_args:
                            resultat_outil = f"Erreur d'arguments pour `{nom_fn}` : {err_args}. Corrige les paramètres et réessaie."
                    else:
                        resultat_outil = f"Outil inconnu : {nom_fn}"

                    panels.log_outil(f"Retour {nom_fn} :\n{resultat_outil[:1500]}")

                    if appels_compteur[signature_appel] == 2:
                        resultat_outil += "\n[AVERTISSEMENT : Appel déjà effectué. Change d'approche ou délègue via `transferer_tache`.]"

                    messages.append({"role": "tool", "content": resultat_outil})

                if demande_transfert or raison_interruption:
                    break

            if demande_transfert:
                cible_raw, raison_tr, consigne_tr = demande_transfert
                nouveau_modele = self.routeur.resoudre_modele(cible_raw)
                role_phatique = self.routeur.modele_par_role("phatique")

                if nouveau_modele and nouveau_modele != modele_courant and nouveau_modele != role_phatique and nb_transferts < MAX_TRANSFERTS_TOUR:
                    panels.ecrire_chat(f"\n  [Transfert autonome] {modele_courant} -> {nouveau_modele} ({raison_tr})\n")
                    modele_delegue_final = nouveau_modele
                    prompt_courant = (
                        f"[TÂCHE DÉLÉGUÉE PAR {modele_courant}]\n"
                        f"Pourquoi il t'a passé la main : {raison_tr}\n"
                        f"Demande initiale de l'utilisateur : {message}\n"
                        f"Action précise à accomplir : {consigne_tr}"
                    )
                    modele_courant = nouveau_modele
                    continue

            motif_arret = raison_interruption or f"Plafond de {MAX_ETAPES_AGENT} étapes atteint"
            bilan = self._generer_bilan_arret(
                modele_courant,
                messages,
                options,
                motif_arret,
                interruption_event=interruption_event,
            )
            return bilan, modele_courant, outils_utilises_tour, modele_delegue_final

        return "Transferts maximum atteints.", modele_courant, outils_utilises_tour, modele_delegue_final


def afficher_aide() -> None:
    panels.ecrire_chat("""
Commandes disponibles :
  @<alias> <prompt>              Force un modèle précis (ex: @code lis main.py, @12b analyse...)
  /model <alias | auto>          Verrouille un modèle pour la session, ou repasse en 'auto'
  /models                        Affiche l'équipe de modèles, leurs alias et leurs permissions
  /set <alias> <param> <val>     Modifie temperature, num_ctx, num_predict, web_tools, file_tools, shell_tools, defaut, optimiseur
  /add-example <alias> <phrase>  Ajoute une phrase d'entraînement au routeur et recharge le cache
  /web <ask | auto | off>        Accès Internet ('ask' = permission, 'auto' = libre, 'off' = coupé)
  /files <ask | auto | ro>       Accès Fichiers ('ask' = écriture sur confirmation, 'auto' = libre, 'ro' = lecture seule)
  /shell <ask | auto | off>      Accès Terminal ('ask' = permission, 'auto' = libre autonome, 'off' = coupé)
  /enroll [nom]                  Enregistre le visage actuel (ex: /enroll Bryan, /enroll Pere)
  /enroll reset [nom]            Réinitialise un profil ou toute la biométrie
  /security <auto | absent | off> Mode sécurité caméra ('absent' = alerte immédiate si inconnu)
  /history [n]                   Affiche les n derniers échanges enregistrés
  /clear [session | all]         Vide l'historique du jour, de la session ou total
  /memory                        Affiche la mémoire utile permanente
  /forget <clé>                  Supprime une entrée de la mémoire utile
  /optimize                      Lance l'optimiseur (audit + entraînement routeur + consolidation mémoire)
  /help                          Affiche cette aide
  [Touche Échap ou voix]         Interrompt Eon instantanément pendant qu'il répond
""")


def traiter_commande(cmd: str, app: OrchestrateurAgent, session_id: str) -> tuple[bool, str]:
    parties = cmd.split()
    action = parties[0].lower()
    cfg = app.routeur.config

    if action == "/help":
        afficher_aide()
        return True, session_id

    if action == "/enroll":
        if app.perception is None:
            panels.ecrire_chat("[Erreur] Module de perception inactif.\n")
            return True, session_id
        if len(parties) >= 2 and parties[1].lower() == "reset":
            cible = parties[2] if len(parties) >= 3 else None
            panels.ecrire_chat(f"[Eon Biométrie] {app.perception.reinitialiser_profil_biometrique(cible)}\n")
        else:
            nom = parties[1] if len(parties) >= 2 else "Bryan"
            panels.ecrire_chat(f"[Eon Biométrie] {app.perception.enregistrer_visage_autorise(nom)}\n")
        return True, session_id

    if action == "/security":
        if app.perception is None:
            return True, session_id
        if len(parties) < 2 or parties[1].lower() not in ("auto", "absent", "off"):
            panels.ecrire_chat(f"Mode sécurité actuel : {app.perception.mode_securite} | Usage : /security <auto|absent|off>\n")
            return True, session_id
        app.perception.mode_securite = parties[1].lower()
        cfg.setdefault("eon", {})["mode_securite"] = app.perception.mode_securite
        app.routeur.sauvegarder_config()
        panels.ecrire_chat(f"[Eon Sécurité] mode_securite = {app.perception.mode_securite}\n")
        return True, session_id

    if action == "/history":
        nb = int(parties[1]) if len(parties) > 1 and parties[1].isdigit() else 10
        panels.ecrire_chat("\n" + consulter_historique(limite=nb) + "\n")
        return True, session_id

    if action == "/models":
        emb = cfg.get("routage", {}).get("utiliser_embeddings", True)
        panels.ecrire_chat(
            f"\nWeb : {cfg.get('web_mode', 'ask')} | Fichiers : {cfg.get('file_mode', 'ask')} | "
            f"Terminal : {cfg.get('shell_mode', 'ask')} | Embeddings : {emb} | Optimiseur : {cfg.get('modele_optimiseur')}"
        )
        panels.ecrire_chat(f"Modèle verrouillé : {app.modele_verrouille or 'Aucun (auto + délégation)'}\n")
        for nom, r in cfg["routes"].items():
            panels.ecrire_chat(
                f"  - [{r.get('alias', '?'):5}] {nom:32} | rôle={r['role']:18} | "
                f"web={r.get('web_tools', False)} | files={r.get('file_tools', False)} | shell={r.get('shell_tools', False)}"
            )
        panels.ecrire_chat("")
        return True, session_id

    if action == "/model":
        if len(parties) < 2:
            panels.ecrire_chat("Usage : /model <alias_ou_nom> ou /model auto\n")
            return True, session_id
        if parties[1].lower() == "auto":
            app.modele_verrouille = None
            panels.ecrire_chat("[Système] Routage automatique réactivé.\n")
        else:
            modele = app.routeur.resoudre_modele(parties[1])
            if not modele:
                panels.ecrire_chat(f"[Erreur] Modèle inconnu : {parties[1]}\n")
            else:
                app.modele_verrouille = modele
                panels.ecrire_chat(f"[Système] Session verrouillée sur : {modele}\n")
        return True, session_id

    if action == "/set":
        if len(parties) < 4:
            panels.ecrire_chat("Usage : /set <alias> <temperature|num_ctx|num_predict|web_tools|file_tools|shell_tools|defaut|optimiseur> <valeur>\n")
            return True, session_id
        modele = app.routeur.resoudre_modele(parties[1])
        if not modele:
            panels.ecrire_chat(f"[Erreur] Modèle inconnu : {parties[1]}\n")
            return True, session_id
        param = parties[2].lower()
        val_str = parties[3]

        try:
            if param == "temperature":
                cfg["routes"][modele]["temperature"] = float(val_str)
            elif param == "num_ctx":
                cfg["routes"][modele]["num_ctx"] = int(val_str)
            elif param == "num_predict":
                cfg["routes"][modele]["num_predict"] = None if val_str.lower() == "none" else int(val_str)
            elif param in ("web_tools", "file_tools", "shell_tools"):
                cfg["routes"][modele][param] = val_str.lower() in ("true", "1", "oui", "on")
            elif param == "defaut":
                cfg["route_defaut"] = modele
            elif param == "optimiseur":
                cfg["modele_optimiseur"] = modele
            else:
                panels.ecrire_chat(f"[Erreur] Paramètre inconnu : {param}\n")
                return True, session_id

            app.routeur.sauvegarder_config()
            panels.ecrire_chat(f"[Système] {modele} -> {param} = {val_str} (sauvegardé)\n")
        except ValueError:
            panels.ecrire_chat(f"[Erreur] Valeur invalide pour {param} : {val_str}\n")
        return True, session_id

    if action == "/add-example":
        if len(parties) < 3:
            panels.ecrire_chat("Usage : /add-example <alias> <phrase d'exemple>\n")
            return True, session_id
        modele = app.routeur.resoudre_modele(parties[1])
        if not modele:
            panels.ecrire_chat(f"[Erreur] Modèle inconnu : {parties[1]}\n")
            return True, session_id
        phrase = " ".join(parties[2:]).strip().lower()
        cfg["routes"][modele].setdefault("examples", []).append(phrase)
        app.routeur.sauvegarder_config()
        app.routeur.charger()
        panels.ecrire_chat(f"[Système] Exemple ajouté à {modele} et vecteurs mis à jour.\n")
        return True, session_id

    if action in ("/web", "/files", "/shell"):
        cles = {"/web": ("web_mode", ("ask", "auto", "off")), "/files": ("file_mode", ("ask", "auto", "ro")), "/shell": ("shell_mode", ("ask", "auto", "off"))}
        cle_cfg, valides = cles[action]
        if len(parties) < 2 or parties[1].lower() not in valides:
            panels.ecrire_chat(f"Usage : {action} <{'|'.join(valides)}>\n")
            return True, session_id
        cfg[cle_cfg] = parties[1].lower()
        app._synchroniser_modes_outils()
        app.routeur.sauvegarder_config()
        panels.ecrire_chat(f"[Système] {cle_cfg} = {cfg[cle_cfg]}\n")
        return True, session_id

    if action == "/clear":
        sous_cmd = parties[1].lower() if len(parties) > 1 else "jour"
        nouvelle_session = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        if sous_cmd == "session":
            panels.ecrire_chat("[Système] Contexte de la session actuelle réinitialisé.\n")
            return True, nouvelle_session
        if sous_cmd == "all":
            confirm = panels.saisir("Confirmer la suppression de TOUT l'historique ? [o/N] : ").strip().lower()
            if confirm in ("o", "oui"):
                effacer_tout_historique()
                panels.ecrire_chat("[Système] Historique entièrement effacé.\n")
            return True, nouvelle_session
        nb = effacer_historique_jour()
        panels.ecrire_chat(f"[Système] {nb} échange(s) d'aujourd'hui supprimé(s) et session réinitialisée.\n")
        return True, nouvelle_session

    if action == "/memory":
        panels.ecrire_chat("\n" + json.dumps(charger_memoire_utile(), indent=2, ensure_ascii=False) + "\n")
        return True, session_id

    if action == "/forget":
        if len(parties) < 2:
            panels.ecrire_chat("Usage : /forget <cle>\n")
            return True, session_id
        if supprimer_fait_utilisateur(parties[1]):
            panels.ecrire_chat(f"[Système] Clé '{parties[1]}' supprimée.\n")
        else:
            panels.ecrire_chat(f"[Système] Clé '{parties[1]}' introuvable.\n")
        return True, session_id

    if action == "/optimize":
        bilan = executer_optimisation_nocturne()
        panels.ecrire_chat(f"\n[Système] {bilan}\n")
        app.verifier_hot_reload()
        return True, session_id

    panels.ecrire_chat(f"[Système] Commande inconnue : {action} (tape /help)\n")
    return True, session_id


def _extraire_image_camera_bytes(perception: PerceptionEon) -> list[bytes] | None:
    """Capture l'image brute de la webcam pour un envoi direct en 1 passe au VLM."""
    try:
        import cv2
    except ImportError:
        return None
    if not perception.camera_active:
        return None
    with perception._lock_vision:
        frame = perception._derniere_frame.copy() if perception._derniere_frame is not None else None
    if frame is None:
        return None
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return [bytes(buf)] if ok else None


def boucle_principale() -> None:
    global _PERCEPTION_ACTIVE

    app = OrchestrateurAgent()
    session_courante = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    dernier_modele: str | None = None
    dernier_prompt_valide: str | None = None
    derniere_reponse_valide: str | None = None
    attente_confirmation_web_texte = False
    attente_arret_tache_bloquee = False
    modele_phatique = app.routeur.modele_par_role("phatique")

    def _on_alerte(msg_alerte: str) -> None:
        panels.ecrire_chat(f"\n🚨 {msg_alerte}\n")

    perception = PerceptionEon(app.routeur.config.get("eon", {}), on_alerte_securite=_on_alerte)
    app.perception = perception
    _PERCEPTION_ACTIVE = perception

    panels.enregistrer_callbacks_ui(
        on_prompt=perception.soumettre_texte_clavier,
        on_escape=perception.declencher_interruption,
    )

    panels.ecrire_chat("Eon est prêt (3 panneaux actifs : Conversation / Outils / Perception). Tape /help.\n")
    perception.demarrer()

    try:
        while True:
            entrant = perception.attendre_message()
            if entrant is None:
                panels.ecrire_chat("\nFin de session.")
                break

            try:
                msg = entrant.texte.strip()
                app.verifier_hot_reload()

                if not msg:
                    continue

                if msg.startswith("/"):
                    _, session_courante = traiter_commande(msg, app, session_courante)
                    attente_confirmation_web_texte = False
                    continue

                tache_active = obtenir_tache_en_cours()
                if tache_active is not None:
                    if attente_arret_tache_bloquee:
                        est_oui, _ = app.routeur.interpreter_consentement(msg)
                        attente_arret_tache_bloquee = False
                        if est_oui:
                            panels.ecrire_chat(f"-> Eon : {arreter_tache_en_cours()}\n")
                        else:
                            panels.ecrire_chat(f"-> Eon : D'accord, je laisse « {tache_active.commande} » continuer en fond.\n")
                        continue

                    if tache_active.est_bloquee(seuil_sec=20.0):
                        attente_arret_tache_bloquee = True
                        panels.ecrire_chat(
                            f"\n-> Eon : La tâche « {tache_active.commande} » tourne en arrière-plan sans sortie depuis "
                            f"{tache_active.secondes_sans_activite():.0f}s. Veux-tu l'arrêter ?\n"
                        )
                        continue

                if REGEX_CAPTURE_PHOTO.search(msg):
                    res_photo = perception.capturer_photo()
                    panels.ecrire_chat(f"-> Eon [Caméra] : {res_photo}\n")
                    continue

                modele_force: str | None = None
                prompt_effectif = msg
                images_tour: list[bytes] | None = None

                if attente_confirmation_web_texte and dernier_modele and not msg.startswith("@"):
                    est_oui, detail = app.routeur.interpreter_consentement(msg)
                    panels.ecrire_chat(f"  [Réponse Web : {'OUI' if est_oui else 'NON'} | {detail}]")
                    modele_force = dernier_modele
                    if est_oui:
                        app.web_preautorise_tour = True
                        prompt_effectif = f"{msg}\n[AUTORISATION ACCORDÉE : Appelle `rechercher_sur_internet` maintenant.]"
                    else:
                        prompt_effectif = f"{msg}\n[AUTORISATION REFUSÉE : N'utilise pas Internet.]"
                    attente_confirmation_web_texte = False
                elif msg.startswith("@"):
                    morceaux = msg[1:].split(maxsplit=1)
                    if len(morceaux) < 2:
                        panels.ecrire_chat("[Erreur] Précise un prompt après @<modèle> (ex: @code lis main.py).\n")
                        continue
                    modele_trouve = app.routeur.resoudre_modele(morceaux[0])
                    if not modele_trouve:
                        panels.ecrire_chat(f"[Erreur] Modèle inconnu : @{morceaux[0]} (tape /models).\n")
                        continue
                    modele_force = modele_trouve
                    prompt_effectif = morceaux[1].strip()
                    if app.routeur.config["routes"].get(modele_force, {}).get("role") == "vision_multimodale":
                        images_tour = _extraire_image_camera_bytes(perception)

                # Routage direct en 1 seule passe vers gemma3:12b-it-qat uniquement pour l'analyse visuelle fine
                if modele_force is None and REGEX_VISION_COMPLEXE.search(prompt_effectif):
                    modele_vlm = app.routeur.modele_par_role("vision_multimodale")
                    imgs = _extraire_image_camera_bytes(perception)
                    if imgs:
                        modele_force = modele_vlm
                        images_tour = imgs
                        panels.log_perception("Routage direct VLM (1 passe) vers gemma3:12b-it-qat.")

                if (
                    dernier_modele
                    and dernier_modele != modele_phatique
                    and dernier_prompt_valide
                    and derniere_reponse_valide is not None
                ):
                    if modele_force and modele_force != dernier_modele:
                        marquer_dernier_echange_echec("Forçage manuel @alias au tour suivant", modele_corrige=modele_force)
                    else:
                        est_echec, detail_echec = app.routeur.evaluer_insatisfaction(
                            prompt_effectif,
                            dernier_prompt_valide,
                            derniere_reponse_valide,
                        )
                        if est_echec:
                            marquer_dernier_echange_echec(f"{detail_echec} -> « {msg} »")

                t0 = time.perf_counter()
                modele_initial, sim, raison = app.routeur.choisir_modele(
                    prompt_effectif,
                    dernier_modele=dernier_modele,
                    modele_force=modele_force,
                    modele_verrouille=app.modele_verrouille,
                )
                dt_route = (time.perf_counter() - t0) * 1000
                panels.ecrire_chat(f"-> {modele_initial} [{raison} | conf: {sim:.2f} | routage: {dt_route:.1f} ms]\n")

                perception.interruption_event.clear()
                perception.ia_en_cours.set()
                try:
                    rep, modele_final, outils_utilises, modele_delegue = app.executer_tour_autonome(
                        prompt_effectif,
                        modele_initial,
                        session_id=session_courante,
                        interruption_event=perception.interruption_event,
                        images_tour=images_tour,
                    )
                except Exception as e:
                    panels.ecrire_chat(f"\n[Erreur Ollama] : {e}\n")
                    app.web_preautorise_tour = False
                    continue
                finally:
                    perception.ia_en_cours.clear()

                if perception.interruption_event.is_set():
                    panels.ecrire_chat("\n[Eon interrompu — à l'écoute]\n")

                app.web_preautorise_tour = False
                dernier_modele = modele_final
                dernier_prompt_valide = msg
                derniere_reponse_valide = rep
                attente_confirmation_web_texte = bool("?" in rep and REGEX_PROPOSITION_WEB_TEXTE.search(rep))

                dt_total = (time.perf_counter() - t0) * 1000
                panels.ecrire_chat(f"\n(-> total {dt_total:.1f} ms)\n")

                entree_log = {
                    "session_id": session_courante,
                    "date": datetime.now().isoformat(timespec="seconds"),
                    "modele": modele_final,
                    "modele_initial": modele_initial,
                    "raison": raison,
                    "score": round(sim, 4),
                    "outils_utilises": outils_utilises,
                    "prompt": msg,
                    "reponse": rep,
                }
                if modele_delegue:
                    entree_log["echec_detecte"] = True
                    entree_log["raison_echec"] = f"Auto-transfert de {modele_initial} vers {modele_delegue}"
                    entree_log["modele_attendu"] = modele_delegue

                journaliser_echange(entree_log)
            finally:
                perception.autoriser_saisie()
    except KeyboardInterrupt:
        perception.arret_global.set()
    finally:
        perception.arret_global.set()


if __name__ == "__main__":
    if "--optimize" in sys.argv:
        print(executer_optimisation_nocturne())
        sys.exit(0)

    panels.lancer_avec_interface(BASE_DIR, boucle_principale)