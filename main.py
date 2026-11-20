import datetime
import inspect
import json
import re
import threading
import time
import uuid
from typing import Any, Callable

import ollama

import terminal_panels as panels
from memory_manager import (
    BASE_DIR,
    consulter_historique,
    exporter_historique,
    formater_contexte_memoire,
    journaliser_echange,
    marquer_dernier_echange_echec,
    memoriser_fait_utilisateur,
    rechercher_dans_historique,
)
from perception_eon import MessageEntrant, PerceptionEon
from router import RouteurAutonome
from tts_client import ClientTTSEon

try:
    from context_compressor import GestionnaireContexteMultiNiveaux
except ImportError:
    GestionnaireContexteMultiNiveaux = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import web_tools
except ImportError:
    web_tools = None

try:
    import file_tools
except ImportError:
    file_tools = None

try:
    import shell_tools
except ImportError:
    shell_tools = None

try:
    import optimizer
except ImportError:
    optimizer = None


MODELES_LOURDS_GPU = {
    "gemma3:12b-it-qat",
    "qwen2.5-coder:14b",
    "qwen2.5-coder:7b",
    "qwen2.5:7b",
    "huihui_ai/qwen3-abliterated:8b",
}

OUTILS_SENSIBLES_FICHIERS = {"ecrire_fichier", "modifier_fichier", "supprimer_fichier", "deplacer_fichier"}
OUTILS_SENSIBLES_SHELL = {"executer_commande_shell"}

REGEX_PREFIXE_ASR = re.compile(
    r"^(?:voici\s+)?(?:la\s+)?(?:phrase|transcription)\s+corrig[ée]e\s*(?:est)?\s*:\s*",
    re.IGNORECASE,
)
REGEX_META_REFUS_ASR = re.compile(
    r"^(?:(?:la\s+)?phrase\s+(?:est\s+)?incompl[èe]te|je\s+n['’]ai\s+pas\s+compris|impossible\s+de\s+(?:corriger|comprendre)|aucune\s+correction)\b",
    re.IGNORECASE,
)


def _nettoyer_sortie_asr(texte_whisper: str, sortie_normaliseur: str) -> str:
    """Empêche le normaliseur vocal d'injecter un commentaire méta à la place de la requête."""
    original = texte_whisper.strip()
    propre = (sortie_normaliseur or "").strip()
    if "</think>" in propre:
        propre = propre.split("</think>", 1)[1].strip()

    if not propre or REGEX_META_REFUS_ASR.search(propre):
        return original

    propre = REGEX_PREFIXE_ASR.sub("", propre).strip()
    if (propre.startswith('"') and propre.endswith('"')) or (propre.startswith("«") and propre.endswith("»")):
        propre = propre[1:-1].strip()
    elif propre.startswith('"') and '"' not in propre[1:]:
        propre = propre[1:].strip()

    if not propre or REGEX_META_REFUS_ASR.search(propre):
        return original

    # Garde-fou anti-hallucination si la sortie diverge trop de l'original
    if len(propre) > len(original) * 1.8 + 20:
        return original

    return propre


def _log_outil(message: str) -> None:
    if hasattr(panels, "log_outil"):
        panels.log_outil(message)
    elif hasattr(panels, "log_perception"):
        panels.log_perception(message)


def _extraire_image_camera_bytes(perception: PerceptionEon) -> list[bytes]:
    """Extrait instantanément la dernière frame webcam en JPEG pour un appel VLM direct en 1 passe."""
    if cv2 is None or not perception.camera_active:
        return []
    with perception._lock_vision:
        frame = perception._derniere_frame.copy() if perception._derniere_frame is not None else None
    if frame is None:
        return []
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return [bytes(buf)] if ok else []


def _collecter_fonctions_module(mod: Any, noms_autorises: set[str] | None = None) -> dict[str, Callable]:
    """Récupère les fonctions publiques d'un module d'outils."""
    if mod is None:
        return {}
    if hasattr(mod, "OUTILS") and isinstance(mod.OUTILS, dict):
        return {
            k: v
            for k, v in mod.OUTILS.items()
            if callable(v) and (noms_autorises is None or k in noms_autorises)
        }
    fonctions: dict[str, Callable] = {}
    for nom, obj in inspect.getmembers(mod, inspect.isfunction):
        if nom.startswith("_"):
            continue
        if obj.__module__ != mod.__name__:
            continue
        if noms_autorises is None or nom in noms_autorises:
            fonctions[nom] = obj
    return fonctions


class ApplicationEon:
    """Orchestrateur principal d'Eon (Routage sémantique, mémoire 3 niveaux, pré-chauffage au regard et outils)."""

    def __init__(self) -> None:
        self.session_id = uuid.uuid4().hex[:12]
        self.routeur = RouteurAutonome()
        self.tts = ClientTTSEon(config_eon=self.routeur.config.get("tts", {}))
        self.perception = PerceptionEon(
            config_eon=self.routeur.config.get("perception", {}),
            on_alerte_securite=self._gerer_alerte_securite,
            on_regard_detecte=self._prechauffer_au_regard,
        )
        self.perception.attacher_tts(self.tts)

        embed_model = self.routeur.config.get("routage", {}).get("embed_model", "bge-m3")
        self.compresseur = (
            GestionnaireContexteMultiNiveaux(modele_compresseur="qwen3:0.6b", embed_model=embed_model)
            if GestionnaireContexteMultiNiveaux is not None
            else None
        )

        self.modeles_sans_tools: set[str] = set()
        self.modele_verrouille: str | None = None
        self.dernier_modele: str | None = None
        self.modele_gpu_charge: str | None = None
        self.dernier_prompt_valide: str = ""
        self.derniere_reponse_valide: str = ""
        self.journal_outils_session: list[str] = []

        self._lock_prechauffage = threading.Lock()
        self._cache_capteurs_prechauffes: tuple[float, str] = (0.0, "")

        self.catalogue_web = _collecter_fonctions_module(web_tools)
        self.catalogue_files = _collecter_fonctions_module(file_tools)
        self.catalogue_shell = _collecter_fonctions_module(shell_tools)
        self.catalogue_base: dict[str, Callable] = {
            "observer_camera": self.perception.observer_camera,
            "capturer_photo": self.perception.capturer_photo,
            "consulter_historique": consulter_historique,
            "rechercher_dans_historique": rechercher_dans_historique,
            "exporter_historique": exporter_historique,
            "memoriser_fait_utilisateur": memoriser_fait_utilisateur,
        }

    def _gerer_alerte_securite(self, message_alerte: str) -> None:
        _log_outil(message_alerte)
        panels.ecrire_chat(f"\n⚠️ {message_alerte}")

    def _prechauffer_au_regard(self) -> None:
        """Pré-assemble le contexte et garde le modèle conversationnel chaud dès que l'utilisateur regarde l'écran."""
        if not self._lock_prechauffage.acquire(blocking=False):
            return
        try:
            if self.perception.ia_en_cours.is_set() or self.tts.est_en_train_de_parler():
                return

            etat_frais = self.perception.decrire_etat_capteurs()
            self._cache_capteurs_prechauffes = (time.monotonic(), etat_frais)

            cible = self.modele_verrouille or self.dernier_modele or self.routeur.config.get("route_defaut", "qwen2.5:7B")
            if "12b" in cible.lower() or "14b" in cible.lower():
                cible = self.routeur.config.get("route_defaut", "qwen2.5:7B")

            _log_outil(f"[PRÉDICTIF] Regard détecté -> pré-chauffage contexte & {cible}...")
            ollama.generate(
                model=cible,
                prompt=" ",
                options={"num_predict": 1, "temperature": 0.0},
                keep_alive="2m",
            )
            self.modele_gpu_charge = cible
        except Exception:
            pass
        finally:
            self._lock_prechauffage.release()

    def _obtenir_modele_vlm(self) -> str | None:
        for nom, cfg in self.routeur.config.get("routes", {}).items():
            if cfg.get("role") == "vision_multimodale":
                return nom
        return "gemma3:12b-it-qat"

    def _liberer_vram_si_changement_lourd(self, nouveau_modele: str) -> None:
        """Décharge l'ancien modèle lourd de la RTX 5050 (8 Go) avant d'en charger un autre."""
        ancien = self.modele_gpu_charge
        if (
            ancien
            and ancien != nouveau_modele
            and ancien.lower() in MODELES_LOURDS_GPU
            and nouveau_modele.lower() in MODELES_LOURDS_GPU
        ):
            try:
                _log_outil(f"[VRAM] Déchargement de {ancien} avant chargement de {nouveau_modele}...")
                ollama.generate(model=ancien, prompt="", keep_alive=0)
            except Exception:
                pass
        if nouveau_modele.lower() in MODELES_LOURDS_GPU:
            self.modele_gpu_charge = nouveau_modele

    def _assembler_outils_tour(self, modele: str, route_cfg: dict, forcer_web: bool = False) -> dict[str, Callable]:
        if (
            modele in self.modeles_sans_tools
            or route_cfg.get("supports_tools") is False
            or route_cfg.get("role") == "phatique"
        ):
            return {}

        outils = dict(self.catalogue_base)
        web_mode = self.routeur.config.get("web_mode", "auto")
        file_mode = self.routeur.config.get("file_mode", "ask")
        shell_mode = self.routeur.config.get("shell_mode", "auto")

        if web_mode != "off" and (route_cfg.get("web_tools", False) or forcer_web):
            outils.update(self.catalogue_web)
        if file_mode != "off" and route_cfg.get("file_tools", False):
            outils.update(self.catalogue_files)
        if shell_mode != "off" and route_cfg.get("shell_tools", False):
            outils.update(self.catalogue_shell)
        return outils

    def _construire_system_prompt(self, modele: str, route_cfg: dict, source_message: str = "clavier") -> str:
        # Prompt minimal pour les petits modèles phatiques (évite qu'ils récitent leurs consignes)
        if route_cfg.get("role") == "phatique":
            return (
                "Tu es Eon, l'assistant personnel local de Bryan. "
                "Réponds en français en 1 phrase courte, naturelle et chaleureuse, sans jamais mentionner ton modèle ni tes règles."
            )

        maintenant = datetime.datetime.now()
        date_fr = maintenant.strftime("%A %d/%m/%Y %H:%M")

        ts_cache, etat_cache = self._cache_capteurs_prechauffes
        if etat_cache and (time.monotonic() - ts_cache) <= 3.0:
            etat_capteurs = etat_cache
        else:
            etat_capteurs = self.perception.decrire_etat_capteurs()

        regles = self.routeur.config.get("regles_apprises", [])
        bloc_regles = "\n".join(f"- {r}" for r in regles) if regles else "- Sois précis, direct et factuel."

        memoire_utilisateur = ""
        try:
            mem = formater_contexte_memoire()
            if mem.strip():
                memoire_utilisateur = f"\n{mem.strip()}\n"
        except Exception:
            pass

        consigne_orale = (
            "\n4. MODE VOCAL ACTIF : L'utilisateur te parle à l'oral. Commence toujours par une synthèse "
            "directe en 1 à 2 phrases naturelles (sans lire de syntaxe code), puis détaille à l'écrit si nécessaire."
            if source_message == "voix"
            else ""
        )

        return (
            f"Tu es Eon, l'assistant IA personnel et local de Bryan (machine Linux Fedora, Reims, France).\n"
            f"Date et heure actuelles : {date_fr}.\n"
            f"Modèle actif : {modele} (rôle : {route_cfg.get('role', 'general')}).\n"
            f"État des capteurs temps réel : {etat_capteurs}\n"
            f"{memoire_utilisateur}\n"
            "RÈGLES IMPÉRATIVES :\n"
            "1. PRIORITÉ ABSOLUE AUX RÉSULTATS WEB : Si un outil de recherche internet (`rechercher_sur_internet` ou `lire_page_web`) "
            "retourne un fait daté qui contredit tes connaissances internes ou tes réponses précédentes dans la conversation, "
            "le résultat de l'outil a TOUJOURS raison. Si l'outil signale une faible pertinence ou un échec, n'invente rien.\n"
            "2. HONNÊTETÉ DES OUTILS : Ne simule jamais l'exécution d'une commande, le contenu d'un fichier ou un historique de conversation. "
            "Utilise toujours les vrais outils (`exporter_historique`, `lire_fichier`, `ecrire_fichier`, `executer_commande_shell`, `observer_camera`).\n"
            "3. CONCISION ET TON : Réponds en français naturel, direct et efficace, sans blabla inutile."
            f"{consigne_orale}\n"
            f"{bloc_regles}"
        )

    def _demander_confirmation_outil(self, nom_outil: str, args: dict, source_message: str = "clavier") -> bool:
        """Demande l'accord de l'utilisateur (clavier ou voix) pour un outil sensible."""
        resume_args = json.dumps(args, ensure_ascii=False)[:180]
        panels.ecrire_chat(f"\n[AUTORISATION REQUISE] Exécuter `{nom_outil}({resume_args})` ? (oui/non)")
        if source_message == "voix":
            self.tts.alerter(f"Confirmation requise pour exécuter {nom_outil}. Dis oui ou non.")

        self.perception.autoriser_saisie()
        reponse = self.perception.attendre_message()
        if reponse is None:
            return False
        accord, raison = self.routeur.interpreter_consentement(reponse.texte)
        _log_outil(f"[CONSENTEMENT] {nom_outil} -> {'AUTORISÉ' if accord else 'REFUSÉ'} ({raison})")
        return accord

    def _executer_outil(
        self,
        nom_outil: str,
        args: dict,
        outils_disponibles: dict[str, Callable],
        source_message: str = "clavier",
    ) -> str:
        fn = outils_disponibles.get(nom_outil)
        if fn is None:
            return f"Erreur : outil `{nom_outil}` non autorisé ou indisponible pour ce modèle."

        file_mode = self.routeur.config.get("file_mode", "ask")
        shell_mode = self.routeur.config.get("shell_mode", "auto")

        if nom_outil in OUTILS_SENSIBLES_FICHIERS and file_mode == "ask":
            if not self._demander_confirmation_outil(nom_outil, args, source_message=source_message):
                return "Action annulée par l'utilisateur."
        elif nom_outil in OUTILS_SENSIBLES_SHELL and shell_mode == "ask":
            if not self._demander_confirmation_outil(nom_outil, args, source_message=source_message):
                return "Commande shell annulée par l'utilisateur."

        ts = datetime.datetime.now().strftime("%H:%M:%S")
        entree_trace = f"[{ts}] Appel outil : {nom_outil}({args})"
        self.journal_outils_session.append(entree_trace)
        _log_outil(entree_trace)

        try:
            sig = inspect.signature(fn)
            accepte_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            if accepte_kwargs:
                args_filtres = args
            else:
                args_filtres = {k: v for k, v in args.items() if k in sig.parameters}

            resultat = fn(**args_filtres)
            texte_res = str(resultat)
        except Exception as err:
            texte_res = f"Erreur lors de l'exécution de `{nom_outil}` : {type(err).__name__} - {err}"

        retour_trace = f"[{ts}] Retour {nom_outil} : {texte_res[:350]}"
        self.journal_outils_session.append(retour_trace)
        _log_outil(retour_trace)
        return texte_res

    def executer_tour_autonome(
        self,
        modele: str,
        route_cfg: dict,
        prompt_utilisateur: str,
        historique: list[dict[str, Any]],
        images: list[bytes] | None = None,
        forcer_web: bool = False,
        source_message: str = "clavier",
    ) -> str:
        """Exécute la boucle autonome d'appel d'outils, de streaming TUI et de synthèse vocale."""
        self._liberer_vram_si_changement_lourd(modele)
        self.tts.demarrer_tour_reponse(source_message)

        outils_dispo = self._assembler_outils_tour(modele, route_cfg, forcer_web=forcer_web)
        liste_callables = list(outils_dispo.values()) if not images else []

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._construire_system_prompt(modele, route_cfg, source_message=source_message)}
        ]
        messages.extend(historique)

        msg_user: dict[str, Any] = {"role": "user", "content": prompt_utilisateur}
        if images:
            msg_user["images"] = images
        messages.append(msg_user)

        options_ollama: dict[str, Any] = {
            "temperature": float(route_cfg.get("temperature") if route_cfg.get("temperature") is not None else 0.2),
            "num_ctx": int(route_cfg.get("num_ctx") or 4096),
            "num_predict": int(route_cfg.get("num_predict") or 768),
        }
        if route_cfg.get("num_gpu") is not None:
            options_ollama["num_gpu"] = int(route_cfg["num_gpu"])

        tts_actif = self.tts.daemon_disponible()
        if "12b" in modele.lower() or "14b" in modele.lower():
            keep_alive = 0 if tts_actif else "30s"
        else:
            keep_alive = "45s" if tts_actif else "5m"

        self.perception.interruption_event.clear()
        self.perception.ia_en_cours.set()

        reponse_finale_chunks: list[str] = []
        appels_deja_faits: set[str] = set()
        max_etapes = 6

        try:
            for _etape in range(max_etapes):
                if self.perception.interruption_event.is_set():
                    reponse_finale_chunks.append("\n[Interrompu]")
                    break

                kwargs_chat: dict[str, Any] = {
                    "model": modele,
                    "messages": messages,
                    "options": options_ollama,
                    "keep_alive": keep_alive,
                    "stream": True,
                }
                if liste_callables:
                    kwargs_chat["tools"] = liste_callables

                try:
                    if "qwen3" in modele.lower():
                        flux = ollama.chat(**kwargs_chat, think=False)
                    else:
                        flux = ollama.chat(**kwargs_chat)
                except TypeError:
                    flux = ollama.chat(**kwargs_chat)
                except ollama.ResponseError as err:
                    if "does not support tools" in str(err).lower() and "tools" in kwargs_chat:
                        self.modeles_sans_tools.add(modele)
                        liste_callables = []
                        kwargs_chat.pop("tools", None)
                        _log_outil(f"[TOOLS] {modele} ne supporte pas les outils -> bascule texte direct.")
                        flux = ollama.chat(**kwargs_chat)
                    else:
                        raise

                texte_etape = ""
                tool_calls_etape: list[Any] = []
                dans_balise_think = False

                for chunk in flux:
                    if self.perception.interruption_event.is_set():
                        break

                    msg_chunk = chunk.get("message") if isinstance(chunk, dict) else getattr(chunk, "message", None)
                    if msg_chunk is None:
                        continue

                    t_calls = (
                        msg_chunk.get("tool_calls")
                        if isinstance(msg_chunk, dict)
                        else getattr(msg_chunk, "tool_calls", None)
                    )
                    if t_calls:
                        tool_calls_etape.extend(t_calls)

                    contenu = (
                        msg_chunk.get("content", "")
                        if isinstance(msg_chunk, dict)
                        else getattr(msg_chunk, "content", "")
                    )
                    if contenu:
                        if "<think>" in contenu:
                            dans_balise_think = True
                            contenu = contenu.split("<think>", 1)[0]
                        if "</think>" in contenu:
                            dans_balise_think = False
                            contenu = contenu.split("</think>", 1)[1]
                        if dans_balise_think or not contenu:
                            continue

                        if not texte_etape and not reponse_finale_chunks:
                            panels.ecrire_chat(f"\nEon ({modele}) : ", fin="")
                        panels.ecrire_chat(contenu, fin="")
                        self.tts.pousser_token_stream(contenu)
                        texte_etape += contenu
                        reponse_finale_chunks.append(contenu)

                if self.perception.interruption_event.is_set():
                    self.tts.arreter()
                    panels.ecrire_chat("\n[Génération interrompue]")
                    break

                if not tool_calls_etape:
                    break

                messages.append(
                    {
                        "role": "assistant",
                        "content": texte_etape,
                        "tool_calls": tool_calls_etape,
                    }
                )

                for tc in tool_calls_etape:
                    if self.perception.interruption_event.is_set():
                        break
                    fn_info = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
                    nom_fn = fn_info.get("name") if isinstance(fn_info, dict) else getattr(fn_info, "name", "")
                    raw_args = fn_info.get("arguments", {}) if isinstance(fn_info, dict) else getattr(fn_info, "arguments", {})

                    if isinstance(raw_args, str):
                        try:
                            args_dict = json.loads(raw_args)
                        except json.JSONDecodeError:
                            args_dict = {}
                    else:
                        args_dict = dict(raw_args or {})

                    signature_appel = f"{nom_fn}:{json.dumps(args_dict, sort_keys=True)}"
                    if signature_appel in appels_deja_faits:
                        res_outil = (
                            "Tu as déjà exécuté exactement cet appel d'outil avec ces mêmes paramètres. "
                            "Utilise directement les résultats obtenus précédemment pour répondre."
                        )
                    else:
                        appels_deja_faits.add(signature_appel)
                        res_outil = self._executer_outil(
                            nom_fn,
                            args_dict,
                            outils_dispo,
                            source_message=source_message,
                        )

                    messages.append(
                        {
                            "role": "tool",
                            "content": res_outil,
                        }
                    )

            if not self.perception.interruption_event.is_set():
                self.tts.vider_stream()
            panels.ecrire_chat("")
            panels.flush_chat()
        except Exception as err:
            err_msg = f"\n[Erreur modèle {modele}] {type(err).__name__} : {err}"
            panels.ecrire_chat(err_msg)
            panels.flush_chat()
            reponse_finale_chunks.append(err_msg)
        finally:
            self.perception.ia_en_cours.clear()
            self.perception.interruption_event.clear()

        return "".join(reponse_finale_chunks).strip()

    # ---------------- COMMANDES SLASH ----------------

    def traiter_commande_slash(self, cmd_brute: str) -> bool:
        """Traite les commandes `/...`. Retourne False si l'utilisateur demande à quitter."""
        morceaux = cmd_brute.strip().split(maxsplit=1)
        cmd = morceaux[0].lower()
        arg = morceaux[1].strip() if len(morceaux) > 1 else ""

        if cmd in ("/quit", "/exit"):
            self.tts.arreter()
            self.perception.arret_global.set()
            return False

        if cmd == "/help":
            panels.ecrire_chat(
                "\nCommandes disponibles :\n"
                "  @<alias> <prompt>      : Forcer un modèle sur un seul message (ex: @code+, @12b, @c)\n"
                "  /model <alias|auto>    : Verrouiller un modèle ou revenir au routage automatique\n"
                "  /models                : Lister les modèles, alias et rôles configurés\n"
                "  /tts [status|test|stop]: Tester ou contrôler le démon vocal Chatterbox\n"
                "  /history [N]           : Afficher les N derniers échanges et les logs d'outils\n"
                "  /export [chemin]       : Exporter l'historique réel dans un fichier texte\n"
                "  /enroll [nom|reset]    : Enregistrer un visage autorisé ou réinitialiser la biométrie\n"
                "  /secu [auto|absent|off]: Régler le mode de surveillance caméra\n"
                "  /capteurs              : Afficher l'état temps réel de la caméra et du micro\n"
                "  /web|file|shell <mode> : Changer les permissions d'outils (auto | ask | off)\n"
                "  /forget                : Purger le contexte actif en RAM\n"
                "  /reload                : Recharger config_ia.json et entrainement_routage.json\n"
                "  /optimize [min]        : Lancer l'optimisation fine en arrière-plan (défaut 55 min)\n"
                "  /quit                  : Quitter Eon"
            )
            return True

        if cmd == "/tts":
            sous = arg.lower() or "status"
            if sous == "stop":
                self.tts.arreter()
                panels.ecrire_chat("\n[TTS] Lecture vocale interrompue.")
            elif sous == "test":
                if self.tts.daemon_disponible():
                    self.tts.alerter("Liaison vocale opérationnelle entre Eon et le démon Chatterbox.")
                    panels.ecrire_chat("\n[TTS] Test envoyé au démon vocal.")
                else:
                    panels.ecrire_chat(f"\n[TTS] Démon inactif sur {self.tts.socket_path}. Lance `IA_voice_lab/tts_daemon.py`.")
            else:
                etat = "ACTIF" if self.tts.daemon_disponible() else "INACTIF"
                panels.ecrire_chat(f"\n[TTS] Statut : {etat} | Socket : {self.tts.socket_path}")
            return True

        if cmd == "/optimize":
            if optimizer is None:
                panels.ecrire_chat("\nModule optimizer.py introuvable.")
                return True
            duree = int(arg) if arg.isdigit() else 55
            panels.ecrire_chat(
                f"\nLancement de l'optimisation fine en arrière-plan (budget max {duree} min, contrôle thermique actif)..."
            )

            def _run_opti() -> None:
                rapport = optimizer.executer_optimisation_nocturne(duree_max_minutes=duree)
                self.routeur.charger(silencieux=True)
                panels.ecrire_chat(f"\n{rapport}")

            threading.Thread(target=_run_opti, daemon=True).start()
            return True

        if cmd == "/models":
            lignes = ["\nModèles configurés :"]
            for nom, cfg in self.routeur.config.get("routes", {}).items():
                marqueur = " [VERROUILLÉ]" if self.modele_verrouille == nom else ""
                lignes.append(
                    f"  - {nom} (@{cfg.get('alias', '?')}) -> rôle={cfg.get('role')} | "
                    f"web={cfg.get('web_tools')} file={cfg.get('file_tools')} shell={cfg.get('shell_tools')}{marqueur}"
                )
            panels.ecrire_chat("\n".join(lignes))
            return True

        if cmd == "/model":
            if not arg or arg.lower() == "auto":
                self.modele_verrouille = None
                panels.ecrire_chat("Routage automatique réactivé.")
            else:
                resolu = self.routeur.resoudre_modele(arg)
                if resolu:
                    self.modele_verrouille = resolu
                    panels.ecrire_chat(f"Modèle verrouillé sur : {resolu}")
                else:
                    panels.ecrire_chat(f"Modèle inconnu : {arg}. Tape `/models` pour voir la liste.")
            return True

        if cmd == "/history":
            limite = int(arg) if arg.isdigit() else 30
            hist = consulter_historique(inclure_episodes=True, limite=limite)
            panels.ecrire_chat(f"\n{hist}")
            if self.journal_outils_session:
                panels.ecrire_chat("\n--- DERNIERS APPELS OUTILS (SESSION) ---")
                panels.ecrire_chat("\n".join(self.journal_outils_session[-30:]))
            return True

        if cmd == "/export":
            res = exporter_historique(chemin_destination=arg or "~/Téléchargements/conversation_eon.txt", limite=150)
            panels.ecrire_chat(f"\n{res}")
            return True

        if cmd == "/enroll":
            if arg.lower() == "reset":
                panels.ecrire_chat(f"\n{self.perception.reinitialiser_profil_biometrique()}")
            elif arg.lower().startswith("del "):
                panels.ecrire_chat(f"\n{self.perception.reinitialiser_profil_biometrique(arg[4:])}")
            else:
                panels.ecrire_chat(f"\n{self.perception.enregistrer_visage_autorise(arg or 'Bryan')}")
            return True

        if cmd == "/secu":
            if arg.lower() in ("auto", "absent", "off"):
                self.perception.mode_securite = arg.lower()
                panels.ecrire_chat(f"\nMode sécurité caméra : {self.perception.mode_securite}")
            else:
                panels.ecrire_chat(f"\nMode actuel : {self.perception.mode_securite} (options : auto | absent | off)")
            return True

        if cmd == "/capteurs":
            panels.ecrire_chat(f"\n{self.perception.decrire_etat_capteurs()}")
            return True

        if cmd in ("/web", "/file", "/shell"):
            cle = f"{cmd[1:]}_mode"
            if arg.lower() in ("auto", "ask", "off"):
                self.routeur.config[cle] = arg.lower()
                self.routeur.sauvegarder_config()
                panels.ecrire_chat(f"\n{cle} réglé sur : {arg.lower()}")
            else:
                panels.ecrire_chat(f"\nValeur actuelle {cle} : {self.routeur.config.get(cle)} (auto | ask | off)")
            return True

        if cmd in ("/forget", "/reset"):
            if self.compresseur is not None:
                with self.compresseur._lock:
                    self.compresseur.resume_actif = ""
                    self.compresseur.vecteur_actif = []
                    self.compresseur.fenetre_ram.clear()
            self.dernier_modele = None
            panels.ecrire_chat("\nContexte actif en RAM purgé.")
            return True

        if cmd == "/reload":
            self.routeur.charger(silencieux=False)
            panels.ecrire_chat("\nConfiguration et fichier d'entraînement sémantique rechargés.")
            return True

        panels.ecrire_chat(f"\nCommande inconnue : {cmd} (tape `/help`)")
        return True

    # ---------------- BOUCLE PRINCIPALE ----------------

    def boucle_principale(self) -> None:
        self.perception.demarrer()
        etat_tts = "voix active" if self.tts.daemon_disponible() else "démon vocal en cours d'initialisation"
        panels.ecrire_chat(
            f"Eon est prêt (Routage sémantique + Mémoire 3 niveaux + {etat_tts}). Tape `/help` pour les commandes."
        )

        while not self.perception.arret_global.is_set():
            self.perception.autoriser_saisie()
            entrant: MessageEntrant | None = self.perception.attendre_message()
            if entrant is None:
                break

            msg = entrant.texte.strip()
            if not msg:
                continue

            if self.routeur.verifier_hot_reload():
                _log_outil("[ROUTER] Rechargement à chaud de la configuration / dataset sémantique.")

            if msg.startswith("/"):
                if not self.traiter_commande_slash(msg):
                    break
                continue

            modele_force: str | None = None
            prompt_effectif = msg
            match_at = re.match(r"^@([^\s]+)\s+(.+)$", msg, flags=re.S)
            if match_at:
                cand = self.routeur.resoudre_modele(match_at.group(1))
                if cand:
                    modele_force = cand
                    prompt_effectif = match_at.group(2).strip()

            if getattr(entrant, "source", "") == "voix":
                panels.ecrire_chat(f"\nVous (voix) : {prompt_effectif}")
                brut_normalise = self.routeur.normaliser_intention_vocale(
                    prompt_effectif,
                    dernier_prompt=self.dernier_prompt_valide,
                    derniere_reponse=self.derniere_reponse_valide,
                )
                prompt_normalise = _nettoyer_sortie_asr(prompt_effectif, brut_normalise)
                if prompt_normalise != prompt_effectif:
                    _log_outil(f"[ASR FIX] « {prompt_effectif} » -> « {prompt_normalise} »")
                    prompt_effectif = prompt_normalise

            # 1. Évaluation d'insatisfaction / rejet de la réponse précédente
            est_echec = False
            if self.dernier_prompt_valide and self.derniere_reponse_valide:
                est_echec, raison_echec = self.routeur.evaluer_insatisfaction(
                    prompt_effectif,
                    self.dernier_prompt_valide,
                    self.derniere_reponse_valide,
                )
                if est_echec:
                    _log_outil(f"[AUTO-CORRECTION] Insatisfaction détectée ({raison_echec}).")
                    try:
                        marquer_dernier_echange_echec(f"{raison_echec} | Retour utilisateur : {prompt_effectif[:160]}")
                    except Exception:
                        pass

            # 2. Gestionnaire de contexte 3 niveaux (VRAM / RAM / Disque)
            if self.compresseur is not None:
                texte_routage, mode_ctx, historique_tour = self.compresseur.preparer_contexte_routage(prompt_effectif)
                if mode_ctx in ("rupture_sujet", "rappel_disque"):
                    _log_outil(f"[MÉMOIRE 3-NIVEAUX] Transition : {mode_ctx}")
                    if mode_ctx == "rupture_sujet":
                        self.dernier_modele = None
            else:
                texte_routage = prompt_effectif
                historique_tour = []

            # 3. Analyse sémantique des intentions latentes
            intentions = self.routeur.analyser_intentions(
                prompt_effectif,
                dernier_prompt=self.dernier_prompt_valide,
                derniere_reponse=self.derniere_reponse_valide,
            )
            seuil_intent = float(
                self.routeur.config.get("routage", {}).get("seuils", {}).get("intention_outil") or 0.58
            )

            if intentions.get("purge_contexte", 0.0) >= seuil_intent or est_echec:
                historique_tour = []
                if self.compresseur is not None:
                    with self.compresseur._lock:
                        self.compresseur.resume_actif = ""
                        self.compresseur.vecteur_actif = []
                        self.compresseur.fenetre_ram.clear()
                _log_outil("[CONTEXTE] Historique actif purgé pour repartir sur une base saine.")

            # Attachement webcam direct en 1 passe si intention visuelle détectée
            images_tour: list[bytes] | None = None
            if modele_force is None and intentions.get("vision_camera", 0.0) >= seuil_intent:
                modele_vlm = self._obtenir_modele_vlm()
                imgs = _extraire_image_camera_bytes(self.perception)
                if modele_vlm and imgs:
                    modele_force = modele_vlm
                    images_tour = imgs
                    _log_outil(f"[VISION 1-PASSE] Frame webcam jointe directement à {modele_vlm}.")

            # Directives système dynamiques selon les intentions sémantiques
            forcer_web = False
            consignes_dynamiques: list[str] = []
            if intentions.get("web_obligatoire", 0.0) >= seuil_intent:
                forcer_web = True
                consignes_dynamiques.append(
                    "Appelle obligatoirement `rechercher_sur_internet` avant de répondre et base-toi uniquement sur ses résultats."
                )
            if intentions.get("export_historique", 0.0) >= seuil_intent:
                consignes_dynamiques.append(
                    "Utilise l'outil `exporter_historique` (ou `consulter_historique`) pour récupérer les vrais échanges sans jamais les inventer."
                )

            prompt_pour_llm = prompt_effectif
            if consignes_dynamiques:
                prompt_pour_llm += "\n[CONSIGNE SYSTÈME : " + " ".join(consignes_dynamiques) + "]"

            # 4. Choix du modèle optimal (Embedding contextualisé + Superviseur IA)
            modele_final, score_route, raison_route = self.routeur.choisir_modele(
                texte_routage,
                dernier_modele=self.dernier_modele,
                modele_force=modele_force,
                modele_verrouille=self.modele_verrouille,
                dernier_prompt=self.dernier_prompt_valide,
                derniere_reponse=self.derniere_reponse_valide,
            )
            route_cfg = self.routeur.config.get("routes", {}).get(modele_final, {})
            _log_outil(f"[ROUTE] {modele_final} (score={score_route:.2f} | {raison_route})")

            # 5. Exécution autonome (streaming + outils + voix si source == "voix")
            reponse = self.executer_tour_autonome(
                modele=modele_final,
                route_cfg=route_cfg,
                prompt_utilisateur=prompt_pour_llm,
                historique=historique_tour,
                images=images_tour,
                forcer_web=forcer_web,
                source_message=entrant.source,
            )

            # 6. Mise à jour mémoire SQLite + compression asynchrone en RAM (0 ms de latence)
            if reponse:
                self.dernier_modele = modele_final
                self.dernier_prompt_valide = prompt_effectif
                self.derniere_reponse_valide = reponse

                try:
                    journaliser_echange(
                        {
                            "session_id": self.session_id,
                            "date": datetime.datetime.now().isoformat(timespec="seconds"),
                            "modele": modele_final,
                            "raison": raison_route,
                            "score": score_route,
                            "source": entrant.source,
                            "prompt": prompt_effectif,
                            "reponse": reponse,
                        }
                    )
                except Exception:
                    pass

                if self.compresseur is not None:
                    self.compresseur.mettre_a_jour_asynchrone(prompt_effectif, reponse, modele_final)


def main() -> None:
    app = ApplicationEon()

    try:
        panels.connecter_capteurs_et_voix(perception=app.perception, tts=app.tts)
        panels.enregistrer_callbacks_ui(
            on_prompt=app.perception.soumettre_texte_clavier,
            on_escape=app.perception.declencher_interruption,
        )
        panels.lancer_avec_interface(BASE_DIR, app.boucle_principale)
    except KeyboardInterrupt:
        pass
    finally:
        app.tts.arreter()
        app.perception.arret_global.set()
        panels.fermer()


if __name__ == "__main__":
    main()