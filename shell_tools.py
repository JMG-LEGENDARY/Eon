import os
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import terminal_panels as panels

BASE_DIR = Path(__file__).resolve().parent
MODE_SHELL = "auto"  # La confirmation est gérée en amont par main.py (ou via CALLBACK_CONFIRMATION)
CALLBACK_CONFIRMATION: Callable[[str], bool] | None = None

COMMANDES_BLOQUEES = re.compile(
    r"(\brm\s+-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*\s+(/|~|\$HOME|\*)\b|"
    r"\bmkfs\b|\bdd\s+if=|:\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;|"
    r"\b(shutdown|reboot|poweroff|init\s+0)\b|>\s*/dev/sd[a-z])",
    re.IGNORECASE,
)
REGEX_SUDO = re.compile(r"(^|[;&|]\s*)sudo\b")
REGEX_COMMANDE_LONGUE = re.compile(r"\b(dnf|flatpak|cargo|pip|uv|npm|git\s+clone)\b", re.IGNORECASE)
REGEX_DNF_INTERACTIF = re.compile(r"\bdnf\s+(install|upgrade|update|remove|autoremove)\b(?![^\n;|&]*\s-y\b)", re.IGNORECASE)


@dataclass
class TacheShellActive:
    commande: str
    processus: subprocess.Popen
    debut: float = field(default_factory=time.monotonic)
    derniere_activite: float = field(default_factory=time.monotonic)
    lignes: list[str] = field(default_factory=list)
    code_retour: int | None = None
    terminee: threading.Event = field(default_factory=threading.Event)

    def duree_ecoulee(self) -> float:
        return time.monotonic() - self.debut

    def secondes_sans_activite(self) -> float:
        return time.monotonic() - self.derniere_activite

    def est_bloquee(self, seuil_sec: float = 20.0) -> bool:
        return not self.terminee.is_set() and self.secondes_sans_activite() >= seuil_sec


TACHE_EN_COURS: TacheShellActive | None = None
_LOCK_TACHE = threading.Lock()


def obtenir_tache_en_cours() -> TacheShellActive | None:
    with _LOCK_TACHE:
        if TACHE_EN_COURS is not None and not TACHE_EN_COURS.terminee.is_set():
            return TACHE_EN_COURS
        return None


def arreter_tache_en_cours() -> str:
    """Stoppe proprement la commande système en cours d'exécution."""
    tache = obtenir_tache_en_cours()
    if tache is None:
        return "Aucune commande système n'est actuellement en cours."
    try:
        os.killpg(os.getpgid(tache.processus.pid), signal.SIGTERM)
        tache.terminee.wait(timeout=3)
    except Exception:
        try:
            tache.processus.kill()
        except Exception:
            pass
    panels.log_outil(f"[STOP] Tâche arrêtée : {tache.commande}")
    return f"La tâche « {tache.commande} » a été arrêtée."


def statut_tache_en_cours() -> str:
    """Retourne l'état et les dernières lignes de la tâche en arrière-plan."""
    with _LOCK_TACHE:
        t = TACHE_EN_COURS
    if t is None:
        return "Aucune tâche système récente."
    extrait = "\n".join(t.lignes[-12:]).strip() or "(aucune sortie texte pour le moment)"
    if t.terminee.is_set():
        return f"Tâche terminée (code {t.code_retour}) : « {t.commande} »\nDernières lignes :\n{extrait}"
    etat = "BLOQUÉE / EN ATTENTE RÉSEAU" if t.est_bloquee() else "EN COURS"
    return (
        f"Statut : {etat} ({t.duree_ecoulee():.0f}s écoulées, sans nouvelle sortie depuis {t.secondes_sans_activite():.0f}s)\n"
        f"Commande : « {t.commande} »\nDernières lignes :\n{extrait}"
    )


def _normaliser_commande_llm(commande: str) -> str:
    """Corrige les artefacts d'échappement JSON courants produits par les LLM locaux."""
    cmd = commande.strip()
    # Corrige les subshells mal échappés par le parser JSON (ex: `\\(seq 1 10\\)` -> `$(seq 1 10)`)
    cmd = re.sub(r"\\+\(([^()]+)\\+\)", r"$(\1)", cmd)
    # Évite qu'un dnf interactif se fige sur [o/N]
    if REGEX_DNF_INTERACTIF.search(cmd):
        cmd = REGEX_DNF_INTERACTIF.sub(r"dnf -y \1", cmd)
    return cmd


def _preparer_sudo(commande: str) -> tuple[str, str | None]:
    if not REGEX_SUDO.search(commande):
        return commande, None

    if re.search(r"\bsudo\s+(-S|-n)\b", commande):
        return commande, None

    test_ticket = subprocess.run(["sudo", "-n", "true"], capture_output=True)
    if test_ticket.returncode == 0:
        return REGEX_SUDO.sub(r"\1sudo -n", commande), None

    mdp = panels.saisir_mot_de_passe("[Sudo] Mot de passe administrateur : ")
    commande_stdin = REGEX_SUDO.sub(r'\1sudo -S -p ""', commande)
    return commande_stdin, f"{mdp}\n"


def executer_commande_shell(commande: str, dossier_travail: str | None = None) -> str:
    """Exécute une commande bash/shell locale sans bloquer Eon si la tâche est longue ou figée.

    Args:
        commande: La commande bash à exécuter.
        dossier_travail: Répertoire d'exécution optionnel.

    Returns:
        Le code retour et la sortie de la commande (ou son état en arrière-plan).
    """
    global TACHE_EN_COURS

    if MODE_SHELL == "off":
        return "Erreur : exécution shell désactivée (/shell off)."

    tache_active = obtenir_tache_en_cours()
    if tache_active is not None:
        return (
            f"TÂCHE DÉJÀ EN COURS : « {tache_active.commande} » tourne depuis {tache_active.duree_ecoulee():.0f}s "
            f"(inactive depuis {tache_active.secondes_sans_activite():.0f}s). "
            "Demande à l'utilisateur s'il souhaite l'arrêter avant d'en lancer une autre."
        )

    cmd_propre = _normaliser_commande_llm(commande)
    if not cmd_propre:
        return "Erreur : commande vide."

    if COMMANDES_BLOQUEES.search(cmd_propre):
        return f"BLOQUÉ PAR SÉCURITÉ : la commande « {cmd_propre} » est interdite."

    if dossier_travail:
        candidat = Path(dossier_travail).expanduser()
        cwd = candidat.resolve() if candidat.is_absolute() else (BASE_DIR / candidat).resolve()
    else:
        cwd = BASE_DIR

    if not cwd.exists() or not cwd.is_dir():
        cwd = BASE_DIR

    if MODE_SHELL == "ask" and CALLBACK_CONFIRMATION is not None:
        if not CALLBACK_CONFIRMATION(f"[{cwd}] $ {cmd_propre}"):
            return "REFUS UTILISATEUR : l'utilisateur a refusé d'exécuter cette commande. Ne la simule pas."

    try:
        cmd_exec, stdin_sudo = _preparer_sudo(cmd_propre)
    except (KeyboardInterrupt, EOFError):
        return "Annulé : saisie du mot de passe sudo interrompue."

    delai_max = 900 if REGEX_COMMANDE_LONGUE.search(cmd_propre) else 45
    panels.log_outil(f"$ {cmd_propre}")

    env_exec = os.environ.copy()
    env_exec["GIT_TERMINAL_PROMPT"] = "0"
    env_exec["DEBIAN_FRONTEND"] = "noninteractive"

    try:
        proc = subprocess.Popen(
            cmd_exec,
            shell=True,
            cwd=str(cwd),
            env=env_exec,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        if proc.stdin is not None:
            try:
                if stdin_sudo is not None:
                    proc.stdin.write(stdin_sudo.encode("utf-8"))
                proc.stdin.close()
            except BrokenPipeError:
                pass

        tache = TacheShellActive(commande=cmd_propre, processus=proc)
        with _LOCK_TACHE:
            TACHE_EN_COURS = tache

        def _lecteur_fond() -> None:
            tampon = ""
            try:
                if proc.stdout is not None:
                    fd = proc.stdout.fileno()
                    while True:
                        bloc = os.read(fd, 512)
                        if not bloc:
                            break
                        tache.derniere_activite = time.monotonic()
                        texte = bloc.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
                        tampon += texte
                        while "\n" in tampon:
                            ligne, tampon = tampon.split("\n", 1)
                            ligne_propre = ligne.rstrip()
                            if ligne_propre:
                                tache.lignes.append(ligne_propre)
                                if len(tache.lignes) > 400:
                                    tache.lignes.pop(0)
                                panels.log_outil(f"  | {ligne_propre}")
                    if tampon.strip():
                        tache.lignes.append(tampon.strip())
                        panels.log_outil(f"  | {tampon.strip()}")
                tache.code_retour = proc.wait(timeout=delai_max)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                tache.code_retour = -1
                panels.log_outil(f"[TIMEOUT] Commande interrompue après {delai_max}s.")
            except Exception as err:
                tache.code_retour = -1
                panels.log_outil(f"[ERREUR SHELL] {err}")
            finally:
                tache.terminee.set()
                panels.log_outil(f"[FIN code={tache.code_retour}] {cmd_propre}")

        threading.Thread(target=_lecteur_fond, daemon=True).start()

        if tache.terminee.wait(timeout=8.0):
            sortie_complete = "\n".join(tache.lignes).strip()
            if len(sortie_complete) > 4000:
                sortie_complete = sortie_complete[:1500] + "\n... [milieu tronqué] ...\n" + sortie_complete[-2200:]

            info_code = f"Code retour : {tache.code_retour}"
            if "check-update" in cmd_propre and tache.code_retour == 100:
                info_code += " (des mises à jour sont disponibles)"
            elif tache.code_retour != 0:
                info_code += " (ÉCHEC DE LA COMMANDE : ne prétends pas qu'elle a réussi, analyse l'erreur ci-dessous)"

            return f"{info_code}\nSORTIE :\n{sortie_complete}" if sortie_complete else f"{info_code}\n(Aucune sortie texte)"

        extrait_debut = "\n".join(tache.lignes[-6:]).strip()
        return (
            f"TÂCHE EN COURS EN ARRIÈRE-PLAN (PID {proc.pid}) : « {cmd_propre} » continue en fond.\n"
            f"Progression visible dans le panneau Outils.\n"
            f"Dernières lignes :\n{extrait_debut or '(en attente de sortie...)'}\n"
            "Informe brièvement l'utilisateur que la tâche tourne en arrière-plan et qu'il peut continuer à te parler."
        )

    except Exception as err:
        return f"Erreur d'exécution : {err}"