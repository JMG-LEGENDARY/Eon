import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

import terminal_panels as panels

BASE_DIR = Path(__file__).resolve().parent
SOCKET_TTS_DEFAUT = "/tmp/eon_tts.sock"
VOICE_LAB_DIR = BASE_DIR.parent / "IA_voice_lab"


def nettoyer_texte_pour_voix(texte: str) -> str:
    """Transforme une réponse Markdown/technique en texte fluide prononçable."""
    texte = re.sub(r"```[\s\S]*?```", " Je t'ai affiché le code à l'écran. ", texte)
    texte = re.sub(r"(?:Je t'ai affiché le code à l'écran\.\s*){2,}", "Je t'ai affiché le code à l'écran. ", texte)
    texte = re.sub(r"`([^`]+)`", r"\1", texte)
    texte = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", texte)
    texte = re.sub(r"https?://\S+", "lien affiché à l'écran", texte)
    texte = re.sub(r"[#*_~>|]+", " ", texte)
    texte = (
        texte.replace("->", " vers ")
        .replace("=>", " donne ")
        .replace("!=", " différent de ")
        .replace("==", " égal ")
        .replace(">=", " supérieur ou égal à ")
        .replace("<=", " inférieur ou égal à ")
    )
    return re.sub(r"\s+", " ", texte).strip()


class ClientTTSEon:
    """
    Client IPC vers le démon Chatterbox (/tmp/eon_tts.sock) avec auto-démarrage
    du serveur vocal situé dans ../IA_voice_lab/tts_daemon.py.
    """

    def __init__(self, config_eon: dict | None = None) -> None:
        self.cfg = config_eon or {}
        self.socket_path = str(self.cfg.get("tts_socket_path", SOCKET_TTS_DEFAUT))
        self.max_phrases_orales = int(self.cfg.get("tts_max_phrases_orales", 3))
        self.max_caracteres_oraux = int(self.cfg.get("tts_max_caracteres_oraux", 380))
        self.auto_start = bool(self.cfg.get("tts_auto_start", True))

        self._lock = threading.Lock()
        self._tampon_stream: str = ""
        self._phrases_emises_tour: int = 0
        self._caracteres_emis_tour: int = 0
        self._plafond_atteint: bool = False
        self._mode_vocal_actif_tour: bool = False
        self._fin_parole_estimee_ts: float = 0.0
        self._erreur_connexion_loggee: bool = False
        self._dernier_lancement_daemon_ts: float = 0.0

        if self.auto_start:
            threading.Thread(target=self._assurer_daemon_actif, daemon=True).start()

    def _trouver_python_et_script_daemon(self) -> tuple[str | None, Path | None]:
        script_cfg = self.cfg.get("tts_daemon_script")
        candidats_script = [
            Path(script_cfg).expanduser() if script_cfg else None,
            VOICE_LAB_DIR / "tts_daemon.py",
            BASE_DIR / "tts_daemon.py",
        ]
        script = next((p for p in candidats_script if p is not None and p.is_file()), None)
        if script is None:
            return None, None

        py_cfg = self.cfg.get("tts_python_bin")
        candidats_py = [
            Path(py_cfg).expanduser() if py_cfg else None,
            VOICE_LAB_DIR / ".venv" / "bin" / "python",
            VOICE_LAB_DIR / "venv" / "bin" / "python",
            VOICE_LAB_DIR / "env" / "bin" / "python",
            BASE_DIR / ".venv" / "bin" / "python",
        ]
        py_bin = next((str(p) for p in candidats_py if p is not None and p.is_file()), None)
        if py_bin is None:
            py_bin = shutil.which("python3")

        return py_bin, script

    def _demarrer_daemon_arriere_plan(self) -> bool:
        maintenant = time.monotonic()
        if (maintenant - self._dernier_lancement_daemon_ts) < 15.0:
            return False
        self._dernier_lancement_daemon_ts = maintenant

        py_bin, script = self._trouver_python_et_script_daemon()
        if not py_bin or not script:
            return False

        log_dir = BASE_DIR / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "tts_daemon.log"

        device_tts = str(self.cfg.get("tts_device", "auto"))
        cmd = [py_bin, str(script), "--socket", self.socket_path, "--device", device_tts]

        try:
            with log_file.open("a", encoding="utf-8") as f_log:
                subprocess.Popen(
                    cmd,
                    cwd=str(script.parent),
                    stdin=subprocess.DEVNULL,
                    stdout=f_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            panels.log_audio(f"[TTS] Démarrage automatique de Chatterbox ({script.name}, device={device_tts})...")
            return True
        except Exception as err:
            panels.log_audio(f"[TTS] Échec lancement auto du démon : {err}")
            return False

    def _assurer_daemon_actif(self) -> bool:
        """Vérifie la socket, purge si orpheline et lance le démon si nécessaire."""
        sock = Path(self.socket_path)
        if sock.exists():
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as test_cli:
                    test_cli.settimeout(0.4)
                    test_cli.connect(self.socket_path)
                    test_cli.sendall(b'{"cmd": "ping"}\n')
                    _ = test_cli.recv(256)
                return True
            except OSError:
                # Socket orpheline laissée après un arrêt précédent
                sock.unlink(missing_ok=True)

        if self.auto_start:
            self._demarrer_daemon_arriere_plan()
        return False

    def daemon_disponible(self) -> bool:
        return Path(self.socket_path).exists()

    def est_en_train_de_parler(self) -> bool:
        """Indique à perception.py si les haut-parleurs diffusent encore la voix d'Eon."""
        with self._lock:
            return time.monotonic() < self._fin_parole_estimee_ts

    def _enregistrer_duree_parole(self, texte_propre: str, reinitialiser: bool = False) -> None:
        duree_estimee = max(1.3, len(texte_propre) / 14.5)
        maintenant = time.monotonic()
        with self._lock:
            if reinitialiser or self._fin_parole_estimee_ts < maintenant:
                self._fin_parole_estimee_ts = maintenant + duree_estimee + 0.45
            else:
                self._fin_parole_estimee_ts += duree_estimee + 0.15

    def _envoyer_commande(self, payload: dict) -> bool:
        if not self.daemon_disponible():
            if self.auto_start:
                self._demarrer_daemon_arriere_plan()
            if not self._erreur_connexion_loggee:
                panels.log_audio(f"[TTS] Démon vocal en cours d'initialisation ({self.socket_path})...")
                self._erreur_connexion_loggee = True
            return False

        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as cli:
                cli.settimeout(0.8)
                cli.connect(self.socket_path)
                cli.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
                _ = cli.recv(1024)
            self._erreur_connexion_loggee = False
            return True
        except ConnectionRefusedError:
            Path(self.socket_path).unlink(missing_ok=True)
            if self.auto_start:
                self._demarrer_daemon_arriere_plan()
            if not self._erreur_connexion_loggee:
                panels.log_audio("[TTS] Socket orpheline nettoyée, relance de Chatterbox...")
                self._erreur_connexion_loggee = True
            return False
        except OSError as err:
            if not self._erreur_connexion_loggee:
                panels.log_audio(f"[TTS] Connexion démon impossible : {err}")
                self._erreur_connexion_loggee = True
            return False

    def demarrer_tour_reponse(self, source_message: str) -> None:
        with self._lock:
            self._tampon_stream = ""
            self._phrases_emises_tour = 0
            self._caracteres_emis_tour = 0
            self._plafond_atteint = False
            self._mode_vocal_actif_tour = (source_message == "voix")

    def arreter(self) -> None:
        with self._lock:
            self._tampon_stream = ""
            self._plafond_atteint = True
            self._fin_parole_estimee_ts = 0.0
        threading.Thread(target=self._envoyer_commande, args=({"cmd": "stop"},), daemon=True).start()

    def alerter(self, texte: str) -> None:
        propre = nettoyer_texte_pour_voix(texte)
        if not propre:
            return
        self._enregistrer_duree_parole(propre, reinitialiser=True)
        threading.Thread(
            target=self._envoyer_commande,
            args=({"cmd": "speak", "text": propre, "interrupt": True},),
            daemon=True,
        ).start()

    def parler_texte_complet(self, texte: str, source_message: str = "voix", prioritaire: bool = False) -> None:
        if source_message != "voix":
            return
        propre = nettoyer_texte_pour_voix(texte)
        if not propre:
            return

        morceaux = re.split(r"(?<=[.!?;])\s+", propre)
        selection: list[str] = []
        total_chars = 0
        tronque = False

        for m in morceaux:
            m = m.strip()
            if not m:
                continue
            if len(selection) >= self.max_phrases_orales or (total_chars + len(m)) > self.max_caracteres_oraux:
                tronque = True
                break
            selection.append(m)
            total_chars += len(m)

        if tronque:
            selection.append("Tu as tout le détail écrit dans le terminal.")

        final = " ".join(selection).strip()
        if not final:
            return

        self._enregistrer_duree_parole(final, reinitialiser=prioritaire)
        threading.Thread(
            target=self._envoyer_commande,
            args=({"cmd": "speak", "text": final, "interrupt": prioritaire},),
            daemon=True,
        ).start()

    def pousser_token_stream(self, morceau: str) -> None:
        if not morceau:
            return

        phrases_a_envoyer: list[str] = []
        with self._lock:
            if not self._mode_vocal_actif_tour or self._plafond_atteint:
                return

            self._tampon_stream += morceau
            while True:
                match = re.search(r"([.!?\n]+)\s+", self._tampon_stream)
                if not match:
                    break
                fin = match.end()
                brut = self._tampon_stream[:fin].strip()
                self._tampon_stream = self._tampon_stream[fin:]

                propre = nettoyer_texte_pour_voix(brut)
                if len(propre) < 2:
                    continue

                if (
                    self._phrases_emises_tour >= self.max_phrases_orales
                    or (self._caracteres_emis_tour + len(propre)) > self.max_caracteres_oraux
                ):
                    self._plafond_atteint = True
                    self._tampon_stream = ""
                    phrases_a_envoyer.append("Tu as tout le détail écrit dans le terminal.")
                    break

                phrases_a_envoyer.append(propre)
                self._phrases_emises_tour += 1
                self._caracteres_emis_tour += len(propre)

        for phrase in phrases_a_envoyer:
            self._enregistrer_duree_parole(phrase, reinitialiser=False)
            threading.Thread(
                target=self._envoyer_commande,
                args=({"cmd": "speak", "text": phrase, "interrupt": False},),
                daemon=True,
            ).start()

    def vider_stream(self) -> None:
        phrase_finale: str | None = None
        with self._lock:
            if not self._mode_vocal_actif_tour or self._plafond_atteint:
                self._tampon_stream = ""
                return
            reste = nettoyer_texte_pour_voix(self._tampon_stream)
            self._tampon_stream = ""
            if len(reste) >= 2:
                if (
                    self._phrases_emises_tour >= self.max_phrases_orales
                    or (self._caracteres_emis_tour + len(reste)) > self.max_caracteres_oraux
                ):
                    phrase_finale = "Tu as tout le détail écrit dans le terminal."
                else:
                    phrase_finale = reste
                self._plafond_atteint = True

        if phrase_finale:
            self._enregistrer_duree_parole(phrase_finale, reinitialiser=False)
            threading.Thread(
                target=self._envoyer_commande,
                args=({"cmd": "speak", "text": phrase_finale, "interrupt": False},),
                daemon=True,
            ).start()