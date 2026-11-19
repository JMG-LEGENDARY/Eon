import contextvars
import getpass
import queue
import sys
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.widgets import Footer, Header, Input, RichLog

    TEXTUAL_DISPONIBLE = True
except ImportError:
    TEXTUAL_DISPONIBLE = False

_CANAL_ACTIF: contextvars.ContextVar[str] = contextvars.ContextVar("eon_terminal_channel", default="chat")


class _GestionnaireConsoleFallback:
    """Fallback propre si Textual n'est pas installé ou hors TTY."""

    def __init__(self, dossier_logs: Path) -> None:
        self.dossier_logs = dossier_logs
        self.dossier_logs.mkdir(parents=True, exist_ok=True)
        self.chemins = {
            "logs": dossier_logs / "perception_eon.log",
            "outils": dossier_logs / "outils_eon.log",
        }
        self._lock = threading.Lock()
        self._file_prompts: queue.Queue[str] = queue.Queue()

    def ecrire_chat(self, texte: str, fin: str = "\n") -> None:
        with self._lock:
            sys.stdout.write(texte + fin)
            sys.stdout.flush()

    def journaliser(self, canal: str, message: str) -> None:
        horodatage = datetime.now().strftime("%H:%M:%S")
        lignes = str(message).splitlines() or [""]
        bloc = "".join(f"[{horodatage}] {l}\n" for l in lignes)
        chemin = self.chemins.get(canal)
        with self._lock:
            if chemin is not None:
                with chemin.open("a", encoding="utf-8") as f:
                    f.write(bloc)
            if canal == "outils":
                sys.stdout.write(f"\033[33m[OUTIL]\033[0m {lignes[0]}\n")
                sys.stdout.flush()

    def demander_ligne(self, invite: str, secret: bool = False) -> str:
        if secret:
            return getpass.getpass(invite)
        return input(invite)


if TEXTUAL_DISPONIBLE:

    class EonTUIApp(App):
        TITLE = "EON · Assistant Local Autonome"
        CSS = """
        Screen {
            background: #0d1117;
        }
        #corps {
            height: 1fr;
        }
        #panneau_chat {
            width: 62%;
            border: round #38bdf8;
            background: #0d1117;
            padding: 0 1;
        }
        #colonne_droite {
            width: 38%;
        }
        #panneau_outils {
            height: 1fr;
            border: round #facc15;
            background: #0d1117;
            padding: 0 1;
        }
        #panneau_perception {
            height: 1fr;
            border: round #4ade80;
            background: #0d1117;
            padding: 0 1;
        }
        #saisie_eon {
            dock: bottom;
            border: round #94a3b8;
            background: #161b22;
        }
        #saisie_eon.confirmation {
            border: round #f97316;
        }
        """

        BINDINGS = [
            Binding("escape", "interrompre_ia", "Interrompre Eon", show=True),
            Binding("ctrl+l", "effacer_vue_chat", "Nettoyer vue", show=True),
        ]

        def __init__(
            self,
            dossier_logs: Path,
            worker_fn: Callable[[], None],
            on_prompt_fn: Callable[[str], None],
            on_escape_fn: Callable[[], None],
        ) -> None:
            super().__init__()
            self.dossier_logs = dossier_logs
            self.dossier_logs.mkdir(parents=True, exist_ok=True)
            self.chemins = {
                "logs": dossier_logs / "perception_eon.log",
                "outils": dossier_logs / "outils_eon.log",
            }
            self._worker_fn = worker_fn
            self._on_prompt_fn = on_prompt_fn
            self._on_escape_fn = on_escape_fn
            self._lock_fichiers = threading.Lock()
            self._tampon_chat = ""

            # File d'attente pour les demandes de confirmation synchrones (shell/fichiers/sudo)
            self._attente_reponse: queue.Queue[str] | None = None
            self._lock_confirmation = threading.Lock()

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Horizontal(id="corps"):
                yield RichLog(id="panneau_chat", wrap=True, highlight=False, markup=False)
                with Vertical(id="colonne_droite"):
                    yield RichLog(id="panneau_outils", wrap=True, highlight=False, markup=False)
                    yield RichLog(id="panneau_perception", wrap=True, highlight=False, markup=False)
            yield Input(placeholder="Écris à Eon ou parle en regardant l'écran (/help)...", id="saisie_eon")
            yield Footer()

        def on_mount(self) -> None:
            self.query_one("#panneau_chat", RichLog).border_title = "Conversation"
            self.query_one("#panneau_outils", RichLog).border_title = "Outils · Terminal / Fichiers / Web"
            self.query_one("#panneau_perception", RichLog).border_title = "Perception · Caméra / Sécurité / Micro"
            self.query_one("#saisie_eon", Input).focus()
            threading.Thread(target=self._executer_worker, daemon=True).start()

        def _executer_worker(self) -> None:
            try:
                self._worker_fn()
            finally:
                self.call_from_thread(self.exit)

        def action_interrompre_ia(self) -> None:
            self._on_escape_fn()

        def action_effacer_vue_chat(self) -> None:
            self.query_one("#panneau_chat", RichLog).clear()

        def _ecrire_chat_ui(self, texte: str) -> None:
            log_chat = self.query_one("#panneau_chat", RichLog)
            self._tampon_chat += texte
            while "\n" in self._tampon_chat:
                ligne, self._tampon_chat = self._tampon_chat.split("\n", 1)
                log_chat.write(ligne)

        def flush_chat_partiel(self) -> None:
            def _flush() -> None:
                if self._tampon_chat:
                    self.query_one("#panneau_chat", RichLog).write(self._tampon_chat)
                    self._tampon_chat = ""

            self.call_from_thread(_flush)

        def ecrire_chat(self, texte: str, fin: str = "\n") -> None:
            complet = f"{texte}{fin}"
            if not complet:
                return
            self.call_from_thread(self._ecrire_chat_ui, complet)

        def journaliser(self, canal: str, message: str) -> None:
            horodatage = datetime.now().strftime("%H:%M:%S")
            lignes = str(message).splitlines() or [""]
            bloc_fichier = "".join(f"[{horodatage}] {l}\n" for l in lignes)

            chemin = self.chemins.get(canal)
            if chemin is not None:
                with self._lock_fichiers, chemin.open("a", encoding="utf-8") as f:
                    f.write(bloc_fichier)

            cible_id = "#panneau_outils" if canal == "outils" else "#panneau_perception"

            def _maj_panneau() -> None:
                widget = self.query_one(cible_id, RichLog)
                for l in lignes:
                    widget.write(f"[{horodatage}] {l}")

            self.call_from_thread(_maj_panneau)

        def demander_ligne(self, invite: str, secret: bool = False) -> str:
            """Permet à un thread d'arrière-plan (shell_tools/file_tools) de demander une confirmation ou un mot de passe."""
            q: queue.Queue[str] = queue.Queue(maxsize=1)
            with self._lock_confirmation:
                self._attente_reponse = q

            def _activer_mode_question() -> None:
                inp = self.query_one("#saisie_eon", Input)
                inp.value = ""
                inp.password = secret
                inp.placeholder = invite.strip()
                inp.add_class("confirmation")
                inp.focus()

            self.call_from_thread(_activer_mode_question)
            try:
                return q.get()
            finally:
                with self._lock_confirmation:
                    self._attente_reponse = None

                def _restaurer_mode_normal() -> None:
                    inp = self.query_one("#saisie_eon", Input)
                    inp.password = False
                    inp.value = ""
                    inp.placeholder = "Écris à Eon ou parle en regardant l'écran (/help)..."
                    inp.remove_class("confirmation")

                self.call_from_thread(_restaurer_mode_normal)

        async def on_input_submitted(self, event: Input.Submitted) -> None:
            valeur = event.value
            event.input.value = ""

            with self._lock_confirmation:
                q_active = self._attente_reponse

            if q_active is not None:
                if not event.input.password:
                    self.query_one("#panneau_chat", RichLog).write(f"  > {valeur}")
                q_active.put(valeur)
                return

            texte = valeur.strip()
            if not texte:
                return
            self.query_one("#panneau_chat", RichLog).write(f"\nVous : {texte}")
            self._on_prompt_fn(texte)


_APP_TUI: "EonTUIApp | None" = None
_FALLBACK: _GestionnaireConsoleFallback | None = None
_CB_ENTREE_CLAVIER: Callable[[str], None] | None = None
_CB_INTERRUPTION: Callable[[], None] | None = None


def enregistrer_callbacks_ui(
    on_prompt: Callable[[str], None],
    on_escape: Callable[[], None],
) -> None:
    """Connecte la barre de saisie Textual et la touche Échap à PerceptionEon."""
    global _CB_ENTREE_CLAVIER, _CB_INTERRUPTION
    _CB_ENTREE_CLAVIER = on_prompt
    _CB_INTERRUPTION = on_escape


def tui_active() -> bool:
    return _APP_TUI is not None


def demarrer_panneaux(dossier_base: Path, script: str = "", arguments: list[str] | None = None) -> bool:
    (dossier_base / "logs").mkdir(parents=True, exist_ok=True)
    return False


def configurer(dossier_base: Path) -> None:
    global _FALLBACK
    if _FALLBACK is None:
        _FALLBACK = _GestionnaireConsoleFallback(dossier_base / "logs")


def lancer_avec_interface(dossier_base: Path, boucle_principale: Callable[[], None]) -> None:
    """Lance Textual si disponible dans un terminal interactif, sinon exécute directement boucle_principale."""
    global _APP_TUI
    configurer(dossier_base)

    est_tty = bool(sys.stdin and sys.stdout and sys.stdin.isatty() and sys.stdout.isatty())
    if TEXTUAL_DISPONIBLE and est_tty:
        _APP_TUI = EonTUIApp(
            dossier_logs=dossier_base / "logs",
            worker_fn=boucle_principale,
            on_prompt_fn=lambda txt: _CB_ENTREE_CLAVIER(txt) if _CB_ENTREE_CLAVIER else None,
            on_escape_fn=lambda: _CB_INTERRUPTION() if _CB_INTERRUPTION else None,
        )
        _APP_TUI.run()
        _APP_TUI = None
    else:
        boucle_principale()


def ecrire_chat(texte: str, fin: str = "\n") -> None:
    if _APP_TUI is not None:
        _APP_TUI.ecrire_chat(texte, fin=fin)
    elif _FALLBACK is not None:
        _FALLBACK.ecrire_chat(texte, fin=fin)
    else:
        print(texte, end=fin, flush=True)


def flush_chat() -> None:
    if _APP_TUI is not None:
        _APP_TUI.flush_chat_partiel()


@contextmanager
def canal(nom: str) -> Iterator[None]:
    token = _CANAL_ACTIF.set(nom)
    try:
        yield
    finally:
        _CANAL_ACTIF.reset(token)


@contextmanager
def suspendre() -> Iterator[None]:
    yield


def saisir(invite: str) -> str:
    if _APP_TUI is not None:
        return _APP_TUI.demander_ligne(invite, secret=False)
    if _FALLBACK is not None:
        return _FALLBACK.demander_ligne(invite, secret=False)
    return input(invite)


def saisir_mot_de_passe(invite: str = "[Sudo] Mot de passe administrateur : ") -> str:
    if _APP_TUI is not None:
        return _APP_TUI.demander_ligne(invite, secret=True)
    if _FALLBACK is not None:
        return _FALLBACK.demander_ligne(invite, secret=True)
    return getpass.getpass(invite)


def fermer() -> None:
    if _APP_TUI is not None:
        try:
            _APP_TUI.call_from_thread(_APP_TUI.exit)
        except Exception:
            pass


def log_perception(message: str) -> None:
    if _APP_TUI is not None:
        _APP_TUI.journaliser("logs", message)
    elif _FALLBACK is not None:
        _FALLBACK.journaliser("logs", message)


def log_outil(message: str) -> None:
    if _APP_TUI is not None:
        _APP_TUI.journaliser("outils", message)
    elif _FALLBACK is not None:
        _FALLBACK.journaliser("outils", message)