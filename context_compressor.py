import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
import ollama
from memory_manager import BASE_DIR

ARCHIVE_DISQUE_FILE = BASE_DIR / "memoire_session_disque.jsonl"


def _norm_l2(vec: list[float]) -> list[float]:
    n = math.hypot(*vec)
    return [x / n for x in vec] if n > 0.0 else vec


def _cosinus(v1: list[float], v2: list[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    return sum(a * b for a, b in zip(v1, v2))


@dataclass
class BlocMemoire:
    resume: str
    vecteur: list[float]
    horodatage: float
    modele_utilise: str


class GestionnaireContexteMultiNiveaux:
    """
    Gère le contexte sur 3 niveaux matériels :
    - VRAM (8 Go)  : Intouchée (réservée à 100% au LLM principal).
    - RAM  (16 Go) : qwen3:0.6b + bge-m3 (num_gpu=0) + état compressé actif (2 phrases) + fenêtre glissante (4 tours).
    - Disque (SSD) : Archive vectorisée des sujets précédents pour rappel profond à la demande.
    """

    def __init__(
        self,
        modele_compresseur: str = "qwen3:0.6b",
        embed_model: str = "bge-m3",
    ) -> None:
        self.modele_compresseur = modele_compresseur
        self.embed_model = embed_model

        # Niveau 2 : RAM (état chaud)
        self._lock = threading.Lock()
        self.resume_actif: str = ""
        self.vecteur_actif: list[float] = []
        self.fenetre_ram: deque[dict[str, str]] = deque(maxlen=4)

        # Niveau 3 : Index léger en RAM pointant vers les résumés archivés sur Disque
        self.index_disque: list[BlocMemoire] = []
        self._charger_index_disque()

    def _embed_cpu(self, texte: str) -> list[float]:
        raw = ollama.embed(
            model=self.embed_model,
            input=texte,
            options={"num_gpu": 0},
            keep_alive=-1,
        )["embeddings"][0]
        return _norm_l2(raw)

    def _charger_index_disque(self) -> None:
        if not ARCHIVE_DISQUE_FILE.exists():
            return
        try:
            for ligne in ARCHIVE_DISQUE_FILE.read_text(encoding="utf-8").splitlines()[-150:]:
                if not ligne.strip():
                    continue
                d = json.loads(ligne)
                self.index_disque.append(
                    BlocMemoire(
                        resume=d["resume"],
                        vecteur=d["vecteur"],
                        horodatage=d.get("ts", 0.0),
                        modele_utilise=d.get("modele", ""),
                    )
                )
        except Exception:
            pass

    def _archiver_sur_disque(self, bloc: BlocMemoire) -> None:
        self.index_disque.append(bloc)
        if len(self.index_disque) > 200:
            self.index_disque.pop(0)
        try:
            with ARCHIVE_DISQUE_FILE.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "ts": bloc.horodatage,
                            "modele": bloc.modele_utilise,
                            "resume": bloc.resume,
                            "vecteur": bloc.vecteur,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except OSError:
            pass

    def _rechercher_rappel_disque(self, vec_requete: list[float], seuil: float = 0.62) -> BlocMemoire | None:
        """Cherche dans l'archive SSD si l'utilisateur reprend un ancien sujet vu beaucoup plus haut."""
        meilleur_bloc: BlocMemoire | None = None
        meilleur_score = seuil
        for bloc in self.index_disque:
            sim = _cosinus(vec_requete, bloc.vecteur)
            if sim > meilleur_score:
                meilleur_score = sim
                meilleur_bloc = bloc
        return meilleur_bloc

    def preparer_contexte_routage(self, message_utilisateur: str) -> tuple[str, str, list[dict[str, str]]]:
        """
        Appelé AVANT le routeur et le LLM.
        Retourne :
        1. `texte_pour_routeur` : message seul si rupture de sujet, sinon [État 2 phrases] + message.
        2. `etat_mode` : "continuité" | "rupture_sujet" | "rappel_disque" | "nouveau".
        3. `historique_ram` : messages à injecter au LLM principal.
        """
        msg = message_utilisateur.strip()
        with self._lock:
            resume_courant = self.resume_actif
            vec_courant = list(self.vecteur_actif)
            fenetre_courante = list(self.fenetre_ram)

        if not resume_courant or not vec_courant:
            return msg, "nouveau", fenetre_courante

        vec_msg = self._embed_cpu(msg)
        sim_active = _cosinus(vec_msg, vec_courant)

        # 1. L'utilisateur continue sur le sujet en cours (même avec une phrase courte type "fais-le")
        if sim_active >= 0.42 or len(msg.split()) <= 3:
            texte_routeur = f"Contexte en cours : {resume_courant}\nNouvelle demande : {msg}"
            return texte_routeur, "continuité", fenetre_courante

        # 2. Rupture de sujet détectée : vérifie d'abord si l'utilisateur reprend un sujet ancien sur Disque
        bloc_ancien = self._rechercher_rappel_disque(vec_msg, seuil=0.62)
        if bloc_ancien is not None:
            with self._lock:
                self.resume_actif = bloc_ancien.resume
                self.vecteur_actif = bloc_ancien.vecteur
            texte_routeur = f"Sujet repris (archive) : {bloc_ancien.resume}\nNouvelle demande : {msg}"
            historique_rappele = [
                {"role": "system", "content": f"[RAPPEL D'UN SUJET PRÉCÉDENT DE LA SESSION] : {bloc_ancien.resume}"}
            ]
            return texte_routeur, "rappel_disque", historique_rappele

        # 3. Sujet totalement neuf : on purge l'état actif en RAM pour ne pas biaiser le routeur
        with self._lock:
            self.resume_actif = ""
            self.vecteur_actif = []
            self.fenetre_ram.clear()

        return msg, "rupture_sujet", []

    def mettre_a_jour_asynchrone(self, prompt_user: str, reponse_ia: str, modele_utilise: str) -> None:
        """
        Compresse le tour en arrière-plan (CPU uniquement) après l'affichage de la réponse.
        Coût en latence pour le tour suivant : 0 ms.
        """
        threading.Thread(
            target=self._worker_compression,
            args=(prompt_user, reponse_ia, modele_utilise),
            daemon=True,
        ).start()

    def _worker_compression(self, prompt_user: str, reponse_ia: str, modele_utilise: str) -> None:
        if not reponse_ia or len(reponse_ia.strip()) < 4:
            return

        with self._lock:
            ancien_resume = self.resume_actif
            ancien_vec = list(self.vecteur_actif)
            self.fenetre_ram.append({"role": "user", "content": prompt_user})
            self.fenetre_ram.append({"role": "assistant", "content": reponse_ia[:1200]})

        prompt_compression = (
            "Compresse l'état actuel de la conversation en 1 ou 2 phrases courtes et purement factuelles "
            "(sujet précis, entités, fichier/commande en cours, action attendue). Aucun blabla.\n"
            f"État précédent : {ancien_resume or 'Aucun'}\n"
            f"Dernier message utilisateur : {prompt_user[:400]}\n"
            f"Dernière réponse assistant : {reponse_ia[:600]}"
        )

        kwargs = {
            "model": self.modele_compresseur,
            "messages": [{"role": "user", "content": prompt_compression}],
            "options": {"num_gpu": 0, "temperature": 0.0, "num_predict": 90, "num_ctx": 2048},
            "keep_alive": -1,
        }
        try:
            try:
                rep = ollama.chat(**kwargs, think=False)
            except TypeError:
                rep = ollama.chat(**kwargs)

            nouveau_resume = rep.get("message", {}).get("content", "").strip()
            # Nettoyage au cas où qwen3 émet une balise <think> vide
            if "</think>" in nouveau_resume:
                nouveau_resume = nouveau_resume.split("</think>", 1)[1].strip()

            if not nouveau_resume:
                return

            nouveau_vec = self._embed_cpu(nouveau_resume)

            # Si le sujet a dérivé significativement par rapport à l'ancien résumé, on archive l'ancien sur Disque
            if ancien_resume and ancien_vec and _cosinus(ancien_vec, nouveau_vec) < 0.55:
                self._archiver_sur_disque(
                    BlocMemoire(
                        resume=ancien_resume,
                        vecteur=ancien_vec,
                        horodatage=time.time(),
                        modele_utilise=modele_utilise,
                    )
                )

            with self._lock:
                self.resume_actif = nouveau_resume
                self.vecteur_actif = nouveau_vec
        except Exception:
            pass