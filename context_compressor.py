import json
import math
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass

import ollama
from memory_manager import BASE_DIR

ARCHIVE_DISQUE_FILE = BASE_DIR / "memoire_session_disque.jsonl"
MAX_BLOCS_DISQUE = 250

REGEX_SECRET_CTX = re.compile(
    r"(sudo\s+-S|mot\s+de\s+passe|password|passwd|api[_-]?key|token\s*=|secret\s*=|-----BEGIN\s+[A-Z ]+PRIVATE\s+KEY-----)",
    re.IGNORECASE,
)


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
    - VRAM (8 Go)  : Intouchée (réservée à 100 % au LLM principal).
    - RAM  (16 Go) : qwen3:0.6b + bge-m3 (num_gpu=0) + état compressé actif + fenêtre glissante (4 tours).
    - Disque (SSD) : Archive vectorisée bornée des sujets précédents pour rappel profond à la demande.
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
        self._worker_lock = threading.Lock()
        self._generation_id: int = 0
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
            lignes = ARCHIVE_DISQUE_FILE.read_text(encoding="utf-8").splitlines()
            for ligne in lignes[-MAX_BLOCS_DISQUE:]:
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
            # Compacte le fichier sur disque s'il a dépassé la limite
            if len(lignes) > MAX_BLOCS_DISQUE + 40:
                self._persister_index_disque_atomique()
        except Exception:
            pass

    def _persister_index_disque_atomique(self) -> None:
        try:
            tmp = Path(f"{ARCHIVE_DISQUE_FILE}.tmp")
            contenu = "\n".join(
                json.dumps(
                    {
                        "ts": b.horodatage,
                        "modele": b.modele_utilise,
                        "resume": b.resume,
                        "vecteur": b.vecteur,
                    },
                    ensure_ascii=False,
                )
                for b in self.index_disque[-MAX_BLOCS_DISQUE:]
            )
            if contenu:
                contenu += "\n"
            tmp.write_text(contenu, encoding="utf-8")
            os.replace(tmp, ARCHIVE_DISQUE_FILE)
        except OSError:
            pass

    def _archiver_sur_disque(self, bloc: BlocMemoire) -> None:
        if REGEX_SECRET_CTX.search(bloc.resume):
            return
        self.index_disque.append(bloc)
        if len(self.index_disque) > MAX_BLOCS_DISQUE:
            self.index_disque = self.index_disque[-MAX_BLOCS_DISQUE:]
            self._persister_index_disque_atomique()
            return
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

    def forcer_flush_disque(self) -> None:
        """Archive l'état chaud courant sur le SSD (utile lors d'un arrêt propre)."""
        with self._lock:
            res = self.resume_actif
            vec = list(self.vecteur_actif)
        if res and vec:
            self._archiver_sur_disque(
                BlocMemoire(resume=res, vecteur=vec, horodatage=time.time(), modele_utilise="flush")
            )

    def _rechercher_rappels_disque(self, vec_requete: list[float], seuil: float = 0.62, max_blocs: int = 2) -> list[BlocMemoire]:
        """Cherche dans l'archive SSD jusqu'à 2 sujets distincts pertinents vus plus tôt."""
        candidats: list[tuple[float, BlocMemoire]] = []
        for bloc in self.index_disque:
            sim = _cosinus(vec_requete, bloc.vecteur)
            if sim >= seuil:
                candidats.append((sim, bloc))

        candidats.sort(key=lambda item: item[0], reverse=True)
        selection: list[BlocMemoire] = []
        for _, bloc in candidats:
            if all(_cosinus(bloc.vecteur, deja.vecteur) < 0.88 for deja in selection):
                selection.append(bloc)
            if len(selection) >= max_blocs:
                break
        return selection

    def preparer_contexte_routage(self, message_utilisateur: str) -> tuple[str, str, list[dict[str, str]]]:
        """
        Appelé AVANT le routeur et le LLM.
        Retourne :
        1. `texte_pour_routeur` : message seul si rupture de sujet, sinon [État] + message.
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

        # 1. Continuité sur le sujet actif
        if sim_active >= 0.42 or len(msg.split()) <= 3:
            texte_routeur = f"Contexte en cours : {resume_courant}\nNouvelle demande : {msg}"
            return texte_routeur, "continuité", fenetre_courante

        # 2. Rupture : vérifie si l'utilisateur reprend un ou plusieurs sujets archivés sur SSD
        blocs_anciens = self._rechercher_rappels_disque(vec_msg, seuil=0.62, max_blocs=2)
        if blocs_anciens:
            principal = blocs_anciens[0]
            synthese_rappel = " | ".join(b.resume for b in blocs_anciens)
            with self._lock:
                self._generation_id += 1
                self.resume_actif = principal.resume
                self.vecteur_actif = principal.vecteur
            texte_routeur = f"Sujet repris (archive) : {synthese_rappel}\nNouvelle demande : {msg}"
            historique_rappele = [
                {"role": "system", "content": f"[RAPPEL DE SUJET(S) PRÉCÉDENT(S) DE LA SESSION] : {synthese_rappel}"}
            ]
            return texte_routeur, "rappel_disque", historique_rappele

        # 3. Sujet totalement neuf : archive l'ancien sujet chaud puis réinitialise la RAM
        self._archiver_sur_disque(
            BlocMemoire(
                resume=resume_courant,
                vecteur=vec_courant,
                horodatage=time.time(),
                modele_utilise="rupture",
            )
        )
        with self._lock:
            self._generation_id += 1
            self.resume_actif = ""
            self.vecteur_actif = []
            self.fenetre_ram.clear()

        return msg, "rupture_sujet", []

    def mettre_a_jour_asynchrone(self, prompt_user: str, reponse_ia: str, modele_utilise: str) -> None:
        """
        Compresse le tour en arrière-plan (CPU uniquement) avec verrouillage séquentiel
        pour garantir qu'un tour plus ancien n'écrase jamais un tour plus récent.
        """
        with self._lock:
            self._generation_id += 1
            gen_courante = self._generation_id
            self.fenetre_ram.append({"role": "user", "content": prompt_user})
            self.fenetre_ram.append({"role": "assistant", "content": reponse_ia[:1200]})

        threading.Thread(
            target=self._worker_compression,
            args=(prompt_user, reponse_ia, modele_utilise, gen_courante),
            daemon=True,
        ).start()

    def _worker_compression(
        self,
        prompt_user: str,
        reponse_ia: str,
        modele_utilise: str,
        generation_attendue: int,
    ) -> None:
        if not reponse_ia or len(reponse_ia.strip()) < 4:
            return
        if REGEX_SECRET_CTX.search(prompt_user):
            return

        with self._worker_lock:
            with self._lock:
                if generation_attendue != self._generation_id:
                    return
                ancien_resume = self.resume_actif
                ancien_vec = list(self.vecteur_actif)

            prompt_compression = (
                "Compresse l'état actuel de la conversation en 1 ou 2 phrases courtes et purement factuelles "
                "(sujet précis, entités, fichier/commande en cours, action attendue). Aucun blabla ni secret.\n"
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
                if "</think>" in nouveau_resume:
                    nouveau_resume = nouveau_resume.split("</think>", 1)[1].strip()

                if not nouveau_resume or REGEX_SECRET_CTX.search(nouveau_resume):
                    return

                nouveau_vec = self._embed_cpu(nouveau_resume)

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
                    if generation_attendue == self._generation_id:
                        self.resume_actif = nouveau_resume
                        self.vecteur_actif = nouveau_vec
            except Exception:
                pass