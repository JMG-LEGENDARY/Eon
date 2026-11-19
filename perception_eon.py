import json
import math
import os
import queue
import re
import select
import signal
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import ollama

import terminal_panels as panels
from memory_manager import BASE_DIR

PROFIL_EON_FILE = BASE_DIR / "profil_eon.json"
MODELS_VISION_DIR = BASE_DIR / "models_vision"

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

try:
    import termios
    import tty
except ImportError:
    termios = None
    tty = None

COCO_CLASSES_FR = {
    0: "personne",
    1: "vélo",
    2: "voiture",
    15: "chat",
    16: "chien",
    24: "sac à dos",
    26: "sac",
    28: "valise",
    39: "bouteille",
    41: "tasse",
    43: "couteau",
    63: "ordinateur portable",
    64: "souris",
    66: "clavier",
    67: "téléphone",
    73: "livre",
}

# Détection du mot-clé "Éon" (y compris variantes phonétiques fréquentes de Whisper en français)
REGEX_WAKE_WORD = re.compile(
    r"(?:\b(?:[ée]ons?|eons?|h[ée]ons?|eyon|ayon)\b|^(?:eh\s+on|et\s+on|est\s+on)\b)",
    re.IGNORECASE,
)

HALLUCINATIONS_SUBSTRINGS = (
    "sous-titres réalisés",
    "sous-titrage",
    "amara.org",
    "merci d'avoir regardé",
    "abonnez-vous",
)

BRUITS_PARASITES_WHISPER = {
    "à bientôt",
    "merci.",
    "merci",
    "...",
    "eh",
    "heu",
    "euh",
}


@dataclass
class MessageEntrant:
    texte: str
    source: str  # "clavier" | "voix"
    confiance_locuteur: float = 1.0


def _cosinus(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _iou_boites(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> float:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0
    union = (w1 * h1) + (w2 * h2) - inter
    return inter / float(max(1, union))


def _barre_vu(rms: float, echelle: float = 0.06, largeur: int = 10) -> str:
    ratio = min(1.0, max(0.0, rms / max(1e-6, echelle)))
    remplis = int(round(ratio * largeur))
    return "▮" * remplis + "▯" * (largeur - remplis)


def _reechantillonner_16k(audio: np.ndarray, sr_source: int) -> np.ndarray:
    """Rééchantillonnage vers 16 kHz avec filtre passe-bas anti-repliement (anti-aliasing)."""
    if sr_source == 16000 or len(audio) == 0:
        return audio.astype(np.float32, copy=False)

    signal_f = audio.astype(np.float32, copy=False)
    if sr_source > 16000:
        fc = 7200.0 / float(sr_source)
        nb_taps = 31
        n = np.arange(nb_taps) - (nb_taps - 1) / 2.0
        noyau = np.sinc(2.0 * fc * n) * np.hamming(nb_taps)
        noyau /= np.sum(noyau)
        signal_f = np.convolve(signal_f, noyau, mode="same").astype(np.float32)

    duree = len(signal_f) / float(sr_source)
    nb_cible = max(1, int(round(duree * 16000)))
    x_ancien = np.linspace(0.0, 1.0, num=len(signal_f), endpoint=False)
    x_nouveau = np.linspace(0.0, 1.0, num=nb_cible, endpoint=False)
    return np.interp(x_nouveau, x_ancien, signal_f).astype(np.float32)


class PerceptionEon:
    """Perception locale basse consommation (100 % licences permissives MIT / Apache-2.0 / BSD)."""

    def __init__(
        self,
        config_eon: dict | None = None,
        on_alerte_securite: Callable[[str], None] | None = None,
    ) -> None:
        self.cfg = config_eon or {}
        self.on_alerte_securite = on_alerte_securite
        self.file_messages: queue.Queue[MessageEntrant] = queue.Queue()
        self.interruption_event = threading.Event()
        self.ia_en_cours = threading.Event()
        self.pret_pour_saisie = threading.Event()
        self.pret_pour_saisie.set()
        self.arret_global = threading.Event()

        # Mode sécurité : "auto" (analyse si inconnu seul), "absent" (alerte immédiate), "off"
        self.mode_securite: str = str(self.cfg.get("mode_securite", "auto"))
        self._dernier_check_intrus_ts: float = 0.0
        self._debut_inconnu_seul_ts: float = 0.0
        self._dernier_verdict_securite: str = "RAS"

        self._lock_vision = threading.Lock()
        self._derniere_frame: np.ndarray | None = None
        self._dernier_regard_ts: float = 0.0
        self._regard_ema: float = 0.0
        self._dernier_vecteur_visage: np.ndarray | None = None
        self._dernier_resident_reconnu: str = ""
        self._dernier_resident_ts: float = 0.0
        self._visage_detecte_actuel: bool = False
        self._nb_personnes_actuel: int = 0
        self._personnes_identifiees: list[str] = []
        self._objets_detectes_actuels: list[str] = []
        self._position_visage_actuel: str = ""
        self._luminosite_actuelle: float = 0.0

        self.camera_active: bool = False
        self.statut_camera: str = "initialisation"
        self.micro_actif: bool = False
        self.statut_micro: str = "initialisation"

        # Biométrie multi-résidents (stockage exclusivement local)
        self.profils_visages: dict[str, np.ndarray] = {}
        self.profil_voix: np.ndarray | None = None
        self._charger_profil()

        # Détecteurs vision légers CPU (Apache-2.0 / MIT uniquement)
        self._yolox_net = None
        self._hog_personnes = None
        self._yunet = None
        self._sface = None
        self._initialiser_modeles_vision_cpu()

        self._whisper = None
        self._whisper_erreur: str = ""
        self._lock_whisper = threading.Lock()
        self._transcriptions_en_attente = 0
        self._logs_audio_live: queue.Queue[str] = queue.Queue(maxsize=200)
        self._installer_signal_interruption()

    # ---------------- CONTRÔLES UI & INTERRUPTION ----------------

    def _dimension_attendue_visage(self) -> int:
        return 128 if self._sface is not None else 288

    def _profils_visages_compatibles(self) -> dict[str, np.ndarray]:
        dim = self._dimension_attendue_visage()
        return {nom: vec for nom, vec in self.profils_visages.items() if vec.shape == (dim,)}

    def autoriser_saisie(self) -> None:
        self.pret_pour_saisie.set()

    def soumettre_texte_clavier(self, texte: str) -> None:
        """Appelé directement par Textual lors de la validation de l'Input."""
        propre = texte.strip()
        if not propre:
            return
        self.pret_pour_saisie.clear()
        self.file_messages.put(MessageEntrant(texte=propre, source="clavier"))

    def declencher_interruption(self) -> None:
        """Appelé par la touche Échap (Textual ou TTY) ou le barge-in vocal validé."""
        if self.ia_en_cours.is_set():
            self.interruption_event.set()

    def _installer_signal_interruption(self) -> None:
        def _handler(signum, frame):
            self.declencher_interruption()

        try:
            signal.signal(signal.SIGUSR1, _handler)
        except (ValueError, OSError):
            pass

    def _push_log_audio(self, msg: str) -> None:
        try:
            self._logs_audio_live.put_nowait(msg)
        except queue.Full:
            try:
                self._logs_audio_live.get_nowait()
                self._logs_audio_live.put_nowait(msg)
            except queue.Empty:
                pass

    def _vider_logs_audio(self) -> None:
        while True:
            try:
                msg = self._logs_audio_live.get_nowait()
                panels.log_audio(msg)
            except queue.Empty:
                break

    # ---------------- BIOMÉTRIE MULTI-PROFILS ----------------

    def _charger_profil(self) -> None:
        if not PROFIL_EON_FILE.exists():
            return
        try:
            data = json.loads(PROFIL_EON_FILE.read_text(encoding="utf-8"))
            if isinstance(data.get("visages"), dict):
                for nom, vec in data["visages"].items():
                    if vec:
                        self.profils_visages[nom] = np.array(vec, dtype=np.float32)
            elif data.get("visage"):
                self.profils_visages["Bryan"] = np.array(data["visage"], dtype=np.float32)

            if data.get("voix"):
                self.profil_voix = np.array(data["voix"], dtype=np.float32)
        except (json.JSONDecodeError, OSError, ValueError):
            pass

    def _sauvegarder_profil(self) -> None:
        data = {
            "visages": {nom: vec.tolist() for nom, vec in self.profils_visages.items()},
            "voix": self.profil_voix.tolist() if self.profil_voix is not None else None,
        }
        tmp = PROFIL_EON_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, PROFIL_EON_FILE)

    def enregistrer_visage_autorise(self, nom: str = "Bryan") -> str:
        """Enregistre le visage actuellement visible devant la caméra comme personne autorisée."""
        nom_propre = nom.strip().capitalize() or "Bryan"
        with self._lock_vision:
            vec = self._dernier_vecteur_visage.copy() if self._dernier_vecteur_visage is not None else None
        if vec is None:
            return "Aucun visage détecté face à la caméra : regarde l'objectif puis relance `/enroll`."
        self.profils_visages[nom_propre] = vec
        self._sauvegarder_profil()
        panels.log_perception(f"[BIOMÉTRIE] Visage enregistré pour : {nom_propre}")
        return f"Profil facial enregistré pour « {nom_propre} » ({len(self.profils_visages)} profil(s) autorisé(s))."

    def reinitialiser_profil_biometrique(self, nom: str | None = None) -> str:
        if nom:
            cle = nom.strip().capitalize()
            if cle in self.profils_visages:
                del self.profils_visages[cle]
                self._sauvegarder_profil()
                return f"Profil facial « {cle} » supprimé."
            return f"Profil « {cle} » introuvable."
        self.profils_visages.clear()
        self.profil_voix = None
        PROFIL_EON_FILE.unlink(missing_ok=True)
        return "Tous les profils biométriques ont été réinitialisés."

    # ---------------- INITIALISATION VISION LÉGÈRE CPU (APACHE-2.0 / MIT) ----------------

    def _initialiser_modeles_vision_cpu(self) -> None:
        if cv2 is None:
            return
        MODELS_VISION_DIR.mkdir(parents=True, exist_ok=True)

        # Modèles YOLOX sous licence Apache-2.0 (aucun modèle AGPL)
        for nom_yolox in ("object_detection_yolox_2022nov.onnx", "yolox_nano.onnx", "yolox_s.onnx"):
            chemin_yolox = MODELS_VISION_DIR / nom_yolox
            if chemin_yolox.exists():
                try:
                    net = cv2.dnn.readNetFromONNX(str(chemin_yolox))
                    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    self._yolox_net = net
                    break
                except Exception:
                    pass

        if self._yolox_net is None and hasattr(cv2, "HOGDescriptor"):
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self._hog_personnes = hog

        # YuNet (MIT) + SFace (Apache-2.0) issus d'OpenCV Model Zoo
        yunet_path = MODELS_VISION_DIR / "face_detection_yunet_2023mar.onnx"
        sface_path = MODELS_VISION_DIR / "face_recognition_sface_2021dec.onnx"
        if yunet_path.exists() and hasattr(cv2, "FaceDetectorYN"):
            try:
                seuil_yunet = float(self.cfg.get("seuil_detection_visage", 0.85))
                self._yunet = cv2.FaceDetectorYN.create(str(yunet_path), "", (320, 320), seuil_yunet, 0.3, 50)
            except Exception:
                self._yunet = None
        if sface_path.exists() and hasattr(cv2, "FaceRecognizerSF"):
            try:
                self._sface = cv2.FaceRecognizerSF.create(str(sface_path), "")
            except Exception:
                self._sface = None

    def _detecter_objets_rapide(self, frame: np.ndarray) -> tuple[int, list[str]]:
        """Détection d'objets Apache-2.0 (YOLOX ONNX ou OpenCV HOG) en ~15-25 ms sur CPU."""
        h_img, w_img = frame.shape[:2]

        if self._yolox_net is not None:
            blob = cv2.dnn.blobFromImage(frame, 1.0, (640, 640), swapRB=True, crop=False)
            self._yolox_net.setInput(blob)
            out = self._yolox_net.forward()
            preds = out[0] if out.ndim == 3 else out
            if preds.shape[0] == 85 and preds.shape[1] != 85:
                preds = preds.T

            obj_conf = preds[:, 4]
            class_scores = preds[:, 5:]
            class_ids = np.argmax(class_scores, axis=1)
            scores = obj_conf * class_scores[np.arange(len(preds)), class_ids]

            masque = scores >= 0.45
            preds_valides = preds[masque]
            scores_valides = scores[masque]
            class_ids_valides = class_ids[masque]
            if len(preds_valides) == 0:
                return 0, []

            boites = []
            for cx, cy, w, h in preds_valides[:, :4]:
                x1 = int((cx - w / 2.0) * w_img / 640.0)
                y1 = int((cy - h / 2.0) * h_img / 640.0)
                boites.append([x1, y1, int(w * w_img / 640.0), int(h * h_img / 640.0)])

            indices = cv2.dnn.NMSBoxes(boites, scores_valides.tolist(), 0.45, 0.45)
            nb_pers = 0
            compteur_objets: dict[str, int] = {}
            if len(indices) > 0:
                for idx in np.array(indices).flatten():
                    cid = int(class_ids_valides[idx])
                    if cid == 0:
                        nb_pers += 1
                    elif cid in COCO_CLASSES_FR:
                        nom_obj = COCO_CLASSES_FR[cid]
                        compteur_objets[nom_obj] = compteur_objets.get(nom_obj, 0) + 1

            objets = [f"{k} (x{v})" if v > 1 else k for k, v in compteur_objets.items()]
            return nb_pers, objets

        if self._hog_personnes is not None:
            petite = cv2.resize(frame, (320, 240))
            _, weights = self._hog_personnes.detectMultiScale(petite, winStride=(8, 8), padding=(8, 8), scale=1.08)
            nb_pers = sum(1 for w in weights if float(w) > 0.70) if len(weights) > 0 else 0
            return nb_pers, []

        return 0, []

    @staticmethod
    def _extraire_signature_visage_fallback(gris_roi: np.ndarray) -> np.ndarray:
        redim = cv2.resize(gris_roi, (64, 64))
        egalise = cv2.equalizeHist(redim)
        grille = cv2.resize(egalise, (16, 16)).astype(np.float32).flatten()
        hist = cv2.calcHist([egalise], [0], None, [32], [0, 256]).flatten().astype(np.float32)
        vec = np.concatenate([grille / (np.linalg.norm(grille) + 1e-6), hist / (np.linalg.norm(hist) + 1e-6)])
        norme = float(np.linalg.norm(vec))
        return vec / norme if norme > 0 else vec

    def _identifier_visage(
        self,
        vec: np.ndarray,
        identites_deja_attribuees: set[str] | None = None,
    ) -> tuple[str, float]:
        """Compare un vecteur facial aux résidents enregistrés sans autoriser de clones sur la même frame."""
        profils_valides = self._profils_visages_compatibles()
        if not profils_valides:
            return "Non calibré", 0.0

        deja_pris = identites_deja_attribuees or set()
        meilleur_nom = "Inconnu"
        meilleur_score = -1.0
        seuil = float(self.cfg.get("seuil_visage_sface", 0.45)) if self._sface is not None else 0.78

        for nom, ref in profils_valides.items():
            if nom in deja_pris or ref.shape != vec.shape:
                continue
            if self._sface is not None:
                score = float(self._sface.match(ref.reshape(1, -1), vec.reshape(1, -1), cv2.FaceRecognizerSF_FR_COSINE))
            else:
                score = _cosinus(ref, vec)
            if score > meilleur_score:
                meilleur_score = score
                meilleur_nom = nom

        if meilleur_score >= seuil:
            if meilleur_score >= 0.62:
                nouveau_ref = 0.94 * self.profils_visages[meilleur_nom] + 0.06 * vec
                norme = float(np.linalg.norm(nouveau_ref))
                if norme > 0:
                    self.profils_visages[meilleur_nom] = nouveau_ref / norme
            return meilleur_nom, round(meilleur_score, 2)
        return "Inconnu", round(max(0.0, meilleur_score), 2)

    @staticmethod
    def _score_regard_frontal(face_row: np.ndarray) -> float:
        """
        Calcule un score [0.0..1.0] d'attention visuelle vers la caméra/écran dans un repère
        invariant à l'inclinaison de la tête (roll) grâce aux 5 points clés YuNet (yeux, nez, bouche).
        """
        re_x, re_y = float(face_row[4]), float(face_row[5])   # Œil droit sujet
        le_x, le_y = float(face_row[6]), float(face_row[7])   # Œil gauche sujet
        nt_x, nt_y = float(face_row[8]), float(face_row[9])   # Pointe du nez
        rc_x, rc_y = float(face_row[10]), float(face_row[11]) # Coin droit bouche
        lc_x, lc_y = float(face_row[12]), float(face_row[13]) # Coin gauche bouche

        dx_yeux = le_x - re_x
        dy_yeux = le_y - re_y
        dist_yeux = math.hypot(dx_yeux, dy_yeux)
        if dist_yeux < 18.0:
            return 0.0

        # 1. Inclinaison latérale de la tête (roll)
        roll_deg = abs(math.degrees(math.atan2(dy_yeux, dx_yeux)))
        if roll_deg > 28.0:
            return 0.0

        # Repère orthonormé lié au visage : u = axe inter-oculaire, v = axe vertical descendant
        ux, uy = dx_yeux / dist_yeux, dy_yeux / dist_yeux
        vx, vy = -uy, ux

        mid_eyes_x = (re_x + le_x) * 0.5
        mid_eyes_y = (re_y + le_y) * 0.5
        mid_mouth_x = (rc_x + lc_x) * 0.5
        mid_mouth_y = (rc_y + lc_y) * 0.5

        # Projections nez et bouche
        dn_x, dn_y = nt_x - mid_eyes_x, nt_y - mid_eyes_y
        dm_x, dm_y = mid_mouth_x - mid_eyes_x, mid_mouth_y - mid_eyes_y

        yaw_nose = (dn_x * ux + dn_y * uy) / dist_yeux
        yaw_mouth = (dm_x * ux + dm_y * uy) / dist_yeux
        yaw_err = 0.70 * abs(yaw_nose) + 0.30 * abs(yaw_mouth)

        pitch_nose = (dn_x * vx + dn_y * vy) / dist_yeux
        pitch_mouth = max(1.0, dm_x * vx + dm_y * vy)
        ratio_vertical = (dn_x * vx + dn_y * vy) / pitch_mouth

        # Plages normales face à la caméra :
        # - yaw_err <= 0.16 (tête bien de face, pas tournée à gauche/droite)
        # - pitch_nose dans [0.35, 0.76] (pas tête baissée sur clavier/téléphone ni levée)
        # - ratio_vertical dans [0.42, 0.68] (proportion yeux-nez / yeux-bouche cohérente)
        if yaw_err > 0.22 or not (0.30 <= pitch_nose <= 0.82) or not (0.38 <= ratio_vertical <= 0.74):
            return 0.0

        score_yaw = max(0.0, 1.0 - (yaw_err / 0.22))
        score_pitch = max(0.0, 1.0 - abs(pitch_nose - 0.56) / 0.26)
        score_vert = max(0.0, 1.0 - abs(ratio_vertical - 0.56) / 0.18)
        return round(0.50 * score_yaw + 0.30 * score_pitch + 0.20 * score_vert, 2)

    # ---------------- ESCALADE SÉMANTIQUE VERS GEMMA 3 12B ----------------

    def _analyser_intrusion_vlm(self, frame: np.ndarray, motif_declenchement: str) -> None:
        dossier_alertes = BASE_DIR / "captures" / "alertes"
        dossier_alertes.mkdir(parents=True, exist_ok=True)
        chemin_img = dossier_alertes / f"alerte_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(str(chemin_img), frame)
        panels.log_perception(f"[SÉCURITÉ] Escalade VLM ({motif_declenchement}) -> {chemin_img.name}")

        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return

        modele_vision = self.cfg.get("modele_vision", "gemma3:12b-it-qat")
        prompt_secu = (
            "Tu analyses une image de webcam dans une chambre/bureau. "
            "Une personne dont le visage n'a pas été reconnu est assise ou présente dans la pièce. "
            "RÈGLE IMPORTANTE : Être assis devant un ordinateur, porter un casque audio/micro, "
            "avoir les cheveux longs ou porter des vêtements sombres dans une pièce éclairée est TOTALEMENT NORMAL. "
            "Classe en 'suspect' ou 'intrusion' UNIQUEMENT en cas d'acte manifestement anormal "
            "(fouille de tiroirs/affaires, cagoule/masque couvrant le visage, lampe torche dans le noir, dégradation). "
            "Réponds UNIQUEMENT en JSON valide sur une ligne : "
            '{"niveau": "normal|suspect|intrusion", "description": "phrase factuelle courte"}'
        )
        try:
            rep = ollama.chat(
                model=modele_vision,
                messages=[{"role": "user", "content": prompt_secu, "images": [bytes(buf)]}],
                options={"temperature": 0.1, "num_predict": 120},
                keep_alive=0,
            )
            brut = rep.get("message", {}).get("content", "").strip()
            match_json = re.search(r"\{.*\}", brut, flags=re.S)
            if match_json:
                verdict = json.loads(match_json.group(0))
                niveau = str(verdict.get("niveau", "suspect")).lower()
                desc = str(verdict.get("description", brut))
            else:
                niveau = "suspect"
                desc = brut

            self._dernier_verdict_securite = f"{niveau.upper()} : {desc}"
            panels.log_perception(f"[VERDICT SÉCURITÉ] {self._dernier_verdict_securite}")

            if niveau in ("suspect", "intrusion") and self.on_alerte_securite is not None:
                self.on_alerte_securite(
                    f"[ALERTE CAMÉRA - {niveau.upper()}] {desc} (capture : {chemin_img})"
                )
        except Exception as err:
            panels.log_perception(f"[SÉCURITÉ] Erreur analyse VLM : {err}")

    # ---------------- API PUBLIQUE CAPTEURS & OUTILS ----------------

    def regarde_ecran_maintenant(self, tolerance_sec: float = 0.45) -> bool:
        if not self.cfg.get("activer_vision_voix", True):
            return True
        if not self.camera_active:
            return False
        with self._lock_vision:
            return self._visage_detecte_actuel and ((time.monotonic() - self._dernier_regard_ts) <= tolerance_sec)

    def decrire_etat_capteurs(self) -> str:
        with self._lock_vision:
            pers = self._nb_personnes_actuel
            ident = ", ".join(self._personnes_identifiees) if self._personnes_identifiees else "aucune"
            objs = ", ".join(self._objets_detectes_actuels) if self._objets_detectes_actuels else "aucun objet notable"
            pos = self._position_visage_actuel or "non cadré"
            regard = "oui" if (self._visage_detecte_actuel and (time.monotonic() - self._dernier_regard_ts) <= 0.5) else "non"

        return (
            f"Caméra : {'ACTIVE' if self.camera_active else 'INACTIVE'} ({self.statut_camera}) | "
            f"Présence : {pers} personne(s) [identifiées : {ident} | position : {pos} | regarde écran : {regard}] | "
            f"Objets visibles : {objs} | Sécurité ({self.mode_securite}) : {self._dernier_verdict_securite} | "
            f"Micro : {self.statut_micro}."
        )

    def capturer_photo(self, nom_fichier: str = "") -> str:
        if not self.camera_active or cv2 is None:
            return f"Erreur : la caméra est inactive ({self.statut_camera})."

        with self._lock_vision:
            frame = self._derniere_frame.copy() if self._derniere_frame is not None else None

        if frame is None:
            return "Erreur : aucune image reçue de la caméra pour le moment."

        dossier_captures = BASE_DIR / "captures"
        dossier_captures.mkdir(parents=True, exist_ok=True)
        nom = Path(nom_fichier).name if nom_fichier.strip() else f"capture_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        if not nom.lower().endswith((".jpg", ".jpeg", ".png")):
            nom += ".jpg"

        chemin = dossier_captures / nom
        cv2.imwrite(str(chemin), frame)
        panels.log_perception(f"Capture caméra enregistrée : {chemin}")
        return f"Capture photo enregistrée sur le disque : {chemin}"

    def observer_camera(self, question: str = "Décris objectivement tout ce qui est visible sur l'image.") -> str:
        panels.log_perception(f"Analyse sémantique VLM : {question}")
        if not self.camera_active or cv2 is None:
            return f"Caméra INACTIVE ({self.statut_camera}). Impossible de voir quoi que ce soit."

        with self._lock_vision:
            frame = self._derniere_frame.copy() if self._derniere_frame is not None else None

        if frame is None:
            return "Caméra active mais aucune image reçue pour le moment."

        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return "Erreur d'encodage de l'image webcam."

        modele_vision = self.cfg.get("modele_vision", "gemma3:12b-it-qat")
        prompt_vision = (
            "Tu analyses une photo prise à l'instant par la webcam de l'utilisateur. "
            "Décris UNIQUEMENT ce qui est réellement visible (personne, attitude, vêtements et couleurs exactes, "
            "texte lisible, objets, arrière-plan). Ne devine rien. "
            f"Réponds en français de façon directe à : {question}"
        )
        try:
            rep = ollama.chat(
                model=modele_vision,
                messages=[{"role": "user", "content": prompt_vision, "images": [bytes(buf)]}],
                options={"temperature": 0.1},
                keep_alive=0,
            )
            observation = rep.get("message", {}).get("content", "").strip() or "Aucune description retournée."
            panels.log_perception(f"Observation VLM : {observation[:180]}")
            return observation
        except Exception as err:
            erreur = f"Erreur du modèle de vision ({modele_vision}) : {err}"
            panels.log_perception(erreur)
            return erreur

    # ---------------- BOUCLE CAMÉRA TEMPS RÉEL (CPU) ----------------

    def _boucle_camera(self) -> None:
        if cv2 is None:
            self.statut_camera = "module cv2 absent"
            panels.log_perception(f"{self.statut_camera} -> caméra désactivée.")
            return

        index_cam = int(self.cfg.get("camera_index", 0))
        cap = cv2.VideoCapture(index_cam)
        if not cap.isOpened():
            self.statut_camera = f"webcam index {index_cam} inaccessible"
            panels.log_perception(f"{self.statut_camera} -> mode clavier uniquement.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)

        moteur_obj = "YOLOX-Apache2" if self._yolox_net is not None else "OpenCV-HOG"
        moteur_face = "YuNet+SFace" if (self._yunet and self._sface) else "OpenCV-Face"
        self.camera_active = True
        self.statut_camera = f"active (idx {index_cam}, {moteur_obj}/{moteur_face})"
        panels.log_perception(f"Caméra activée : {self.statut_camera}")

        dernier_resume_log = ""
        dernier_log_ts = 0.0
        derniere_presence_ts = time.monotonic()
        frame_idx = 0

        try:
            while not self.arret_global.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.2)
                    continue

                maintenant = time.monotonic()
                frame_idx += 1
                h_img, w_img = frame.shape[:2]
                gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                luminosite = float(np.mean(gris))

                if frame_idx % 2 == 1:
                    nb_pers_corps, objets_detectes = self._detecter_objets_rapide(frame)
                else:
                    with self._lock_vision:
                        nb_pers_corps = self._nb_personnes_actuel
                        objets_detectes = list(self._objets_detectes_actuels)

                identites: list[str] = []
                identites_prises: set[str] = set()
                boites_retenues: list[tuple[int, int, int, int]] = []
                score_regard_frame = 0.0
                vec_principal: np.ndarray | None = None
                position_desc = ""

                if self._yunet is not None:
                    self._yunet.setInputSize((w_img, h_img))
                    _, faces = self._yunet.detect(frame)
                    if faces is not None and len(faces) > 0:
                        faces_triees = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                        for f in faces_triees:
                            x, y, w, h = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                            if w < 55 or h < 55:
                                continue
                            boite = (x, y, w, h)
                            if any(_iou_boites(boite, b_ex) > 0.35 for b_ex in boites_retenues):
                                continue
                            boites_retenues.append(boite)

                            if self._sface is not None:
                                aligne = self._sface.alignCrop(frame, f)
                                vec = self._sface.feature(aligne).flatten()
                            else:
                                roi = gris[max(0, y) : min(h_img, y + h), max(0, x) : min(w_img, x + w)]
                                vec = self._extraire_signature_visage_fallback(roi) if roi.size > 0 else None

                            nom_id = "Inconnu"
                            if vec is not None:
                                nom_id, _ = self._identifier_visage(vec, identites_prises)
                                # Grâce d'identité de 2.0s pour éviter les faux "Inconnu" quand l'utilisateur tourne légèrement la tête
                                if nom_id == "Inconnu" and self._dernier_resident_reconnu and (maintenant - self._dernier_resident_ts) <= 2.0:
                                    nom_id = self._dernier_resident_reconnu
                                elif nom_id not in ("Inconnu", "Non calibré"):
                                    self._dernier_resident_reconnu = nom_id
                                    self._dernier_resident_ts = maintenant
                                    identites_prises.add(nom_id)
                                identites.append(nom_id)
                                if len(boites_retenues) == 1:
                                    vec_principal = vec
                            else:
                                identites.append("Inconnu")

                            if len(boites_retenues) == 1:
                                centre_x = (x + w / 2.0) / float(w_img)
                                horiz = "gauche" if centre_x < 0.38 else ("droite" if centre_x > 0.62 else "centre")
                                dist = "proche" if (w / float(w_img)) > 0.25 else "moyen"
                                position_desc = f"{horiz}, {dist}"
                                if nom_id != "Inconnu":
                                    score_regard_frame = self._score_regard_frontal(f)
                else:
                    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
                    masque = cv2.inRange(
                        ycrcb,
                        np.array([25, 130, 75], dtype=np.uint8),
                        np.array([250, 178, 132], dtype=np.uint8),
                    )
                    contours, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    candidats = [
                        cv2.boundingRect(c)
                        for c in contours
                        if cv2.boundingRect(c)[2] >= 80 and cv2.boundingRect(c)[3] >= 90
                    ]
                    if candidats:
                        x, y, w, h = max(candidats, key=lambda r: r[2] * r[3])
                        roi = gris[y : y + h, x : x + w]
                        vec_principal = self._extraire_signature_visage_fallback(roi)
                        nom_id, _ = self._identifier_visage(vec_principal)
                        identites.append(nom_id)
                        score_regard_frame = 0.8 if nom_id != "Inconnu" else 0.0
                        position_desc = "centre"

                # Lissage temporel + hystérésis sur l'attention visuelle
                self._regard_ema = 0.60 * self._regard_ema + 0.40 * score_regard_frame
                if self._visage_detecte_actuel:
                    regard_resident = self._regard_ema >= 0.32
                else:
                    regard_resident = self._regard_ema >= 0.48

                nb_total_personnes = max(nb_pers_corps, len(identites))
                if nb_total_personnes > len(identites):
                    identites.extend(["Inconnu (visage masqué/dos)"] * (nb_total_personnes - len(identites)))

                profils_valides = self._profils_visages_compatibles()
                residents_presents = [n for n in identites if not n.startswith("Inconnu") and n != "Non calibré"]
                inconnus_presents = [n for n in identites if n.startswith("Inconnu")]

                if nb_total_personnes > 0:
                    derniere_presence_ts = maintenant
                elif (maintenant - derniere_presence_ts) >= 15.0:
                    self._alerte_session_deja_evaluee = False
                    self._debut_inconnu_seul_ts = 0.0

                if not profils_valides:
                    self._dernier_verdict_securite = "Non calibré (/enroll Bryan requis)"
                elif residents_presents and inconnus_presents:
                    self._debut_inconnu_seul_ts = 0.0
                    self._alerte_session_deja_evaluee = False
                    self._dernier_verdict_securite = f"Invité accompagné par {', '.join(residents_presents)}"
                elif residents_presents:
                    self._debut_inconnu_seul_ts = 0.0
                    self._alerte_session_deja_evaluee = False
                    self._dernier_verdict_securite = f"Résident ({', '.join(residents_presents)})"
                elif inconnus_presents and self.mode_securite != "off":
                    if self._debut_inconnu_seul_ts == 0.0:
                        self._debut_inconnu_seul_ts = maintenant
                    duree_inconnu = maintenant - self._debut_inconnu_seul_ts
                    seuil_sec = 1.5 if self.mode_securite == "absent" else 5.0
                    cooldown_ok = (maintenant - self._dernier_check_intrus_ts) >= 600.0
                    if duree_inconnu >= seuil_sec and (
                        not getattr(self, "_alerte_session_deja_evaluee", False) or cooldown_ok
                    ):
                        self._dernier_check_intrus_ts = maintenant
                        self._alerte_session_deja_evaluee = True
                        threading.Thread(
                            target=self._analyser_intrusion_vlm,
                            args=(
                                frame.copy(),
                                f"{len(inconnus_presents)} inconnu(s) seul(s) depuis {duree_inconnu:.1f}s",
                            ),
                            daemon=True,
                        ).start()
                else:
                    self._debut_inconnu_seul_ts = 0.0
                    if nb_total_personnes == 0:
                        self._dernier_verdict_securite = "Pièce vide"

                with self._lock_vision:
                    self._derniere_frame = frame.copy()
                    self._nb_personnes_actuel = nb_total_personnes
                    self._personnes_identifiees = identites
                    self._objets_detectes_actuels = objets_detectes
                    self._position_visage_actuel = position_desc
                    self._luminosite_actuelle = luminosite
                    self._visage_detecte_actuel = regard_resident
                    if vec_principal is not None:
                        self._dernier_vecteur_visage = vec_principal
                    if regard_resident:
                        self._dernier_regard_ts = maintenant

                regard_txt = f"OUI ({self._regard_ema:.2f})" if regard_resident else f"non ({self._regard_ema:.2f})"
                resume = f"{nb_total_personnes} pers. ({', '.join(identites) or 'aucune'}) | regard={regard_txt}"
                if resume != dernier_resume_log and (maintenant - dernier_log_ts) >= 1.2:
                    panels.log_perception(f"[VISION] {resume} | lum={luminosite:.0f}")
                    dernier_resume_log = resume
                    dernier_log_ts = maintenant

                pause = 0.10 if (maintenant - derniere_presence_ts) < 3.0 else 0.40
                time.sleep(pause)
        except Exception as err:
            panels.log_perception(f"[ERREUR CAMÉRA] {type(err).__name__} : {err}")
        finally:
            self.camera_active = False
            self.statut_camera = "arrêtée"
            cap.release()
            panels.log_perception("Caméra arrêtée.")

    # ---------------- AUDIO, ÉCOUTE CONTINUE & WHISPER (MIT) ----------------

    @staticmethod
    def _extraire_signature_voix(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        taille_trame = int(sr * 0.025)
        trames_actives = [
            audio[i : i + taille_trame]
            for i in range(0, len(audio) - taille_trame, taille_trame)
            if float(np.sqrt(np.mean(audio[i : i + taille_trame] ** 2))) > 0.005
        ]
        if len(trames_actives) < 4:
            return np.zeros(33, dtype=np.float32)

        signal_voise = np.concatenate(trames_actives)
        blocs = np.array_split(signal_voise, 16)
        energies = np.array([float(np.sqrt(np.mean(t**2))) for t in blocs], dtype=np.float32)
        spectre = np.abs(np.fft.rfft(signal_voise * np.hanning(len(signal_voise))))
        bandes = np.array([float(np.mean(b)) for b in np.array_split(spectre[: len(spectre) // 2], 16)], dtype=np.float32)

        extrait_pitch = signal_voise[: min(len(signal_voise), 4096)]
        autocorr = np.correlate(extrait_pitch, extrait_pitch, mode="full")[len(extrait_pitch) - 1 :]
        min_lag, max_lag = sr // 320, min(len(autocorr) - 1, sr // 75)
        pitch_norm = float(np.argmax(autocorr[min_lag:max_lag]) + min_lag) / float(max(1, max_lag))

        vec = np.concatenate([
            energies / (np.linalg.norm(energies) + 1e-6),
            bandes / (np.linalg.norm(bandes) + 1e-6),
            np.array([pitch_norm], dtype=np.float32),
        ])
        norme = float(np.linalg.norm(vec))
        return vec / norme if norme > 0 else vec

    def _verifier_locuteur(self, audio: np.ndarray) -> tuple[bool, float]:
        if not self._profils_visages_compatibles() or self.profil_voix is None:
            return True, 1.0

        with self._lock_vision:
            vec_visage_actuel = self._dernier_vecteur_visage.copy() if self._dernier_vecteur_visage is not None else None

        sim_visage = 0.0
        if vec_visage_actuel is not None:
            _, sim_visage = self._identifier_visage(vec_visage_actuel)

        vec_voix_actuel = self._extraire_signature_voix(audio)
        sim_voix = _cosinus(self.profil_voix, vec_voix_actuel)

        seuil_visage = float(self.cfg.get("seuil_visage_sface", 0.45)) if self._sface is not None else 0.78
        seuil_voix = float(self.cfg.get("seuil_voix", 0.65))
        if sim_visage >= seuil_visage or sim_voix >= seuil_voix:
            if sim_voix >= 0.75:
                self.profil_voix = 0.92 * self.profil_voix + 0.08 * vec_voix_actuel
            return True, round(max(sim_visage, sim_voix), 2)

        return False, round(max(sim_visage, sim_voix), 2)

    def _charger_whisper(self):
        if self._whisper is not None or WhisperModel is None:
            return self._whisper

        taille = str(self.cfg.get("modele_whisper", "small"))
        threads = int(self.cfg.get("whisper_cpu_threads", 6))

        try:
            self._push_log_audio(f"[AUDIO] Init Whisper ({taille}, cache local)...")
            self._whisper = WhisperModel(
                taille,
                device="cpu",
                compute_type="int8",
                cpu_threads=threads,
                local_files_only=True,
            )
            self._whisper_erreur = ""
            self._push_log_audio(f"[AUDIO] Faster-Whisper ({taille}) prêt.")
            return self._whisper
        except Exception as err_local:
            self._push_log_audio(f"[AUDIO] Cache absent ({type(err_local).__name__}), téléchargement...")

        try:
            self._whisper = WhisperModel(
                taille,
                device="cpu",
                compute_type="int8",
                cpu_threads=threads,
                local_files_only=False,
            )
            self._whisper_erreur = ""
            self._push_log_audio(f"[AUDIO] Faster-Whisper ({taille}) téléchargé et prêt.")
            return self._whisper
        except Exception as err:
            self._whisper_erreur = f"{type(err).__name__}: {err}"
            self._push_log_audio(f"[ERREUR WHISPER] {self._whisper_erreur}")
            return None

    def _journaliser_ecoute_continue(self, texte: str, statut: str) -> None:
        if not texte:
            return
        # Par défaut (RGPD / usage commercial), on ne sauvegarde pas les écoutes passives sur disque
        if statut == "passif" and not bool(self.cfg.get("journaliser_audio_passif", False)):
            return
        try:
            dossier = BASE_DIR / "captures"
            dossier.mkdir(parents=True, exist_ok=True)
            fichier = dossier / f"ecoute_{time.strftime('%Y-%m-%d')}.log"
            horodatage = time.strftime("%H:%M:%S")
            with fichier.open("a", encoding="utf-8") as f:
                f.write(f"[{horodatage}] [{statut}] {texte}\n")
        except OSError:
            pass

    def _lister_et_choisir_micro(self) -> tuple[int | str | None, str, int]:
        """Liste toutes les entrées audio dans les logs et choisit le meilleur périphérique."""
        device_cfg = self.cfg.get("micro_device", None)
        try:
            devices = sd.query_devices()
            entrees = [
                f"#{idx}: {d['name']}"
                for idx, d in enumerate(devices)
                if d.get("max_input_channels", 0) > 0
            ]
            self._push_log_audio(f"[MICRO DEVICES] {' | '.join(entrees) or 'AUCUNE ENTREE'}")
        except Exception as err:
            self._push_log_audio(f"[MICRO DEVICES] Erreur liste : {err}")

        cible = device_cfg
        if cible is None:
            try:
                for idx, d in enumerate(sd.query_devices()):
                    if d.get("max_input_channels", 0) > 0 and str(d.get("name", "")).lower() in ("pipewire", "default"):
                        cible = idx
                        break
            except Exception:
                cible = None

        nom_dev = "défaut système"
        sr_natif = 16000
        try:
            info = sd.query_devices(cible, kind="input")
            if isinstance(info, dict):
                nom_dev = str(info.get("name", nom_dev))
                sr_natif = int(info.get("default_samplerate", 16000)) or 16000
        except Exception:
            pass

        return cible, nom_dev, sr_natif

    def _boucle_micro(self) -> None:
        try:
            if sd is None:
                self.statut_micro = "sounddevice absent"
                panels.log_audio("[AUDIO] sounddevice absent -> écoute vocale désactivée.")
                return
            if WhisperModel is None:
                self.statut_micro = "faster-whisper absent"
                panels.log_audio("[AUDIO] faster-whisper absent -> écoute vocale désactivée.")
                return

            device_micro, nom_dev, sr_natif = self._lister_et_choisir_micro()
            self._vider_logs_audio()

            threading.Thread(target=self._charger_whisper, daemon=True).start()

            seuil_rms_min = float(self.cfg.get("seuil_micro_rms", 0.016))
            # Pauses naturelles : 1.4s si on regarde l'écran (laisse finir les phrases), 0.9s sinon
            pause_regard_blocs = int(float(self.cfg.get("pause_fin_phrase_regard_sec", 1.4)) * 10)
            pause_passive_blocs = int(float(self.cfg.get("pause_fin_phrase_passive_sec", 0.9)) * 10)

            sr_capture = 16000
            try:
                sd.check_input_settings(device=device_micro, samplerate=16000, channels=1, dtype="float32")
            except Exception:
                sr_capture = sr_natif

            bloc_duree = 0.1
            bloc_taille = int(sr_capture * bloc_duree)

            bruit_ambiant_rms = seuil_rms_min * 0.75
            blocs_calibration = 0
            echantillons_calib: list[float] = []

            tampon_parole: list[np.ndarray] = []
            pre_buffer: list[np.ndarray] = []
            blocs_silence = 0
            blocs_barge_in = 0
            blocs_regard_parole = 0
            blocs_totaux_parole = 0
            en_parole = False
            pic_rms_seconde = 0.0
            dernier_rms = 0.0
            seuil_courant = seuil_rms_min

            def callback_audio(indata, frames, time_info, status):
                nonlocal en_parole, blocs_silence, blocs_barge_in, tampon_parole, pre_buffer
                nonlocal bruit_ambiant_rms, blocs_regard_parole, blocs_totaux_parole
                nonlocal pic_rms_seconde, dernier_rms, seuil_courant
                nonlocal blocs_calibration, echantillons_calib

                if self.arret_global.is_set():
                    return

                if status:
                    self._push_log_audio(f"[MICRO STATUS] {status}")

                morceau_brut = indata[:, 0].copy()
                morceau = _reechantillonner_16k(morceau_brut, sr_capture)
                rms = float(np.sqrt(np.mean(morceau**2)))
                dernier_rms = rms
                if rms > pic_rms_seconde:
                    pic_rms_seconde = rms

                # Calibration automatique du bruit ambiant sur les 15 premiers blocs (1.5s)
                if blocs_calibration < 15:
                    blocs_calibration += 1
                    echantillons_calib.append(rms)
                    pre_buffer.append(morceau)
                    if len(pre_buffer) > 8:
                        pre_buffer.pop(0)
                    if blocs_calibration == 15:
                        bruit_ambiant_rms = float(np.percentile(echantillons_calib, 80))
                        seuil_courant = max(seuil_rms_min, bruit_ambiant_rms * 1.65)
                        self._push_log_audio(
                            f"[CALIBRATION] Bruit fond={bruit_ambiant_rms:.4f} -> seuil auto={seuil_courant:.4f}"
                        )
                    return

                regarde = self.regarde_ecran_maintenant(tolerance_sec=0.45)
                seuil_dynamique = max(seuil_rms_min, bruit_ambiant_rms * 1.65)
                seuil_courant = seuil_dynamique

                # Barge-in sécurisé
                if self.ia_en_cours.is_set():
                    seuil_interruption = max(seuil_rms_min * 1.6, bruit_ambiant_rms * 2.2)
                    if regarde and rms > seuil_interruption:
                        blocs_barge_in += 1
                        if blocs_barge_in == 8 and len(tampon_parole) >= 7:
                            extrait_test = np.concatenate(tampon_parole[-8:])
                            if self.profil_voix is None:
                                self.interruption_event.set()
                            else:
                                sim_v = _cosinus(self.profil_voix, self._extraire_signature_voix(extrait_test, 16000))
                                if sim_v >= 0.65:
                                    self.interruption_event.set()
                    else:
                        blocs_barge_in = 0
                else:
                    blocs_barge_in = 0

                if not en_parole:
                    bruit_ambiant_rms = 0.94 * bruit_ambiant_rms + 0.06 * rms
                    pre_buffer.append(morceau)
                    if len(pre_buffer) > 8:
                        pre_buffer.pop(0)

                    if rms > seuil_dynamique:
                        en_parole = True
                        blocs_silence = 0
                        blocs_totaux_parole = 1
                        blocs_regard_parole = 1 if regarde else 0
                        tampon_parole = list(pre_buffer)
                        self._push_log_audio(
                            f"[MICRO] Parole détectée (RMS={rms:.4f} > {seuil_dynamique:.4f} | regard={'OUI' if regarde else 'non'})"
                        )
                else:
                    tampon_parole.append(morceau)
                    blocs_totaux_parole += 1
                    if regarde:
                        blocs_regard_parole += 1

                    # Seuil de maintien plus bas (0.55x) pour ne pas couper sur les fins de mots
                    if rms > seuil_dynamique * 0.55:
                        blocs_silence = 0
                    else:
                        blocs_silence += 1

                    cible_silence = pause_regard_blocs if (regarde or blocs_regard_parole >= 3) else pause_passive_blocs

                    if blocs_silence >= cible_silence or len(tampon_parole) >= 180:
                        sature_duree = len(tampon_parole) >= 180
                        # On retire l'excédent de silence de fin tout en gardant 2 blocs (0.2s) de marge
                        a_retirer = max(0, blocs_silence - 2)
                        tranches = tampon_parole[:-a_retirer] if a_retirer > 0 and a_retirer < len(tampon_parole) else tampon_parole
                        audio_complet = np.concatenate(tranches)

                        ratio_regard = blocs_regard_parole / float(max(1, blocs_totaux_parole))
                        regard_segment = (blocs_regard_parole >= 4) and (ratio_regard >= 0.25)
                        duree_sec = len(audio_complet) / 16000.0
                        rms_moy = float(np.sqrt(np.mean(audio_complet**2)))

                        en_parole = False
                        blocs_silence = 0
                        blocs_regard_parole = 0
                        blocs_totaux_parole = 0
                        tampon_parole = []
                        pre_buffer = []

                        if sature_duree:
                            bruit_ambiant_rms = max(bruit_ambiant_rms, rms_moy * 1.1)
                            self._push_log_audio(
                                f"[CALIBRATION] Bruit ambiant ajusté à {bruit_ambiant_rms:.4f}"
                            )

                        if duree_sec < 0.50:
                            return

                        if self._whisper is None:
                            raison = self._whisper_erreur or "modèle en cours de chargement"
                            self._push_log_audio(f"[AUDIO] Parole ({duree_sec:.1f}s) ignorée : {raison}")
                            return

                        if self._transcriptions_en_attente >= 2:
                            self._push_log_audio("[MICRO] File Whisper pleine, segment ignoré.")
                            return

                        self._push_log_audio(
                            f"[MICRO] Transcription ({duree_sec:.2f}s, regard={ratio_regard:.0%})..."
                        )
                        self._transcriptions_en_attente += 1
                        threading.Thread(
                            target=self._traiter_segment_audio,
                            args=(audio_complet, regard_segment),
                            daemon=True,
                        ).start()

            with sd.InputStream(
                samplerate=sr_capture,
                channels=1,
                dtype="float32",
                blocksize=bloc_taille,
                device=device_micro,
                callback=callback_audio,
            ):
                self.micro_actif = True
                self.statut_micro = f"actif ({nom_dev} @ {sr_capture}Hz)"
                panels.log_audio(f"Micro ouvert : {nom_dev} ({sr_capture}Hz, calibration 1.5s...)")

                while not self.arret_global.is_set():
                    self._vider_logs_audio()
                    if blocs_calibration < 15:
                        panels.maj_vu_micro("Calibration bruit ambiant en cours...")
                    else:
                        etat = "PAROLE" if en_parole else "écoute"
                        vu = _barre_vu(dernier_rms, echelle=max(0.05, seuil_courant * 3.5))
                        panels.maj_vu_micro(
                            f"{vu} RMS:{dernier_rms:.4f} (pic:{pic_rms_seconde:.4f} / seuil:{seuil_courant:.4f}) [{etat}]"
                        )
                    pic_rms_seconde *= 0.85
                    time.sleep(0.15)
        except Exception as err:
            self.micro_actif = False
            self.statut_micro = f"erreur ({type(err).__name__}: {err})"
            detail = traceback.format_exc().strip().splitlines()[-1]
            panels.maj_vu_micro("Micro : ERREUR")
            panels.log_audio(f"[ERREUR MICRO] {detail}")

    def _traiter_segment_audio(self, audio: np.ndarray, a_regarde: bool) -> None:
        try:
            if self._whisper is None:
                return
            with self._lock_whisper:
                old_out, old_err = sys.stdout, sys.stderr
                try:
                    if sys.__stdout__ is not None:
                        sys.stdout = sys.__stdout__
                    if sys.__stderr__ is not None:
                        sys.stderr = sys.__stderr__

                    segments, _ = self._whisper.transcribe(
                        audio,
                        language="fr",
                        initial_prompt="Conversation vocale en français avec l'assistant Éon.",
                        beam_size=5,
                        temperature=0.0,
                        condition_on_previous_text=False,
                        no_speech_threshold=0.65,
                        log_prob_threshold=-0.85,
                        compression_ratio_threshold=2.2,
                        vad_filter=True,
                        vad_parameters={
                            "threshold": 0.45,
                            "min_speech_duration_ms": 200,
                            "min_silence_duration_ms": 350,
                            "speech_pad_ms": 200,
                        },
                    )
                    morceaux_bruts = []
                    morceaux_valides = []
                    prob_silence_min = 1.0
                    for seg in segments:
                        txt_seg = seg.text.strip()
                        nsp = float(getattr(seg, "no_speech_prob", 0.0))
                        alp = float(getattr(seg, "avg_logprob", 0.0))
                        prob_silence_min = min(prob_silence_min, nsp)
                        if txt_seg:
                            morceaux_bruts.append(txt_seg)
                        if nsp <= 0.60 and alp >= -0.85:
                            morceaux_valides.append(txt_seg)
                finally:
                    sys.stdout, sys.stderr = old_out, old_err

                texte_brut = " ".join(morceaux_bruts).strip()
                texte = " ".join(morceaux_valides).strip()
        except Exception as err:
            self._push_log_audio(f"[WHISPER ERREUR] {type(err).__name__}: {err}")
            return
        finally:
            self._transcriptions_en_attente = max(0, self._transcriptions_en_attente - 1)

        if not texte_brut:
            self._push_log_audio("[WHISPER] (silence / bruit non verbal)")
            return

        self._push_log_audio(f"[WHISPER] « {texte_brut} » (ns={prob_silence_min:.2f})")

        texte_norm = texte.lower().strip(" !?.,;:")
        if (
            len(texte) < 2
            or texte_norm in BRUITS_PARASITES_WHISPER
            or any(sub in texte_norm for sub in HALLUCINATIONS_SUBSTRINGS)
        ):
            return

        contient_eon = bool(REGEX_WAKE_WORD.search(texte))
        toujours_repondre = bool(self.cfg.get("repondre_sans_mot_cle", False))
        doit_repondre = a_regarde or contient_eon or toujours_repondre

        if not doit_repondre:
            self._push_log_audio(f"[PASSIF] {texte}")
            self._journaliser_ecoute_continue(texte, "passif")
            return

        if self.ia_en_cours.is_set() and contient_eon:
            self.interruption_event.set()

        autorise, conf = self._verifier_locuteur(audio)
        if not autorise and not contient_eon:
            self._push_log_audio(f"[IGNORÉ locuteur={conf:.2f}] {texte}")
            self._journaliser_ecoute_continue(texte, f"ignore_locuteur:{conf:.2f}")
            return

        if (
            a_regarde
            and (not self._profils_visages_compatibles() or self.profil_voix is None)
            and len(audio) >= int(16000 * 1.0)
            and prob_silence_min < 0.35
        ):
            with self._lock_vision:
                vec_v = self._dernier_vecteur_visage.copy() if self._dernier_vecteur_visage is not None else None
            if vec_v is not None:
                self.profils_visages["Bryan"] = vec_v
                self.profil_voix = self._extraire_signature_voix(audio)
                self._sauvegarder_profil()
                self._push_log_audio("[BIOMÉTRIE] Profil calibré (Bryan).")

        motif_declenchement = "mot-clé Éon" if contient_eon and not a_regarde else ("regard+Éon" if contient_eon else "regard")
        self.pret_pour_saisie.clear()
        self._push_log_audio(f"[VOIX -> EON | {motif_declenchement}] {texte}")
        self._journaliser_ecoute_continue(texte, f"actif:{motif_declenchement}")
        self.file_messages.put(MessageEntrant(texte=texte, source="voix", confiance_locuteur=conf))

    # ---------------- CLAVIER FALLBACK (SI TEXTUAL INACTIF) ----------------

    def _boucle_clavier_fallback(self) -> None:
        while not self.arret_global.is_set():
            if self.ia_en_cours.is_set():
                if sys.stdin is not None and sys.stdin.isatty() and termios is not None:
                    fd = sys.stdin.fileno()
                    anciens_params = termios.tcgetattr(fd)
                    try:
                        tty.setcbreak(fd)
                        while self.ia_en_cours.is_set() and not self.arret_global.is_set():
                            pret, _, _ = select.select([sys.stdin], [], [], 0.08)
                            if pret:
                                if sys.stdin.read(1) == "\x1b":
                                    self.interruption_event.set()
                                    break
                    except Exception:
                        time.sleep(0.1)
                    finally:
                        termios.tcsetattr(fd, termios.TCSADRAIN, anciens_params)
                else:
                    time.sleep(0.1)
                continue

            if not self.pret_pour_saisie.is_set():
                time.sleep(0.05)
                continue

            try:
                ligne = panels.saisir("Prompt : ").strip()
                if not ligne:
                    self.arret_global.set()
                    self.file_messages.put(MessageEntrant(texte="", source="clavier"))
                    break
                self.pret_pour_saisie.clear()
                self.file_messages.put(MessageEntrant(texte=ligne, source="clavier"))
            except (KeyboardInterrupt, EOFError):
                self.arret_global.set()
                self.file_messages.put(MessageEntrant(texte="", source="clavier"))
                break

    def demarrer(self) -> None:
        if self.cfg.get("activer_vision_voix", True):
            if cv2 is not None:
                threading.Thread(target=self._boucle_camera, daemon=True).start()
            threading.Thread(target=self._boucle_micro, daemon=True).start()
            time.sleep(0.25)

        if not panels.tui_active():
            threading.Thread(target=self._boucle_clavier_fallback, daemon=True).start()

    def attendre_message(self) -> MessageEntrant | None:
        while not self.arret_global.is_set():
            try:
                msg = self.file_messages.get(timeout=0.2)
                if not msg.texte and msg.source == "clavier":
                    return None
                return msg
            except queue.Empty:
                continue
        return None