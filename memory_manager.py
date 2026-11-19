import hashlib
import json
import os
import re
import sqlite3
import threading
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
HISTORIQUE_LEGACY_FILE = BASE_DIR / "historique_brut.jsonl"
HISTORIQUE_BIN_FILE = BASE_DIR / "historique_brut.bin"
HISTORIQUE_DB_FILE = BASE_DIR / "historique_eon.db"
CLE_HISTORIQUE_FILE = BASE_DIR / ".history.key"
MEMOIRE_UTILE_FILE = BASE_DIR / "memoire_utile.json"
MEMOIRE_EPISODIQUE_FILE = BASE_DIR / "memoire_episodique.jsonl"

_DB_LOCK = threading.RLock()
_DB_INITIALISEE = False


def _obtenir_cle() -> bytes:
    """Charge ou génère une clé locale de 32 octets."""
    if CLE_HISTORIQUE_FILE.exists():
        cle = CLE_HISTORIQUE_FILE.read_bytes()
        if cle:
            return cle
    cle = os.urandom(32)
    CLE_HISTORIQUE_FILE.write_bytes(cle)
    try:
        os.chmod(CLE_HISTORIQUE_FILE, 0o600)
    except OSError:
        pass
    return cle


def _flux_masque(data: bytes, cle: bytes) -> bytes:
    """Chiffrement de flux SHAKE-256 sans répétition de clé."""
    flux = hashlib.shake_256(cle).digest(len(data))
    return bytes(a ^ b for a, b in zip(data, flux))


def _ecrire_atomique_texte(chemin: Path, contenu: str) -> None:
    tmp = chemin.with_suffix(".tmp")
    tmp.write_text(contenu, encoding="utf-8")
    os.replace(tmp, chemin)


def _ecrire_atomique_bytes(chemin: Path, contenu: bytes) -> None:
    tmp = chemin.with_suffix(".tmp")
    tmp.write_bytes(contenu)
    os.replace(tmp, chemin)


def _parser_lignes_jsonl(texte: str) -> list[dict[str, Any]]:
    entrees: list[dict[str, Any]] = []
    for ligne in texte.splitlines():
        ligne = ligne.strip()
        if not ligne:
            continue
        try:
            entrees.append(json.loads(ligne))
        except json.JSONDecodeError:
            continue
    return entrees


def _lire_jsonl(chemin: Path) -> list[dict[str, Any]]:
    if not chemin.exists():
        return []
    return _parser_lignes_jsonl(chemin.read_text(encoding="utf-8"))


def _connexion_db() -> sqlite3.Connection:
    global _DB_INITIALISEE
    conn = sqlite3.connect(str(HISTORIQUE_DB_FILE), timeout=10.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    if not _DB_INITIALISEE:
        with _DB_LOCK:
            if not _DB_INITIALISEE:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                conn.execute("PRAGMA temp_store=MEMORY;")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS echanges (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        date TEXT,
                        modele TEXT,
                        modele_initial TEXT,
                        raison TEXT,
                        score REAL,
                        prompt TEXT,
                        reponse TEXT,
                        echec_detecte INTEGER DEFAULT 0,
                        raison_echec TEXT,
                        modele_attendu TEXT,
                        payload_json TEXT NOT NULL
                    );
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_echanges_session ON echanges(session_id, id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_echanges_date ON echanges(date);")
                conn.commit()
                _DB_INITIALISEE = True
                _migrer_anciennes_archives(conn)
    return conn


def _migrer_anciennes_archives(conn: sqlite3.Connection) -> None:
    """Migre automatiquement historique_brut.bin ou .jsonl vers SQLite au premier lancement."""
    cur = conn.execute("SELECT COUNT(*) FROM echanges")
    if cur.fetchone()[0] > 0:
        return

    anciens: list[dict[str, Any]] = []
    if HISTORIQUE_BIN_FILE.exists():
        try:
            brut = HISTORIQUE_BIN_FILE.read_bytes()
            cle = _obtenir_cle()
            compresse = bytes(b ^ cle[i % len(cle)] for i, b in enumerate(brut))
            texte = zlib.decompress(compresse).decode("utf-8")
            anciens = _parser_lignes_jsonl(texte)
        except Exception:
            anciens = []
    elif HISTORIQUE_LEGACY_FILE.exists():
        anciens = _lire_jsonl(HISTORIQUE_LEGACY_FILE)

    if anciens:
        for e in anciens:
            _inserer_ligne_db(conn, e)
        conn.commit()


def _inserer_ligne_db(conn: sqlite3.Connection, entree: dict[str, Any]) -> None:
    date_val = entree.get("date") or entree.get("timestamp") or datetime.now().isoformat(timespec="seconds")
    conn.execute(
        """
        INSERT INTO echanges (
            session_id, date, modele, modele_initial, raison, score,
            prompt, reponse, echec_detecte, raison_echec, modele_attendu, payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            entree.get("session_id", ""),
            date_val,
            entree.get("modele", ""),
            entree.get("modele_initial", ""),
            entree.get("raison", ""),
            float(entree.get("score", 0.0) or 0.0),
            entree.get("prompt", ""),
            entree.get("reponse", ""),
            1 if entree.get("echec_detecte") else 0,
            entree.get("raison_echec"),
            entree.get("modele_attendu"),
            json.dumps(entree, ensure_ascii=False),
        ),
    )


def charger_tous_echanges() -> list[dict[str, Any]]:
    """Retourne l'ensemble des échanges (utilisé par optimizer.py)."""
    with _DB_LOCK:
        conn = _connexion_db()
        rows = conn.execute("SELECT payload_json FROM echanges ORDER BY id ASC").fetchall()
        res: list[dict[str, Any]] = []
        for r in rows:
            try:
                res.append(json.loads(r["payload_json"]))
            except json.JSONDecodeError:
                continue
        return res


def sauvegarder_tous_echanges(echanges: list[dict[str, Any]]) -> None:
    """Remplace la table SQLite et met à jour l'archive chiffrée de sauvegarde."""
    with _DB_LOCK:
        conn = _connexion_db()
        conn.execute("DELETE FROM echanges")
        for e in echanges:
            _inserer_ligne_db(conn, e)
        conn.commit()

    texte_jsonl = "\n".join(json.dumps(e, ensure_ascii=False) for e in echanges)
    if texte_jsonl:
        texte_jsonl += "\n"
    compresse = zlib.compress(texte_jsonl.encode("utf-8"), level=6)
    masque = _flux_masque(compresse, _obtenir_cle())
    _ecrire_atomique_bytes(HISTORIQUE_BIN_FILE, masque)


def journaliser_echange(entree: dict[str, Any]) -> None:
    """Insertion O(1) instantanée dans SQLite."""
    with _DB_LOCK:
        conn = _connexion_db()
        _inserer_ligne_db(conn, entree)
        conn.commit()


def marquer_dernier_echange_echec(raison_echec: str, modele_corrige: str | None = None) -> None:
    """Annote directement le dernier échange en base pour l'optimiseur IA."""
    with _DB_LOCK:
        conn = _connexion_db()
        row = conn.execute("SELECT id, payload_json FROM echanges ORDER BY id DESC LIMIT 1").fetchone()
        if not row:
            return
        try:
            data = json.loads(row["payload_json"])
        except json.JSONDecodeError:
            data = {}
        data["echec_detecte"] = True
        data["raison_echec"] = raison_echec
        if modele_corrige:
            data["modele_attendu"] = modele_corrige

        conn.execute(
            """
            UPDATE echanges
            SET echec_detecte = 1, raison_echec = ?, modele_attendu = COALESCE(?, modele_attendu), payload_json = ?
            WHERE id = ?
            """,
            (raison_echec, modele_corrige, json.dumps(data, ensure_ascii=False), row["id"]),
        )
        conn.commit()


def charger_memoire_rapide(session_id: str, limite_echanges: int = 6) -> list[dict[str, str]]:
    """Charge uniquement les N derniers échanges de la session active via index SQLite."""
    with _DB_LOCK:
        conn = _connexion_db()
        rows = conn.execute(
            """
            SELECT prompt, reponse FROM echanges
            WHERE session_id = ? AND prompt != '' AND reponse != ''
            ORDER BY id DESC LIMIT ?
            """,
            (session_id, limite_echanges),
        ).fetchall()

    messages: list[dict[str, str]] = []
    for r in reversed(rows):
        messages.append({"role": "user", "content": str(r["prompt"])})
        messages.append({"role": "assistant", "content": str(r["reponse"])})
    return messages


def charger_memoire_utile() -> dict[str, Any]:
    if not MEMOIRE_UTILE_FILE.exists():
        return {}
    try:
        data = json.loads(MEMOIRE_UTILE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def sauvegarder_memoire_utile(memoire: dict[str, Any]) -> None:
    _ecrire_atomique_texte(MEMOIRE_UTILE_FILE, json.dumps(memoire, indent=2, ensure_ascii=False))


def memoriser_fait_utilisateur(cle: str, valeur: str) -> str:
    """Enregistre ou met à jour une information durable sur l'utilisateur."""
    cle_propre = cle.strip().lower().replace(" ", "_")
    val_propre = valeur.strip()
    if not cle_propre or not val_propre:
        return "Erreur : clé ou valeur vide."

    memoire = charger_memoire_utile()
    memoire[cle_propre] = {
        "valeur": val_propre,
        "maj": datetime.now().isoformat(timespec="seconds"),
    }
    sauvegarder_memoire_utile(memoire)
    return f"Fait mémorisé : {cle_propre} = {val_propre}"


def supprimer_fait_utilisateur(cle: str) -> bool:
    memoire = charger_memoire_utile()
    if cle in memoire:
        del memoire[cle]
        sauvegarder_memoire_utile(memoire)
        return True
    return False


def formater_contexte_memoire() -> str:
    blocs: list[str] = []

    memoire = charger_memoire_utile()
    if memoire:
        lignes_faits = []
        for cle, info in memoire.items():
            val = info.get("valeur", info) if isinstance(info, dict) else info
            lignes_faits.append(f"- {cle} : {val}")
        blocs.append("MÉMOIRE UTILISATEUR PERMANENTE :\n" + "\n".join(lignes_faits))

    episodes = _lire_jsonl(MEMOIRE_EPISODIQUE_FILE)
    if episodes:
        lignes_ep = [
            f"- [{ep.get('date', '?')}] {ep.get('resume', '')}"
            for ep in episodes[-3:]
            if ep.get("resume")
        ]
        if lignes_ep:
            blocs.append("RÉSUMÉS DES SESSIONS PRÉCÉDENTES :\n" + "\n".join(lignes_ep))

    return "\n\n".join(blocs)


def consulter_historique(limite: int = 10, inclure_episodes: bool = True) -> str:
    """Consulte l'historique des conversations passées et les résumés journaliers."""
    nb = max(1, min(int(limite), 150))
    sections: list[str] = []

    with _DB_LOCK:
        conn = _connexion_db()
        rows = conn.execute(
            "SELECT date, modele, echec_detecte, prompt, reponse FROM echanges ORDER BY id DESC LIMIT ?",
            (nb,),
        ).fetchall()

    if rows:
        entrees = []
        troncature = 140 if nb > 40 else 220
        for d in reversed(rows):
            flag = " [ÉCHEC]" if d["echec_detecte"] else ""
            p = str(d["prompt"] or "").replace("\n", " ")
            r = str(d["reponse"] or "").replace("\n", " ")[:troncature]
            entrees.append(f"- [{d['date']}] ({d['modele']}){flag} Moi: {p}\n  IA: {r}...")
        sections.append(f"ÉCHANGES RÉCENTS ({len(entrees)} derniers) :\n" + "\n".join(entrees))

    if inclure_episodes:
        episodes = _lire_jsonl(MEMOIRE_EPISODIQUE_FILE)
        if episodes:
            lignes_ep = [
                f"- ({ep.get('date', '?')}) {ep.get('resume', '')}"
                for ep in episodes[-5:]
                if ep.get("resume")
            ]
            if lignes_ep:
                sections.append("RÉSUMÉS ÉPISODIQUES :\n" + "\n".join(lignes_ep))

    return "\n\n".join(sections) if sections else "Aucun historique enregistré pour le moment."


def rechercher_dans_historique(requete: str, limite: int = 15) -> str:
    """Recherche des échanges passés contenant un mot ou sujet précis dans la base SQLite."""
    motif = requete.strip()
    if not motif:
        return "Erreur : terme de recherche vide."
    nb = max(1, min(int(limite), 50))
    like_arg = f"%{motif}%"

    with _DB_LOCK:
        conn = _connexion_db()
        rows = conn.execute(
            """
            SELECT date, modele, prompt, reponse FROM echanges
            WHERE prompt LIKE ? OR reponse LIKE ?
            ORDER BY id DESC LIMIT ?
            """,
            (like_arg, like_arg, nb),
        ).fetchall()

    if not rows:
        return f"Aucun échange trouvé dans l'historique pour « {motif} »."

    lignes = [f"Résultats dans l'historique pour « {motif} » ({len(rows)} trouvé(s)) :"]
    for d in reversed(rows):
        p = str(d["prompt"] or "").replace("\n", " ")
        r = str(d["reponse"] or "").replace("\n", " ")[:240]
        lignes.append(f"- [{d['date']}] ({d['modele']}) Moi: {p}\n  IA: {r}...")
    return "\n".join(lignes)


def exporter_historique(
    chemin_destination: str = "~/Téléchargements/conversation_eon.txt",
    limite: int = 100,
) -> str:
    """Exporte les vrais échanges enregistrés depuis la base SQLite vers un fichier texte sur le disque."""
    nettoye = re.sub(r"/\s+", "/", (chemin_destination or "").strip().strip("\"'"))
    cible = Path(nettoye or "~/Téléchargements/conversation_eon.txt").expanduser()
    if not cible.is_absolute():
        cible = (Path.home() / "Téléchargements" / cible).resolve()
    else:
        cible = cible.resolve()

    if cible.exists() and cible.is_dir():
        cible = cible / "conversation_eon.txt"

    racine_home = Path.home().resolve()
    if racine_home not in cible.parents and cible != racine_home:
        return f"Erreur : chemin d'export refusé hors de {racine_home} ({cible})."

    nb = max(1, min(int(limite), 1000))
    with _DB_LOCK:
        conn = _connexion_db()
        rows = conn.execute(
            """
            SELECT date, modele, echec_detecte, prompt, reponse
            FROM echanges
            WHERE prompt != ''
            ORDER BY id DESC LIMIT ?
            """,
            (nb,),
        ).fetchall()

    lignes_sortie = [
        f"=== HISTORIQUE RÉEL DE CONVERSATION EON (exporté le {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}) ===",
        f"Total exporté : {len(rows)} échange(s)",
        "=" * 72,
        "",
    ]

    for idx, d in enumerate(reversed(rows), start=1):
        statut = " [ÉCHEC SIGNALÉ]" if d["echec_detecte"] else ""
        lignes_sortie.append(f"[{idx}] {d['date']} | Modèle : {d['modele']}{statut}")
        lignes_sortie.append(f"Moi : {d['prompt']}")
        lignes_sortie.append(f"Eon : {d['reponse']}")
        lignes_sortie.append("-" * 72)

    cible.parent.mkdir(parents=True, exist_ok=True)
    _ecrire_atomique_texte(cible, "\n".join(lignes_sortie) + "\n")
    return f"Historique réel exporté avec succès ({len(rows)} échanges) dans : {cible}"


def effacer_historique_jour() -> int:
    aujourdhui = datetime.now().date().isoformat()
    with _DB_LOCK:
        conn = _connexion_db()
        cur = conn.execute("DELETE FROM echanges WHERE date LIKE ?", (f"{aujourdhui}%",))
        conn.commit()
        return cur.rowcount


def effacer_tout_historique() -> None:
    with _DB_LOCK:
        conn = _connexion_db()
        conn.execute("DELETE FROM echanges")
        conn.commit()