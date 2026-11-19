import difflib
import os
from pathlib import Path
from typing import Callable

import terminal_panels as panels

DOSSIERS_IGNORES = {
    ".git", ".venv", "venv", "__pycache__", "node_modules",
    "target", ".idea", ".mypy_cache", ".pytest_cache", "archives_logs",
}
RACINE_AUTORISEE = Path.home().resolve()

MODE_FICHIERS = "ask"  # "ask" | "auto" | "ro"
CALLBACK_CONFIRMATION: Callable[[str, str], bool] | None = None


def _resoudre_chemin(chemin: str) -> Path:
    p = Path(chemin).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    else:
        p = p.resolve()

    if RACINE_AUTORISEE not in p.parents and p != RACINE_AUTORISEE:
        raise PermissionError(f"Accès refusé hors de {RACINE_AUTORISEE} : {p}")
    return p


def _ecrire_atomique(cible: Path, contenu: str) -> None:
    tmp = cible.with_suffix(cible.suffix + ".tmp")
    tmp.write_text(contenu, encoding="utf-8")
    os.replace(tmp, cible)


def _verifier_autorisation_ecriture(action: str, detail: str) -> tuple[bool, str]:
    if MODE_FICHIERS == "ro":
        return False, "REFUS SYSTÈME : Le mode fichiers est en lecture seule (/files ro)."
    if MODE_FICHIERS == "auto":
        return True, ""
    if CALLBACK_CONFIRMATION and CALLBACK_CONFIRMATION(action, detail):
        return True, ""
    return False, "REFUS UTILISATEUR : L'utilisateur a refusé la modification du fichier."


def lister_fichiers(dossier: str = ".", max_profondeur: int = 3) -> str:
    """Liste les fichiers et sous-dossiers d'un répertoire de projet.

    Args:
        dossier: Chemin du dossier à explorer (par défaut '.').
        max_profondeur: Profondeur maximale de parcours (1 à 5).

    Returns:
        L'arborescence textuelle des fichiers trouvés.
    """
    try:
        racine = _resoudre_chemin(dossier)
        if not racine.exists() or not racine.is_dir():
            return f"Dossier introuvable : {dossier}"

        panels.log_outil(f"[FS] Liste : {racine}")
        prof_max = max(1, min(int(max_profondeur), 5))
        lignes = [f"Contenu de {racine} :"]

        def parcourir(courant: Path, niveau: int) -> None:
            if niveau > prof_max or len(lignes) > 150:
                return
            try:
                entrees = sorted(courant.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
            except PermissionError:
                return
            for entree in entrees:
                if entree.name in DOSSIERS_IGNORES:
                    continue
                indent = "  " * niveau
                if entree.is_dir():
                    lignes.append(f"{indent}[DIR] {entree.name}/")
                    parcourir(entree, niveau + 1)
                else:
                    taille_ko = entree.stat().st_size / 1024
                    lignes.append(f"{indent}{entree.name} ({taille_ko:.1f} Ko)")

        parcourir(racine, 1)
        return "\n".join(lignes)
    except Exception as e:
        return f"Erreur lister_fichiers : {e}"


def lire_fichier(chemin: str, ligne_debut: int = 1, ligne_fin: int = 250) -> str:
    """Lit le contenu d'un fichier texte ou code source avec les numéros de lignes.

    Args:
        chemin: Chemin relatif ou absolu du fichier à lire.
        ligne_debut: Numéro de la première ligne à lire (commence à 1).
        ligne_fin: Numéro de la dernière ligne à lire (max 300 lignes par appel).

    Returns:
        Le contenu numéroté du fichier.
    """
    try:
        cible = _resoudre_chemin(chemin)
        if not cible.exists() or not cible.is_file():
            return f"Fichier introuvable : {chemin}"

        lignes = cible.read_text(encoding="utf-8", errors="replace").splitlines()
        total = len(lignes)
        debut = max(1, int(ligne_debut))
        fin = min(total, max(debut, min(int(ligne_fin), debut + 300)))
        panels.log_outil(f"[FS] Lecture : {cible.name} ({debut}-{fin}/{total})")

        extrait = [f"{i:4d} | {lignes[i - 1]}" for i in range(debut, fin + 1)]
        en_tete = f"=== {cible} (lignes {debut}-{fin} sur {total}) ==="
        return en_tete + "\n" + "\n".join(extrait)
    except Exception as e:
        return f"Erreur lire_fichier : {e}"


def chercher_dans_fichiers(motif: str, dossier: str = ".") -> str:
    """Recherche une chaîne de texte (insensible à la casse) dans tous les fichiers d'un dossier.

    Args:
        motif: Texte, nom de fonction ou variable à rechercher.
        dossier: Dossier racine de recherche.

    Returns:
        Les occurrences trouvées avec fichier et numéro de ligne.
    """
    try:
        racine = _resoudre_chemin(dossier)
        if not racine.is_dir():
            return f"Dossier introuvable : {dossier}"

        panels.log_outil(f"[FS] Recherche '{motif}' dans {racine}")
        motif_min = motif.lower()
        resultats = []

        for chemin in racine.rglob("*"):
            if any(part in DOSSIERS_IGNORES for part in chemin.parts) or not chemin.is_file():
                continue
            if chemin.stat().st_size > 512_000:
                continue
            try:
                for idx, ligne in enumerate(chemin.read_text(encoding="utf-8").splitlines(), 1):
                    if motif_min in ligne.lower():
                        rel = chemin.relative_to(racine)
                        resultats.append(f"{rel}:{idx}: {ligne.strip()[:120]}")
                        if len(resultats) >= 50:
                            return "\n".join(resultats) + "\n... (tronqué à 50 résultats)"
            except (UnicodeDecodeError, PermissionError):
                continue

        return "\n".join(resultats) if resultats else f"Aucune occurrence de '{motif}' dans {racine}."
    except Exception as e:
        return f"Erreur chercher_dans_fichiers : {e}"


def modifier_fichier(chemin: str, ancien_texte: str, nouveau_texte: str) -> str:
    """Remplace un bloc de code exact dans un fichier existant après validation utilisateur.

    Args:
        chemin: Chemin du fichier à modifier.
        ancien_texte: Le bloc de texte exact actuellement présent dans le fichier.
        nouveau_texte: Le nouveau bloc de texte de remplacement.

    Returns:
        Confirmation de la modification ou message d'erreur.
    """
    try:
        cible = _resoudre_chemin(chemin)
        if not cible.is_file():
            return f"Fichier introuvable : {chemin}"

        contenu_actuel = cible.read_text(encoding="utf-8")
        if ancien_texte not in contenu_actuel:
            return "Erreur : `ancien_texte` introuvable tel quel dans le fichier. Lis d'abord le fichier avec `lire_fichier`."
        if contenu_actuel.count(ancien_texte) > 1:
            return "Erreur : `ancien_texte` apparaît plusieurs fois. Fournis plus de lignes de contexte."

        nouveau_contenu = contenu_actuel.replace(ancien_texte, nouveau_texte, 1)
        diff = "\n".join(
            difflib.unified_diff(
                contenu_actuel.splitlines(),
                nouveau_contenu.splitlines(),
                fromfile=f"a/{cible.name}",
                tofile=f"b/{cible.name}",
                lineterm="",
            )
        )

        ok, msg_refus = _verifier_autorisation_ecriture(f"Modifier {cible}", diff)
        if not ok:
            return msg_refus

        cible.with_suffix(cible.suffix + ".bak").write_text(contenu_actuel, encoding="utf-8")
        _ecrire_atomique(cible, nouveau_contenu)
        panels.log_outil(f"[FS] Modifié : {cible}")
        return f"Fichier modifié avec succès : {cible} (sauvegarde .bak créée)."
    except Exception as e:
        return f"Erreur modifier_fichier : {e}"


def ecrire_fichier(chemin: str, contenu: str) -> str:
    """Crée un nouveau fichier ou remplace entièrement un fichier existant après validation utilisateur.

    Args:
        chemin: Chemin du fichier à créer ou écraser.
        contenu: Contenu complet à écrire dans le fichier.

    Returns:
        Confirmation d'écriture ou refus.
    """
    try:
        cible = _resoudre_chemin(chemin)
        existe = cible.exists()
        apercu = f"{'Écrasement' if existe else 'Création'} de {cible} ({len(contenu)} caractères)\n---\n{contenu[:600]}"

        ok, msg_refus = _verifier_autorisation_ecriture(f"Écrire {cible}", apercu)
        if not ok:
            return msg_refus

        cible.parent.mkdir(parents=True, exist_ok=True)
        if existe:
            cible.with_suffix(cible.suffix + ".bak").write_text(cible.read_text(encoding="utf-8"), encoding="utf-8")
        _ecrire_atomique(cible, contenu)
        panels.log_outil(f"[FS] Écrit : {cible}")
        return f"Fichier écrit avec succès : {cible}"
    except Exception as e:
        return f"Erreur ecrire_fichier : {e}"