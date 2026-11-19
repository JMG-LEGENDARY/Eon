import html
import re
import urllib.parse
import urllib.request
from ddgs import DDGS

import terminal_panels as panels


def _recherche_ddg_html_fallback(requete: str, max_res: int) -> str:
    url = "[https://lite.duckduckgo.com/lite/](https://lite.duckduckgo.com/lite/)"
    data = urllib.parse.urlencode({"q": requete, "kl": "fr-fr"}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) LocalAssistant/1.0"},
    )
    with urllib.request.urlopen(req, timeout=6) as resp:
        page = resp.read().decode("utf-8", errors="ignore")

    liens = re.findall(r'<a[^>]+class="result-link"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', page, flags=re.S)
    extraits = re.findall(r'<td[^>]+class="result-snippet"[^>]*>(.*?)</td>', page, flags=re.S)

    if not liens:
        return ""

    blocs = []
    for idx, (href, titre_raw) in enumerate(liens[:max_res], 1):
        titre = re.sub(r"<[^>]+>", "", html.unescape(titre_raw)).strip()
        snippet = re.sub(r"<[^>]+>", "", html.unescape(extraits[idx - 1])).strip() if idx - 1 < len(extraits) else ""
        blocs.append(f"[{idx}] {titre}\nURL: {href}\nRésumé: {snippet}")
    return "\n\n".join(blocs)


def rechercher_sur_internet(requete: str, max_resultats: int = 5) -> str:
    """Recherche des informations récentes ou factuelles sur Internet.

    Args:
        requete: Les mots-clés de recherche précis.
        max_resultats: Nombre maximum de résultats à retourner (1 à 8).

    Returns:
        Les titres, URLs et extraits des résultats trouvés.
    """
    max_res = max(1, min(int(max_resultats), 8))
    panels.log_outil(f"[WEB] Recherche : {requete}")
    try:
        with DDGS(timeout=6) as ddgs:
            resultats = list(ddgs.text(requete, region="fr-fr", max_results=max_res))
        if resultats:
            lignes = []
            for idx, r in enumerate(resultats, 1):
                lignes.append(
                    f"[{idx}] {r.get('title', 'Sans titre')}\n"
                    f"URL: {r.get('href', '')}\n"
                    f"Résumé: {r.get('body', '')}"
                )
            return "\n\n".join(lignes)
    except Exception:
        pass

    try:
        fallback = _recherche_ddg_html_fallback(requete, max_res)
        if fallback:
            return fallback
    except Exception as e:
        return (
            f"ERREUR_RECHERCHE_WEB ({e}) : Impossible de joindre Internet. "
            "Dis clairement à l'utilisateur que la recherche en ligne a échoué et NE DONNE AUCUNE information non vérifiée."
        )

    return "Aucun résultat trouvé sur Internet pour cette requête."


def lire_page_web(url: str) -> str:
    """Télécharge et extrait le texte principal d'une page web à partir de son URL.

    Args:
        url: L'adresse HTTP ou HTTPS complète de la page à lire.

    Returns:
        Le contenu textuel nettoyé de la page (tronqué à 4000 caractères).
    """
    if not url.startswith(("http://", "https://")):
        return "URL invalide : doit commencer par http:// ou https://"

    panels.log_outil(f"[WEB] Lecture page : {url}")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) LocalAssistant/1.0"},
        )
        with urllib.request.urlopen(req, timeout=7) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")

        # Supprime scripts, styles, menus de navigation et pieds de page avant l'extraction
        nettoye = re.sub(
            r"<(script|style|nav|header|footer|aside|noscript)[^>]*>.*?</\1>",
            " ",
            raw,
            flags=re.DOTALL | re.IGNORECASE,
        )
        texte = re.sub(r"<[^>]+>", " ", nettoye)
        texte = html.unescape(texte)
        texte = re.sub(r"\s+", " ", texte).strip()
        return texte[:4000] if texte else "Page vide ou illisible."
    except Exception as e:
        return f"Impossible de lire la page ({url}) : {e}"