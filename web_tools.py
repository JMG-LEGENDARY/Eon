from datetime import datetime
import html
import re
import urllib.error
import urllib.parse
import urllib.request
from ddgs import DDGS

import terminal_panels as panels

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64; rv:132.0) Gecko/20100101 Firefox/132.0"
)


def _extraire_url_propre(url_brute: str) -> str:
    """Extrait une URL HTTP(S) valide même si le LLM l'entoure de crochets Markdown ou de guillemets."""
    if not url_brute:
        return ""
    match = re.search(r"https?://[^\s\]\)>\"']+", url_brute.strip())
    return match.group(0).rstrip(".,;:") if match else url_brute.strip()


def _ouvrir_url(req: urllib.request.Request, timeout: int = 8):
    """Ouvre une requête HTTP en contournant d'éventuels proxies d'environnement malformés."""
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.URLError as err:
        if "unknown url type" in str(err).lower():
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            return opener.open(req, timeout=timeout)
        raise


def _nettoyer_html(texte_brut: str) -> str:
    sans_commentaires = re.sub(r"<!--.*?-->", " ", texte_brut, flags=re.S)
    sans_blocs = re.sub(
        r"<(script|style|nav|header|footer|aside|noscript|svg|iframe|form)[^>]*>.*?</\1>",
        " ",
        sans_commentaires,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Privilégie le contenu principal (<article> ou <main>) s'il est suffisamment riche
    zone_principale = sans_blocs
    for balise in ("article", "main"):
        match = re.search(rf"<{balise}\b[^>]*>(.*?)</{balise}>", sans_blocs, flags=re.DOTALL | re.IGNORECASE)
        if match:
            extrait_test = re.sub(r"<[^>]+>", " ", match.group(1))
            if len(re.sub(r"\s+", " ", extrait_test).strip()) >= 350:
                zone_principale = match.group(1)
                break

    texte = re.sub(r"<[^>]+>", " ", zone_principale)
    texte = html.unescape(texte)
    return re.sub(r"\s+", " ", texte).strip()


def _recherche_ddg_html_fallback(requete: str, max_res: int) -> str:
    url = "https://lite.duckduckgo.com/lite/"
    data = urllib.parse.urlencode({"q": requete, "kl": "fr-fr"}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "User-Agent": USER_AGENT,
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    with _ouvrir_url(req, timeout=8) as resp:
        page = resp.read().decode("utf-8", errors="ignore")

    liens = re.findall(
        r"<a[^>]+class=['\"]result-link['\"][^>]*href=['\"]([^'\"]+)['\"][^>]*>(.*?)</a>",
        page,
        flags=re.S | re.I,
    )
    extraits = re.findall(
        r"<td[^>]+class=['\"]result-snippet['\"][^>]*>(.*?)</td>",
        page,
        flags=re.S | re.I,
    )

    if not liens:
        return ""

    blocs = []
    for idx, (href, titre_raw) in enumerate(liens[:max_res], 1):
        if "uddg=" in href:
            parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
            href = parsed.get("uddg", [href])[0]

        href = _extraire_url_propre(href)
        titre = re.sub(r"<[^>]+>", "", html.unescape(titre_raw)).strip()
        snippet = (
            re.sub(r"<[^>]+>", "", html.unescape(extraits[idx - 1])).strip()
            if idx - 1 < len(extraits)
            else ""
        )
        blocs.append(f"[{idx}] {titre}\nURL: {href}\nRésumé: {snippet}")
    return "\n\n".join(blocs)


def rechercher_sur_internet(
    requete: str,
    max_resultats: int = 5,
    actualites: bool = False,
) -> str:
    """Recherche des informations récentes ou factuelles sur Internet.

    Args:
        requete: Les mots-clés de recherche précis (évite les phrases entières).
        max_resultats: Nombre maximum de résultats à retourner (4 à 8).
        actualites: Mettre à True pour inclure la presse et les actualités récentes.

    Returns:
        Les titres, dates, URLs et extraits des résultats trouvés.
    """
    req_propre = requete.strip().strip("\"'[]")
    if not req_propre:
        return "Erreur : requête de recherche vide."

    # Garantit au moins 4 résultats pour éviter qu'un seul résultat obsolète ne biaise le modèle
    max_res = max(4, min(int(max_resultats), 8))
    date_jour = datetime.now().strftime("%d/%m/%Y")

    panels.log_outil(f"[WEB] Recherche ({'NEWS+WEB' if actualites else 'WEB'}) : {req_propre}")
    erreurs: list[str] = []
    resultats_fusionnes: list[dict] = []
    urls_vues: set[str] = set()

    def _ajouter_resultats(items: list[dict]) -> None:
        for r in items:
            u = _extraire_url_propre(r.get("href") or r.get("url", ""))
            if not u or u in urls_vues:
                continue
            urls_vues.add(u)
            r_copie = dict(r)
            r_copie["_url_propre"] = u
            resultats_fusionnes.append(r_copie)
            if len(resultats_fusionnes) >= max_res:
                break

    try:
        with DDGS(timeout=8) as ddgs:
            if actualites:
                try:
                    news_res = list(ddgs.news(req_propre, region="fr-fr", timelimit="m", max_results=max_res))
                    _ajouter_resultats(news_res)
                except Exception as e_news:
                    erreurs.append(f"News: {e_news}")

            # Complète systématiquement avec la recherche web classique pour les fiches factuelles
            if len(resultats_fusionnes) < max_res:
                text_res = list(ddgs.text(req_propre, region="fr-fr", max_results=max_res))
                _ajouter_resultats(text_res)

        if resultats_fusionnes:
            lignes = [f"Date du jour : {date_jour} | Requête exécutée : « {req_propre} »\n"]
            for idx, r in enumerate(resultats_fusionnes[:max_res], 1):
                titre = r.get("title", "Sans titre")
                url = r["_url_propre"]
                corps = r.get("body", "")
                date_art = r.get("date", "")
                source = r.get("source", "")

                entete = f"[{idx}] {titre}"
                if source or date_art:
                    entete += f" ({source} - {date_art})".replace(" - )", ")").replace("( - ", "(")

                lignes.append(f"{entete}\nURL: {url}\nRésumé: {corps}")

            lignes.append(
                "\nCONSIGNE : Base ta réponse EXCLUSIVEMENT sur les faits ci-dessus (ils priment sur ta mémoire interne). "
                "Si un extrait est trop court ou est une page d'accueil générique, appelle `lire_page_web(url)` "
                "sur la meilleure URL avant de répondre."
            )
            return "\n\n".join(lignes)
    except Exception as e:
        erreurs.append(f"DDGS: {e}")

    try:
        fallback = _recherche_ddg_html_fallback(req_propre, max_res)
        if fallback:
            return (
                f"Date du jour : {date_jour} | Requête exécutée : « {req_propre} »\n\n"
                f"{fallback}\n\n"
                "CONSIGNE : Base ta réponse sur ces résultats ou appelle `lire_page_web(url)` pour plus de détails."
            )
    except Exception as e:
        erreurs.append(f"Fallback: {e}")

    if erreurs:
        return (
            f"ERREUR_RECHERCHE_WEB ({' | '.join(erreurs)}) : Impossible de joindre Internet. "
            "Signale clairement à l'utilisateur que la recherche en ligne a échoué et NE DONNE AUCUNE information non vérifiée."
        )

    return (
        f"Aucun résultat trouvé sur Internet pour « {req_propre} ». "
        "Essaie une requête plus courte ou indique qu'aucune information récente n'a été trouvée."
    )


def lire_page_web(url: str) -> str:
    """Télécharge et extrait le texte principal d'une page web à partir de son URL.

    Args:
        url: L'adresse HTTP ou HTTPS complète de la page à lire.

    Returns:
        Le contenu textuel nettoyé de la page (tronqué à 5500 caractères).
    """
    url_propre = _extraire_url_propre(url)
    if not url_propre.startswith(("http://", "https://")):
        return f"URL invalide ({url}) : doit commencer par http:// ou https://"

    panels.log_outil(f"[WEB] Lecture page : {url_propre}")
    try:
        req = urllib.request.Request(
            url_propre,
            headers={
                "User-Agent": USER_AGENT,
                "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
            },
        )
        with _ouvrir_url(req, timeout=8) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            raw = resp.read().decode(charset, errors="ignore")

        texte = _nettoyer_html(raw)
        if not texte or len(texte) < 100:
            return f"Page vide ou protégée contre le scraping ({url_propre}). Essaie une autre URL des résultats."

        extrait = texte[:5500]
        if len(texte) > 5500:
            extrait = extrait.rsplit(" ", 1)[0] + " [...]"

        return f"Contenu extrait de {url_propre} :\n\n{extrait}"
    except urllib.error.HTTPError as e:
        return f"Erreur HTTP {e.code} lors de la lecture de {url_propre} ({e.reason})."
    except Exception as e:
        return f"Impossible de lire la page ({url_propre}) : {e}"