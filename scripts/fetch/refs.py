import html
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ARXIV = {
    "chen2023lpm": "2308.15338",
    "correia2019ppmlhdfe": "1903.01690",
    "correia2019existence": "1903.01633",
    "lee2017poisson": "1707.08538",
    "weidner2019bias": "1909.01327",
    "egger2026heavy": "2609.18750",
    "ospina2011inflated": "1103.2372",
    "alzeley2026zeros": "2608.29954",
    "koslovsky2023zidm": "2302.12283",
    "hoffman2011nuts": "1111.4246",
    "betancourt2017hmc": "1701.02434",
    "vehtari2019rhat": "1903.08008",
    "vehtari2015loo": "1507.04544",
    "margossian2021nested": "2110.13017",
    "grun2024mixture": "2407.05470",
    "abadie2017cluster": "1710.02926",
    "mackinnon2022guide": "2205.03285",
    "mackinnon2024twoway": "2406.08880",
    "mackinnon2024logit": "2406.00650",
    "chetverikov2024csranks": "2401.15205",
    "almohamad2017ranks": "1708.02729",
    "gu2020invidious": "2012.12550",
    "kline2023reportcard": "2306.13005",
    "bazylik2024multinomial": "2402.00192",
    "chandra2025ranks": "2512.00316",
    "strezhnev2026ovb": "2607.13334",
    "wainstein2025wls": "2508.02954",
    "bach2025sensitivity": "2510.09109",
    "morris2017simulation": "1712.03198",
    "schreck2023plasmode": "2305.06028",
    "shaw2025plasmode": "2504.11740",
    "karim2026plasmode": "2609.26186",
    "fernandezval2013panel": "1311.7065",
    "shi2020negative": "2009.05641",
    "hwang2021calibration": "2111.04233",
    "normand2007profiling": "0710.4622",
    "xia2019profiling": "1907.07809",
    "hartman2022null": "2207.07602",
    "efron2007size": "0710.2245",
    "chetverikov2012monotonicity": "1212.6757",
    "espinosa2021ordinal": "2107.04946",
    "ghosh2020backfitting": "2007.10612",
    "papaspiliopoulos2021crossed": "2103.10875",
}

DOI = {
    "baker1994multinomial": "10.2307/2348134",
    "firth1993bias": "10.1093/biomet/80.1.27",
    "efron1973stein": "10.2307/2284155",
    "benjamini1995fdr": "10.1111/j.2517-6161.1995.tb02031.x",
    "stephens2000label": "10.1111/1467-9868.00265",
    "mundlak1978pooling": "10.2307/1913646",
    "chamberlain1980covariance": "10.2307/2297110",
    "papke1996fractional": "10.1002/(SICI)1099-1255(199611)11:6<619::AID-JAE418>3.0.CO;2-1",
    "cameron2011multiway": "10.1198/jbes.2010.07136",
    "guimaraes2010hdfe": "10.1177/1536867X1001000406",
    "goldstein1996league": "10.2307/2983325",
    "cinelli2020sensitivity": "10.1111/rssb.12348",
    "lipsitch2010negative": "10.1097/EDE.0b013e3181d61eeb",
    "schuemie2014calibration": "10.1002/sim.5925",
    "kalbfleisch2013monitoring": "10.1007/s12561-013-9093-x",
    "spiegelhalter2005funnel": "10.1002/sim.1970",
    "forbes2015disclosure": "10.1257/mic.20130164",
}

# arXiv 的作者是「名 姓」單一字串，以最後一個空白切分會把複姓切錯。
FAMILY = {"Diaa Al Mohamad": "Al Mohamad", "Erik W. van Zwet": "van Zwet", "Mark J. van der Laan": "van der Laan",
          "Eric Tchetgen Tchetgen": "Tchetgen Tchetgen"}

# hayagriva 把標題內的 $...$ 當 Typst 數學式解析，LaTeX 指令會編譯失敗。
TITLE = {"$\\widehat{R}$": "R-hat", "$\\hat R$": "R-hat"}

q = json.dumps
out = []

ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
url = "http://export.arxiv.org/api/query?max_results=100&id_list=" + ",".join(ARXIV.values())
feed = ET.fromstring(urllib.request.urlopen(url, timeout=60).read())
entries = {e.findtext("a:id", "", ns).rsplit("/", 1)[1].split("v")[0]: e for e in feed.findall("a:entry", ns)}
for key, aid in ARXIV.items():
    e = entries[aid]
    title = " ".join(e.findtext("a:title", "", ns).split())
    for k, v in TITLE.items():
        title = title.replace(k, v)
    authors = []
    for name in ("".join(a.itertext()) for a in e.findall("a:author/a:name", ns)):
        fam = FAMILY.get(name) or name.rsplit(" ", 1)[1]
        authors.append(f"{fam}, {name[: -len(fam)].strip()}")
    jref = e.findtext("x:journal_ref", None, ns)
    print(aid, "|", title, "|", authors, "|", jref)
    out += [
        f"{key}:",
        "  type: article",
        f"  title: {q(title, ensure_ascii=False)}",
        f"  author: {q(authors, ensure_ascii=False)}",
        f"  date: {e.findtext('a:published', '', ns)[:4]}",
        f"  url: https://arxiv.org/abs/{aid}",
        "  parent:",
        "    type: repository",
        f"    title: {q('arXiv:' + aid if jref is None else ' '.join(jref.split()), ensure_ascii=False)}",
    ]

for key, doi in DOI.items():
    # Crossref 對連續請求回 429。
    time.sleep(2)
    req = urllib.request.Request(
        "https://api.crossref.org/works/" + urllib.parse.quote(doi), headers={"User-Agent": "refs/0.1"}
    )
    m = json.load(urllib.request.urlopen(req, timeout=60))["message"]
    # Crossref 的舊紀錄有全大寫的姓名。
    authors = [", ".join(x.title() if x.isupper() else x for x in (a["family"], a["given"])) for a in m["author"]]
    journal = html.unescape(m["container-title"][0])
    # issued 取線上與紙本較早者，與慣用的引用年份不一定相同。
    year = m.get("published-print", m["issued"])["date-parts"][0][0]
    print(doi, "|", m["title"][0], "|", authors, "|", journal, year)
    out += [
        f"{key}:",
        "  type: article",
        f"  title: {q(re.sub('<[^>]+>', '', m['title'][0]), ensure_ascii=False)}",
        f"  author: {q(authors, ensure_ascii=False)}",
        f"  date: {year}",
        f"  page-range: {q(m['page'])}",
        "  serial-number:",
        f"    doi: {q(doi)}",
        "  parent:",
        "    type: periodical",
        f"    title: {q(journal, ensure_ascii=False)}",
        f"    volume: {m['volume']}",
        f"    issue: {q(m.get('issue', ''))}",
    ]

open("report/refs.yml", "w").write("\n".join(out) + "\n")
