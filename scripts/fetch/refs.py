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
    "semenova2020leebounds": "2008.12720",
    "tchetgen2023udid": "2302.00840",
    "steinbuss2020outlier": "2004.06947",
    "emmott2015anomaly": "1503.01158",
    "goswami2022selection": "2210.01078",
    "crawford2022erclaims": "2206.08093",
    "shekhar2022fraud": "2211.02927",
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
    "ramalho2011fractional": "10.1111/j.1467-6419.2009.00602.x",
    "cameron2011multiway": "10.1198/jbes.2010.07136",
    "guimaraes2010hdfe": "10.1177/1536867X1001000406",
    "goldstein1996league": "10.2307/2983325",
    "cinelli2020sensitivity": "10.1111/rssb.12348",
    "lipsitch2010negative": "10.1097/EDE.0b013e3181d61eeb",
    "schuemie2014calibration": "10.1002/sim.5925",
    "kalbfleisch2013monitoring": "10.1007/s12561-013-9093-x",
    "spiegelhalter2005funnel": "10.1002/sim.1970",
    "forbes2015disclosure": "10.1257/mic.20130164",
    "sofer2016negative": "10.1214/16-sts558",
    "lee2009bounds": "10.1111/j.1467-937x.2009.00536.x",
    "frangakis2002principal": "10.1111/j.0006-341x.2002.00021.x",
    "knox2020administrative": "10.1017/s0003055420000039",
    "schennach2016measurement": "10.1146/annurev-economics-080315-015058",
    "rosner1992measurement": "10.1093/oxfordjournals.aje.a116453",
    "steiner2000cusum": "10.1093/biostatistics/1.4.441",
    "spiegelhalter2012regulation": "10.1111/j.1467-985x.2011.01010.x",
    "gnutzmann2023rights": "10.1016/j.tranpol.2023.03.017",
    "fukui2014tarmac": "10.1016/j.ecotra.2014.02.004",
    "fukui2026penalties": "10.1016/j.tra.2026.105063",
    "aggarwal2019incentives": "10.1093/rcfs/cfz002",
    "bilotkach2022causes": "10.3828/jtep.2022.56.2.103",
    "borsky2019weather": "10.1016/j.ecotra.2019.02.002",
    "wedderburn1974quasi": "10.1093/biomet/61.3.439",
    "mccullagh1983quasi": "10.1214/aos/1176346056",
    "gourieroux1984pml": "10.2307/1913471",
    "santossilva2006log": "10.1162/rest.88.4.641",
    "allison2002negbin": "10.1111/1467-9531.00117",
    "bosquet2014scale": "10.1080/07474938.2013.806102",
    "blackburn2015relative": "10.1111/obes.12074",
    "santossilva2022log15": "10.1007/s10258-021-00203-w",
    "zitzewitz2012forensic": "10.1257/jel.50.3.731",
    "bolton2002fraud": "10.1214/ss/1042727940",
    "zinman2016wintertime": "10.1257/app.20130346",
    "parsons2011strike": "10.1257/aer.101.4.1410",
    "pierce2008spillovers": "10.1287/mnsc.1080.0927",
    "jacob2003rotten": "10.1162/00335530360698441",
    "geruso2020upcoding": "10.1086/704756",
    "sanghavi2020falls": "10.1111/1475-6773.13247",
    "dawid1979observer": "10.2307/2346806",
    "aelvoet2009miscoding": "10.1016/j.respe.2009.02.206",
    "schuemie2018calibration": "10.1073/pnas.1708282114",
    "brakenhoff2018profiling": "10.1186/s12874-018-0510-1",
    "ding2013profiling": "10.1007/s10742-012-0099-5",
    "eccleston1974connected": "10.1214/aos/1176342876",
    "weeks1964connectedness": "10.1080/00401706.1964.10490188",
}

# 法規與政府文件不在 arXiv 與 Crossref 內，書目取自 eCFR、Federal Register 與機關網頁本身。
OFFICIAL = """cfr234:
  type: legislation
  title: "14 CFR § 234.4"
  date: 2026
  url: https://www.ecfr.gov/current/title-14/section-234.4
bts2024causes:
  type: web
  title: "Understanding the Reporting of Causes of Flight Delays and Cancellations"
  author: {name: "Bureau of Transportation Statistics"}
  url: https://www.bts.gov/topics/airlines-and-airports/understanding-reporting-causes-flight-delays-and-cancellations
btsdirective14:
  type: web
  title: "Technical Directive Number 14: On-Time Reporting"
  author: {name: "Bureau of Transportation Statistics"}
  url: https://www.bts.gov/topics/airlines-and-airports/number-14-time-reporting
dot2025commitments:
  type: report
  title: "Airline Customer Service Commitments for Controllable Flight Disruptions: Unscheduled Maintenance in Response to FAA Airworthiness Directives"
  author: {name: "U.S. Department of Transportation"}
  date: 2025-12-10
  url: https://www.federalregister.gov/documents/2025/12/10/2025-22415
  parent:
    type: periodical
    title: Federal Register
    volume: 90
dot2026causes:
  type: report
  title: "Cause of Airline Delay and Cancellation Categories Under Section 511(b) of the FAA Reauthorization Act of 2024"
  author: {name: "U.S. Department of Transportation"}
  date: 2026-09-03
  url: https://www.federalregister.gov/documents/2026/09/03/2026-18040
  parent:
    type: periodical
    title: Federal Register
    volume: 91"""

# Crossref 對 Steiner 一篇只登錄第一作者，其餘三位依 Europe PMC 的紀錄補上；
# 對 Santos Silva 一篇把複姓的前半切進名，依論文署名改回。
AUTHORS = {
    "steiner2000cusum": ["Steiner, S. H.", "Cook, R. J.", "Farewell, V. T.", "Treasure, T."],
    "santossilva2006log": ["Santos Silva, J. M. C.", "Tenreyro, Silvana"],
}

# Crossref 缺頁碼的紀錄，依 OpenAlex 補上；McCullagh 一篇 OpenAlex 亦缺，依 Semantic Scholar。
# Bolton 一篇 OpenAlex 給的是文末 rejoinder 的頁碼，依 Zitzewitz (2012) 的書目。
# Dawid 一篇 Crossref 只登錄首頁，依論文本身。Eccleston 一篇 Crossref 與 OpenAlex 皆缺，依 Semantic Scholar。
PAGES = {"sofer2016negative": "348-361", "mccullagh1983quasi": "59-67", "bolton2002fraud": "235-249",
         "dawid1979observer": "20-28", "eccleston1974connected": "1238-1255"}

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
    authors = AUTHORS.get(key) or [
        ", ".join(x.title() if x.isupper() else x for x in (a["family"], a["given"])) for a in m["author"]
    ]
    journal = html.unescape(m["container-title"][0])
    # Crossref 的舊紀錄有全大寫的標題。
    title = m["title"][0].capitalize() if m["title"][0].isupper() else m["title"][0]
    # issued 取線上與紙本較早者，與慣用的引用年份不一定相同。
    year = m.get("published-print", m["issued"])["date-parts"][0][0]
    print(doi, "|", m["title"][0], "|", authors, "|", journal, year)
    out += [
        f"{key}:",
        "  type: article",
        # Sociological Methodology 的紀錄把章號寫進標題。
        f"  title: {q(re.sub(r'<[^>]+>|^\d+\. ', '', title), ensure_ascii=False)}",
        f"  author: {q(authors, ensure_ascii=False)}",
        f"  date: {year}",
        f"  page-range: {q(PAGES.get(key) or m['page'])}",
        "  serial-number:",
        f"    doi: {q(doi)}",
        "  parent:",
        "    type: periodical",
        f"    title: {q(journal, ensure_ascii=False)}",
        f"    volume: {m['volume']}",
        f"    issue: {q(m.get('issue', ''))}",
    ]

out.append(OFFICIAL)
open("report/refs.yml", "w").write("\n".join(out) + "\n")
