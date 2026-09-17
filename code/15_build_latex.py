"""Step 15 - convert the Markdown manuscript sections into LaTeX fragments.

The preamble and document skeleton live in `tex/manuscript.tex`, which is kept
under hand control; this script only generates the body fragments
(`tex/body_*.tex`) and the bibliography (`tex/refs.tex`), so that the prose in
`manuscript/*.md` remains the single source of truth.

Conversion covers exactly the constructs the manuscript uses: ATX headings,
paragraphs, **bold** (figure legends only), author-year citations mapped to
natbib keys, and the Unicode symbols that appear in the text.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import LOGS, ROOT, Tee

MS = ROOT / "manuscript"
TEX = ROOT / "tex"
TEX.mkdir(exist_ok=True)
log = Tee(LOGS / "15_build_latex.txt")

# ------------------------------------------------------------ citations ----
# Longest strings first so that shorter ones cannot fire inside them.
CITES = [
    # textual
    ("Aitchison (1982, 1986)", r"\citet{aitchison1982,aitchison1986}"),
    ("Hendrycks and Gimpel (2017)", r"\citet{hendrycks2017}"),
    ("Chow (1970)", r"\citet{chow1970}"),
    ("Hein (2026)", r"\citet{hein2026}"),
    ("Hein et al. (2021a)", r"\citet{hein2021}"),
    ("Hein et al. (2021b)", r"\citet{heindata2021}"),
    ("Kaufman et al. (2012)", r"\citet{kaufman2012}"),
    ("Roberts et al. (2017)", r"\citet{roberts2017}"),
    ("Vovk (2015)", r"\citet{vovk2015}"),
    ("UNESCO World Heritage Centre (2026)", r"\citet{unesco2026}"),
    # parenthetical, multi-key (longest first to prevent partial matches)
    ("(Harbottle 1976; Glascock and Neff 2003; Tite 2008)",
     r"\citep{harbottle1976,glascock2003,tite2008}"),
    ("(Shackley 2011; Frahm and Doonan 2013; Hunt and Speakman 2015)",
     r"\citep{shackley2011,frahm2013,hunt2015}"),
    ("(Beier and Mommsen 1994; Baxter and Freestone 2006; Hein 2026)",
     r"\citep{beier1994,baxter2006,hein2026}"),
    ("(Vovk et al. 2005; Shafer and Vovk 2008; Angelopoulos and Bates 2023)",
     r"\citep{vovk2005,shafer2008,angelopoulos2023}"),
    ("(Buxeda i Garrigos 1999; Frahm and Doonan 2013)",
     r"\citep{buxeda1999,frahm2013}"),
    ("(Vovk et al. 2005; Angelopoulos and Bates 2023)",
     r"\citep{vovk2005,angelopoulos2023}"),
    ("(Vovk et al. 2005; Shafer and Vovk 2008)", r"\citep{vovk2005,shafer2008}"),
    ("(Frahm and Doonan 2013; Speakman and Shackley 2013)",
     r"\citep{frahm2013,speakman2013}"),
    ("(Kaufman et al. 2012; Roberts et al. 2017)",
     r"\citep{kaufman2012,roberts2017}"),
    ("(Chow 1970; Geifman and El-Yaniv 2017)", r"\citep{chow1970,geifman2017}"),
    ("(Glascock and Neff 2003; Tite 2008)", r"\citep{glascock2003,tite2008}"),
    ("(Hendrycks and Gimpel 2017; Lee et al. 2018)",
     r"\citep{hendrycks2017,lee2018}"),
    ("(Peacock and Williams 1986; Whitbread 1995)",
     r"\citep{peacock1986,whitbread1995}"),
    ("(Hunt and Speakman 2015; Hein 2026)", r"\citep{hunt2015,hein2026}"),
    ("(Hein et al. 2021a; Hein 2026)", r"\citep{hein2021,hein2026}"),
    ("(Hein et al. 2021a, b)", r"\citep{hein2021,heindata2021}"),
    ("(Rice 1987; Whitbread 1995)", r"\citep{rice1987,whitbread1995}"),
    ("(Vovk 2015; Barber et al. 2021)", r"\citep{vovk2015,barber2021}"),
    ("(UNESCO World Heritage Centre 2026)", r"\citep{unesco2026}"),
    ("(Vovk et al. 2005; Vovk 2015)", r"\citep{vovk2005,vovk2015}"),
    # parenthetical, single key
    ("(XGBoost; Chen and Guestrin 2016)", r"(XGBoost; \citealp{chen2016})"),
    ("(Peacock and Williams 1986)", r"\citep{peacock1986}"),
    ("(Buxeda i Garrigos 1999)", r"\citep{buxeda1999}"),   # after unicode fold
    ("(Baxter and Freestone 2006)", r"\citep{baxter2006}"),
    ("(Hendrycks and Gimpel 2017)", r"\citep{hendrycks2017}"),
    ("(Zadrozny and Elkan 2002)", r"\citep{zadrozny2002}"),
    ("(Geifman and El-Yaniv 2017)", r"\citep{geifman2017}"),
    ("(Angelopoulos and Bates 2023)", r"\citep{angelopoulos2023}"),
    ("(Pedregosa et al. 2011)", r"\citep{pedregosa2011}"),
    ("(Scheirer et al. 2013)", r"\citep{scheirer2013}"),
    ("(Ledoit and Wolf 2004)", r"\citep{ledoit2004}"),
    ("(Shackley 2011)", r"\citep{shackley2011}"),
    ("(Harbottle 1976)", r"\citep{harbottle1976}"),
    ("(Beier and Mommsen 1994)", r"\citep{beier1994}"),
    ("(Guo et al. 2017)", r"\citep{guo2017}"),
    ("(Hein et al. 2021a)", r"\citep{hein2021}"),
    ("(Hein et al. 2021b)", r"\citep{heindata2021}"),
    ("(Lee et al. 2018)", r"\citep{lee2018}"),
    ("(Whitbread 1995)", r"\citep{whitbread1995}"),
    ("(Rice 1987)", r"\citep{rice1987}"),
    ("(Breiman 2001)", r"\citep{breiman2001}"),
    ("(Platt 1999)", r"\citep{platt1999}"),
    ("(Vovk 2015)", r"\citep{vovk2015}"),
    ("(Hein 2026)", r"\citep{hein2026}"),
]

# --------------------------------------------------------------- symbols ---
SYMBOLS = [
    ("\u03c4_p", r"$\tau_{\mathrm{p}}$"), ("\u03c4_d", r"$\tau_{\mathrm{d}}$"),
    ("10\u207b\u2074", r"$10^{-4}$"), ("10\u207b\u00b9\u2076", r"$10^{-16}$"),
    ("10\u207b\u00b2\u2078", r"$10^{-28}$"),
    ("\u221a2", r"$\sqrt{2}$"), ("\u221ap", r"$\sqrt{p}$"),
    ("\u03b3", r"$\gamma$"),
    ("\u03b1", r"$\alpha$"), ("\u03c1", r"$\rho$"),
    ("\u00d7", r"$\times$"), ("\u2212", r"$-$"),
    ("\u2013", "--"),
    ("\u201c", "``"), ("\u201d", "''"),
    ("\u00e8", r"\`{e}"), ("\u00f3", r"\'{o}"), ("\u00f6", r'\"{o}'),
    ("\u00e9", r"\'{e}"), ("\u2264", r"$\le$"), ("\u2265", r"$\ge$"),
    ("\u00b0", r"$^\circ$"), ("\u2032", r"$^\prime$"),
    ("\u2033", r"$^{\prime\prime}$"),
]
ESCAPE = [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
          ("#", r"\#"), ("_", r"\_"), ("~", r"\textasciitilde{}"),
          ("^", r"\textasciicircum{}")]

PLACE = "\u0001{}\u0002"


def convert(text: str) -> str:
    """Markdown paragraph text -> LaTeX, with placeholders around raw LaTeX."""
    slots: list[str] = []

    def stash(s: str) -> str:
        slots.append(s)
        return PLACE.format(len(slots) - 1)

    # 1. citations (Buxeda needs the accent folded before matching)
    text = text.replace("Buxeda i Garrig\u00f3s 1999", "Buxeda i Garrigos 1999")
    for src, dst in CITES:
        if src in text:
            text = text.replace(src, stash(dst))
    # 2. bold -> \textbf
    text = re.sub(r"\*\*(.+?)\*\*",
                  lambda m: stash(r"\textbf{") + m.group(1) + stash("}"), text)
    # 2b. ASCII quotation marks -> LaTeX opening/closing quotes. The Markdown
    # sources use straight quotes; left as-is they set as two upright marks.
    # Only balanced pairs are converted, and only the delimiters are stashed so
    # that the quoted text itself still passes through symbol and escape steps.
    text = re.sub(r'"([^"]*)"',
                  lambda m: stash("``") + m.group(1) + stash("''"), text)
    # 3. symbols
    for src, dst in SYMBOLS:
        if src in text:
            text = text.replace(src, stash(dst))
    # 4. escape the remainder
    for src, dst in ESCAPE:
        text = text.replace(src, dst)
    # 5. restore
    for i, s in enumerate(slots):
        text = text.replace(PLACE.format(i), s)
    return text


def md_to_tex(md: str, label_prefix: str, starred: bool = False) -> str:
    out, buf = [], []

    def flush():
        if buf:
            out.append(convert(" ".join(buf).strip()))
            out.append("")
            buf.clear()

    for line in md.split("\n"):
        if line.startswith("## ") or line.startswith("# "):
            flush()
            hashes, rest = line.split(" ", 1)
            num = re.match(r"^([\d.]+)\s+", rest)
            title = rest[num.end():].strip() if num else rest.strip()
            slug = re.sub(r"[^a-z0-9]+", "-", title.lower())[:20].strip("-")
            key = label_prefix + ("-" + num.group(1).replace(".", "-") if num
                                  else ("-" + slug if len(hashes) > 1 else ""))
            cmd = "subsection" if len(hashes) > 1 else "section"
            star = "*" if starred else ""
            out += [r"\%s%s{%s}\label{sec:%s}" % (cmd, star, convert(title), key),
                    ""]
        elif not line.strip():
            flush()
        else:
            buf.append(line.strip())
    flush()
    return "\n".join(out).strip() + "\n"


# ------------------------------------------------------- bibliography ------
KEY_BY_START = [
    ("Aitchison J (1982)", "aitchison1982"), ("Aitchison J (1986)", "aitchison1986"),
    ("Angelopoulos", "angelopoulos2023"), ("Barber", "barber2021"),
    ("Baxter MJ, Freestone", "baxter2006"), ("Beier", "beier1994"),
    ("Breiman", "breiman2001"),
    ("Buxeda", "buxeda1999"), ("Chen T", "chen2016"), ("Chow", "chow1970"),
    ("Frahm", "frahm2013"), ("Geifman", "geifman2017"), ("Glascock", "glascock2003"),
    ("Guo C", "guo2017"), ("Harbottle", "harbottle1976"),
    ("Hein A (2026)", "hein2026"),
    ("Hein A, Dobosz A, Day", "hein2021"),
    ("Hein A, Dobosz A, Kilikoglou", "heindata2021"),
    ("Hendrycks", "hendrycks2017"), ("Hunt", "hunt2015"), ("Kaufman", "kaufman2012"),
    ("Ledoit", "ledoit2004"), ("Lee K", "lee2018"),
    ("Peacock", "peacock1986"),
    ("Pedregosa", "pedregosa2011"),
    ("Platt J", "platt1999"),
    ("Rice PM", "rice1987"),
    ("Roberts", "roberts2017"), ("Scheirer", "scheirer2013"),
    ("Shackley", "shackley2011"),
    ("Shafer", "shafer2008"), ("Speakman", "speakman2013"), ("Tite", "tite2008"),
    ("Vovk V (2015)", "vovk2015"), ("Vovk V, Gammerman", "vovk2005"),
    ("UNESCO World Heritage Centre", "unesco2026"),
    ("Whitbread", "whitbread1995"), ("Zadrozny", "zadrozny2002"),
]
# natbib short forms: SHORT(YEAR)
SHORT = {
    "aitchison1982": "Aitchison(1982)", "aitchison1986": "Aitchison(1986)",
    "angelopoulos2023": "Angelopoulos and Bates(2023)",
    "barber2021": "Barber et al.(2021)", "baxter2006": "Baxter and Freestone(2006)",
    "beier1994": "Beier and Mommsen(1994)",
    "breiman2001": "Breiman(2001)", "buxeda1999": "Buxeda i Garrig\\'{o}s(1999)",
    "chen2016": "Chen and Guestrin(2016)", "chow1970": "Chow(1970)",
    "frahm2013": "Frahm and Doonan(2013)", "geifman2017": "Geifman and El-Yaniv(2017)",
    "glascock2003": "Glascock and Neff(2003)", "guo2017": "Guo et al.(2017)",
    "harbottle1976": "Harbottle(1976)",
    "hein2026": "Hein(2026)", "hein2021": "Hein et al.(2021a)",
    "heindata2021": "Hein et al.(2021b)", "hendrycks2017": "Hendrycks and Gimpel(2017)",
    "hunt2015": "Hunt and Speakman(2015)", "kaufman2012": "Kaufman et al.(2012)",
    "ledoit2004": "Ledoit and Wolf(2004)", "lee2018": "Lee et al.(2018)",
    "peacock1986": "Peacock and Williams(1986)",
    "pedregosa2011": "Pedregosa et al.(2011)", "platt1999": "Platt(1999)",
    "rice1987": "Rice(1987)",
    "roberts2017": "Roberts et al.(2017)", "scheirer2013": "Scheirer et al.(2013)",
    "shackley2011": "Shackley(2011)",
    "shafer2008": "Shafer and Vovk(2008)", "speakman2013": "Speakman and Shackley(2013)",
    "tite2008": "Tite(2008)", "vovk2015": "Vovk(2015)", "vovk2005": "Vovk et al.(2005)",
    "unesco2026": "UNESCO World Heritage Centre(2026)",
    "whitbread1995": "Whitbread(1995)", "zadrozny2002": "Zadrozny and Elkan(2002)",
}


def build_bibliography(md: str) -> str:
    entries = []
    for raw in md.split("\n"):
        line = raw.strip()
        if not line or line.startswith(("#", "*")):
            continue
        key = next((k for start, k in KEY_BY_START if line.startswith(start)), None)
        if key is None:
            continue
        body = convert(line)
        body = re.sub(r"(https?://\S+?)(?=\s|$)", r"\\url{\1}", body)
        entries.append((key, body))
    log("bibliography entries: {}".format(len(entries)))
    missing = [k for k in SHORT if k not in {e[0] for e in entries}]
    if missing:
        log("WARNING keys with no entry: {}".format(missing))
    lines = [r"\begin{thebibliography}{99}",
             r"\setlength{\itemsep}{2pt plus 1pt minus 1pt}"]
    for key, body in entries:
        lines.append(r"\bibitem[%s]{%s} %s" % (SHORT[key], key, body))
    lines.append(r"\end{thebibliography}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------- run -------
# starred sections are not numbered; the five body sections are numbered by
# LaTeX so that in-text cross-references such as "Section 3.9" stay correct.
SECTIONS = [("abstract.md", "abstract", True), ("introduction.md", "intro", False),
            ("methods.md", "methods", False), ("results.md", "results", False),
            ("discussion.md", "discussion", False),
            ("statements.md", "statements", True)]

for name, prefix, starred in SECTIONS:
    src = MS / name
    tex = md_to_tex(src.read_text(encoding="utf-8"), prefix, starred)
    (TEX / ("body_" + prefix + ".tex")).write_text(tex, encoding="utf-8")
    log("wrote body_{}.tex ({:,} chars)".format(prefix, len(tex)))

(TEX / "refs.tex").write_text(
    build_bibliography((MS / "references.md").read_text(encoding="utf-8")),
    encoding="utf-8")
log("wrote refs.tex")

# unresolved author-year strings left in the body are a conversion failure
leftovers = []
for name, prefix, _ in SECTIONS:
    t = (TEX / ("body_" + prefix + ".tex")).read_text(encoding="utf-8")
    for m in re.finditer(r"\([^()]*\b(?:19|20)\d{2}[^()]*\)", t):
        if "cite" not in m.group(0):
            leftovers.append((prefix, m.group(0)))
if leftovers:
    log("UNCONVERTED CITATIONS:")
    for p, s in leftovers:
        log("  {}: {}".format(p, s))
else:
    log("all inline citations converted")
log("LaTeX body build complete.")
log.close()
