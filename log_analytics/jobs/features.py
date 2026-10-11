"""Feature extraction: one HTTP request -> 42 dense, explainable numbers.

29 STRUCTURAL features describe shape and statistics only (never a word).
13 KEYWORD features count known attack vocabulary.

Pure Python (no Spark import) so it can be unit-tested and shipped to
executors with sparkContext.addPyFile().
"""
import re
from collections import Counter
from math import log2
from urllib.parse import unquote_plus

PCT_ENC = re.compile(r"%[0-9a-fA-F]{2}")
TOKEN = re.compile(r"[A-Za-z0-9_]+")

STRUCTURAL = [
    "len", "dec_len", "entropy", "r_punct", "r_alpha", "r_digit", "r_upper",
    "r_space", "n_quote_single", "n_quote_double", "n_semicolon", "n_angle",
    "n_paren", "n_bracket", "n_brace", "n_slash", "n_backslash", "n_dot",
    "n_dotdot", "n_pct_enc", "n_dash", "n_equals", "n_amp", "n_pipe",
    "n_backtick", "n_dollar", "n_comment", "n_nonascii", "n_tokens",
]

_KW = {
    "kw_sql": r"\b(select|union|insert|update|delete|drop|from|where|having|group\s+by|order\s+by|exec|table)\b",
    "kw_sql_fn": r"\b(sleep|benchmark|concat|char|version|load_file|extractvalue|updatexml|waitfor|pg_sleep)\s*\(",
    "kw_sql_logic": r"\b(or|and)\b\s*['\"(]?\s*\w+\s*['\"]?\s*(=|like)\s*['\"(]?\s*\w+",
    "kw_xss": r"(script|alert|prompt|confirm|javascript|iframe|onerror|onload|document\.|window\.|eval\()",
    "kw_xss_event": r"\bon[a-z]{3,15}\s*=",
    "kw_cmd": r"\b(cat|ls|whoami|wget|curl|nc|netcat|bash|sh|ping|id|uname|chmod|rm|echo|python|perl)\b",
    "kw_cmd_sep": r"(;|&&|\|\||\|)\s*[a-z/]",
    "kw_cmd_path": r"(/bin/|/usr/bin/|/etc/|cmd\.exe|powershell)",
    "kw_trav": r"(\.\./|\.\.\\|%2e%2e)",
    "kw_trav_file": r"(etc/passwd|etc/shadow|boot\.ini|win\.ini|system32|proc/self)",
    "kw_html_tag": r"</?\s*[a-z][a-z0-9]*[^>]*>",
    "kw_proto": r"(file://|php://|data:|expect://|ftp://|https?://)",
    "kw_hex": r"0x[0-9a-f]{2,}",
}
KEYWORD = list(_KW)
_KW_RE = {k: re.compile(v, re.I) for k, v in _KW.items()}

FEATURE_NAMES = STRUCTURAL + KEYWORD
assert len(STRUCTURAL) == 29 and len(KEYWORD) == 13 and len(FEATURE_NAMES) == 42


def shannon(s):
    """How unpredictable the character mix is (bits per character)."""
    n = len(s)
    if n == 0:
        return 0.0
    return -sum((v / n) * log2(v / n) for v in Counter(s).values())


def build(raw):
    """Return the 42 features (list of floats) for one raw request string."""
    raw = raw or ""
    dec = unquote_plus(raw)  # URL-decode first so %27 counts as a quote
    n = max(len(dec), 1)
    punct = sum(1 for c in dec if not c.isalnum() and not c.isspace())
    alpha = sum(c.isalpha() for c in dec)
    digit = sum(c.isdigit() for c in dec)
    upper = sum(c.isupper() for c in dec)
    space = sum(c.isspace() for c in dec)
    cnt = dec.count
    structural = [
        len(raw), len(dec), shannon(dec), punct / n, alpha / n, digit / n,
        upper / n, space / n, cnt("'"), cnt('"'), cnt(";"),
        cnt("<") + cnt(">"), cnt("(") + cnt(")"), cnt("[") + cnt("]"),
        cnt("{") + cnt("}"), cnt("/"), cnt("\\"), cnt("."), cnt(".."),
        len(PCT_ENC.findall(raw)), cnt("-"), cnt("="), cnt("&"), cnt("|"),
        cnt("`"), cnt("$"), cnt("--") + cnt("/*") + cnt("#"),
        sum(ord(c) > 127 for c in dec), len(TOKEN.findall(dec)),
    ]
    low = dec.lower()
    keyword = [len(_KW_RE[k].findall(low)) for k in KEYWORD]
    return [float(x) for x in structural + keyword]
