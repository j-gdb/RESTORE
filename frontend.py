import re
from typing import Dict, List, Optional, Sequence
from config import FILTERED_TOKENS

ESCAPE_CHAR = "\\"

# original struq filter
def filter(s, tokens: Sequence[str] = None):
    tokens = FILTERED_TOKENS if tokens is None else tokens
    filtered = False
    while not filtered:
        for f in tokens:
            if f in s: s = s.replace(f, '')
        filtered = True
        for f in tokens:
            if f in s: filtered = False
    return s

# reversible escaping
def _default_name(token):
    if len(token) == 1:
        return f"U+{ord(token):04X}"
    m = re.fullmatch(r"\[([A-Za-z0-9]+)\]", token)
    if m:
        return m.group(1).upper()
    return "+".join(f"U+{ord(c):04X}" for c in token)


class Escaper:
    # in the init, we assign each token a plain text body
    def __init__(self, tokens: Sequence[str] = None, names: Optional[Dict[str, str]] = None):
        tokens = list(FILTERED_TOKENS if tokens is None else tokens)
        if not tokens:
            raise ValueError("No reserved tokens given")
        names = dict(names or {})
        self.token_to_body: Dict[str, str] = {}
        for t in tokens:
            if t == "":
                raise ValueError("Empty reserved token")
            if ESCAPE_CHAR in t:
                raise ValueError(f"Reserved token {t!r} contains the escape character")
            name = names.get(t, _default_name(t))
            if not re.fullmatch(r"[A-Za-z0-9+]+", name):
                raise ValueError(f"Escape name {name!r} for {t!r} must be [A-Za-z0-9+]")
            self.token_to_body[t] = f"<{name}>"
        if len(set(self.token_to_body.values())) != len(self.token_to_body):
            raise ValueError("Two reserved tokens share an escape name")
        self.body_to_token = {b: t for t, b in self.token_to_body.items()}

        # bodies must never contain a reserved token.
        for b in self.body_to_token:
            for t in tokens:
                if t in b:
                    raise ValueError(f"Escape body {b!r} contains reserved token {t!r}")

        alt = lambda xs: "|".join(re.escape(x) for x in sorted(xs, key=len, reverse=True)) # x xs
        # encoder: a run of backslashes, then either a reserved token or a literal body.
        self._enc_re = re.compile(rf"(\\*)(?:(?P<tok>{alt(self.token_to_body)})|(?P<body>{alt(self.body_to_token)}))")
        # decoder: a non-empty run of backslashes, then a body.
        self._dec_re = re.compile(rf"(\\+)(?P<body>{alt(self.body_to_token)})")
        self.tokens = tokens

    def escape(self, text):
        def repl(m):
            n = len(m.group(1))
            if m.group("tok") is not None:
                return ESCAPE_CHAR * (2 * n + 1) + self.token_to_body[m.group("tok")]
            if n == 0:  # a body with no backslash should be left
                return m.group("body")
            return ESCAPE_CHAR * (2 * n) + m.group("body")
        out = self._enc_re.sub(repl, text)
        leaked = self.contains_reserved(out)
        if leaked:  # should be impossible
            raise AssertionError(f"escape() left reserved tokens {leaked!r} in its output")
        return out

    def unescape(self, text):
        def repl(m):
            run, body = len(m.group(1)), m.group("body")
            if run % 2 == 1:
                return ESCAPE_CHAR * ((run - 1) // 2) + self.body_to_token[body]
            return ESCAPE_CHAR * (run // 2) + body
        return self._dec_re.sub(repl, text)

    def contains_reserved(self, text: str) -> List[str]:
        return [t for t in self.tokens if t in text]


_default: Optional[Escaper] = None

def _get_default():
    global _default
    if _default is None:
        _default = Escaper()
    return _default

def escape(text):
    return _get_default().escape(text)

def unescape(text):
    return _get_default().unescape(text)


# Things for test.py
FRONTENDS = ("filter", "escape", "none")

def apply(text, mode):
    if mode == "filter":
        return filter(text)
    if mode == "escape":
        return escape(text)
    if mode == "none":
        return text
    raise ValueError(f"Unknown front end {mode!r}; choose from {FRONTENDS}")

# undo escaping
def postprocess_output(text, mode):
    return unescape(text) if mode == "escape" else text
