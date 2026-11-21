import re

# Order matters: redact longer/specific patterns before the generic 9+ digit run.
_PATTERNS = [
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("upi", re.compile(r"\b[a-zA-Z0-9._-]{2,}@[a-zA-Z]{2,}\b")),
    ("pan", re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", re.IGNORECASE)),
    ("ifsc", re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b", re.IGNORECASE)),
    ("aadhaar", re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b")),
    ("phone", re.compile(r"(?:\+91[\s-]?|(?<!\d)0)?[6-9]\d{9}\b")),
    ("account", re.compile(r"\b\d{9,}\b")),
]

_REDACTED = "[REDACTED]"


def _mask_digits(match: re.Match) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    if len(digits) <= 4:
        return _REDACTED
    return "*" * (len(digits) - 4) + digits[-4:]


def redact_sensitive(text: str) -> str:
    """Redact PII from text before any LLM call, preserving dates/amounts/merchants."""
    if not text:
        return ""
    redacted = text
    for kind, pattern in _PATTERNS:
        if kind in {"aadhaar", "account"}:
            redacted = pattern.sub(_mask_digits, redacted)
        elif kind == "phone":
            # Avoid touching parts of longer digit runs already handled as accounts.
            def _phone_sub(m: re.Match) -> str:
                start, end = m.start(), m.end()
                if start > 0 and redacted[start - 1].isdigit():
                    return m.group(0)
                if end < len(redacted) and redacted[end].isdigit():
                    return m.group(0)
                digits = re.sub(r"\D", "", m.group(0))
                return "*" * max(len(digits) - 4, 0) + digits[-4:]

            redacted = pattern.sub(_phone_sub, redacted)
        else:
            redacted = pattern.sub(_REDACTED, redacted)
    return redacted
