"""Strict content matching without depending on tool calling or JSON support."""
from collections import Counter
import json
import re
from .math_text import equivalent_math_text


class ProtocolError(ValueError):
    pass


TOKEN = re.compile(r"⟦P\d+⟧")


def aligned_phrase(parent: str, phrase: str) -> str | None:
    """Find a semantic phrase despite harmless spacing/full-width punctuation.

    Returns the ORIGINAL substring. No translation is changed or invented.
    """
    punctuation = str.maketrans({"：": ":", "，": ",", "、": ",", "；": ";", "（": "(", "）": ")", "‐": "-"})
    chars, positions = [], []
    for index, char in enumerate(parent):
        if not char.isspace():
            chars.append(char.translate(punctuation))
            positions.append(index)
    needle = "".join(c.translate(punctuation) for c in phrase if not c.isspace())
    if not needle:
        return None
    start = "".join(chars).find(needle)
    return None if start < 0 else parent[positions[start]:positions[start+len(needle)-1]+1]


def restore_plain_tokens(text: str, protected: dict[str, str], source: str = "") -> str:
    """Plain models output natural numbers/formulas, no invented protocol syntax.

    Recreate internal tokens only after exact literal coverage is verified.
    Multiple identical literals receive interchangeable tokens in occurrence order.
    """
    if not protected:
        return text
    text=equivalent_math_text(text)
    protected={token:equivalent_math_text(value) for token,value in protected.items()}
    # Native font/style boundaries can split one decimal across two assets
    # (e.g. 1 + . + 45×10^10). Verify and recover the complete written number
    # before matching individual literals; never accept a changed mantissa.
    for compound in re.findall(r'⟦P\d+⟧(?:\.⟦P\d+⟧)+',source):
        tokens=TOKEN.findall(compound)
        if not all(t in protected for t in tokens):continue
        literal=compound
        for token in tokens:literal=literal.replace(token,protected[token])
        if not re.match(r'\d+\.\d+',literal):continue
        pattern=re.compile(r'(?<![A-Za-z0-9_])'+re.escape(literal)+r'(?![0-9]|[.,][0-9])')
        if len(list(pattern.finditer(text)))==1:
            text=pattern.sub(lambda _:compound,text)
            protected={k:v for k,v in protected.items() if k not in tokens}
    if not protected:return text
    if re.search(r'\b(?:AM|PM)\b',source,re.I):
        for value in protected.values():
            if re.fullmatch(r'\d{1,2}\.\d{2}',value):
                text=re.sub(r'(?<!\d)'+re.escape(value.replace('.',':'))+r'(?!\d)',value,text)
    # A dimension such as "3-D" is naturally translated as "三维". Canonicalize
    # this exact numeric-unit equivalence; do not rewrite arbitrary Chinese words
    # or relax checks on unrelated quantities, signs or percentages.
    digits = "零一二三四五六七八九"
    for token, value in protected.items():
        if value.isdigit() and 0 <= int(value) <= 9 and re.search(re.escape(token)+r"\s*[-‐]?\s*[dD]\b", source):
            text = re.sub(re.escape(digits[int(value)])+r"(?=维)", value, text)
    queues = {}
    for token, value in protected.items():
        queues.setdefault(value, []).append(token)
    pattern = re.compile("(?<![A-Za-z0-9_])(?:" + "|".join(re.escape(v) for v in sorted(queues, key=len, reverse=True)) + ")(?![0-9]|[.,][0-9])")
    seen = Counter()
    def replace(match):
        literal = match.group()
        index = seen[literal]
        seen[literal] += 1
        return queues[literal][index] if index < len(queues[literal]) else literal
    result = pattern.sub(replace, text)
    expected = Counter(protected.values())
    if seen != expected:
        raise ProtocolError("Plain translation changed or duplicated protected content. Required original written forms: " + repr(list(protected.values())) + "; use the original digits/symbols, not spelled-out numbers")
    return result


def strip_wrapper(text: str) -> str:
    text = re.sub(r"^\s*<think>.*?</think>\s*", "", text, flags=re.S).strip()
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```[^\n]*\n", "", text)[:-3].strip()
    return text


def validate_text(source: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError("Empty or non-string translation")
    value = value.strip()
    if '**' not in source:
        value=re.sub(r'\*\*([^*\n]+)\*\*',r'\1',value)
    if '$' not in source:
        # Models sometimes wrap an ordinary source variable in LaTeX. The PDF
        # compositor owns typography; dollar delimiters must never become ink.
        value=re.sub(r'\$([A-Za-z])\$',r'\1',value)
        if '$' in value or re.search(r'\\(?:frac|mathrm|mathbf|text|begin)\b',value):
            raise ProtocolError('Model added LaTeX formatting')
    if Counter(TOKEN.findall(source)) != Counter(TOKEN.findall(value)):
        raise ProtocolError("Protected formulas/numbers/code placeholders changed or duplicated")
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ufffd]", value):
        raise ProtocolError("Control character or replacement glyph in response")
    # A list marker already present in the source belongs to the document, not
    # to model-added Markdown. Preserve it for the renderer's exact glyph/indent.
    formatting_value = value
    if source.startswith('- ') and value.startswith('- '):
        formatting_value = value[2:]
    if "<<<" in value or "</think>" in value or re.search(r"^\s*(?:```|#{1,6} |[-*] )", formatting_value, re.M):
        raise ProtocolError("Model added markup/list formatting")
    # Physical line wraps and indentations are exclusively owned by the renderer.
    return re.sub(r"\s+", " ", value).strip()


def decode_response(text: str, sources: dict[str, str], mode: str, *, allow_missing=False) -> dict:
    text = strip_wrapper(text)
    if mode == "plain":
        if len(sources) != 1:
            raise ProtocolError("Plain mode requires exactly one output target")
        raw = {next(iter(sources)): text}
    elif mode in {"json", "json_schema"}:
        def unique(pairs):
            obj = {}
            for k, v in pairs:
                if k in obj:
                    raise ProtocolError("Duplicate JSON ID")
                obj[k] = v
            return obj
        try:
            raw = json.loads(text, object_pairs_hook=unique)
        except (ValueError, TypeError) as exc:
            raise ProtocolError("Invalid JSON response") from exc
        if not isinstance(raw, dict):
            raise ProtocolError("Expected JSON object")
    else:
        pattern = re.compile(r"<<<([A-Za-z0-9_-]+)>>>\s*(.*?)\s*<<<END>>>", re.S)
        matches = list(pattern.finditer(text))
        if pattern.sub("", text).strip():
            raise ProtocolError("Unexpected text outside tagged translations")
        raw = {}
        for m in matches:
            if m[1] in raw:
                raise ProtocolError("Duplicate tagged ID")
            raw[m[1]] = m[2]
    if set(raw)-set(sources) or (not allow_missing and set(raw) != set(sources)):
        raise ProtocolError(f"Translation ID mismatch: missing={sorted(set(sources)-set(raw))}, extra={sorted(set(raw)-set(sources))}")
    return raw


def parse_response(text: str, sources: dict[str, str], mode: str, protected: dict[str, dict[str, str]] | None = None) -> dict[str, str]:
    raw = decode_response(text, sources, mode)
    values = {}
    for key in sources:
        try:
            value = raw[key]
            # Some otherwise valid structured replies spell protected numbers
            # literally. Rebuild missing tokens only from exact source literals;
            # wrong, duplicated or unknown tokens still fail the normal gate.
            if protected and key in protected and isinstance(value, str):
                expected, present = Counter(TOKEN.findall(sources[key])), Counter(TOKEN.findall(value))
                missing = expected-present
                if missing and not present-expected:
                    literals = {token: literal for token, literal in protected[key].items() if token in missing}
                    if literals:
                        value = restore_plain_tokens(value, literals, sources[key])
            values[key] = validate_text(sources[key], value)
        except ProtocolError as exc:
            raise ProtocolError(f"{key}: {exc}") from exc
    return values


def instructions(sources: dict[str, str], mode: str) -> str:
    if mode == "plain":
        return "Return ONLY the translated text of the one TARGET. No ID, quotes, explanation, Markdown, indentation or line breaks."
    if mode in {"json", "json_schema"}:
        return "Return a JSON object mapping every exact target ID to its translated string. No other keys or commentary."
    return "Return every target as <<<ID>>>translated text<<<END>>>. Use the exact IDs. Nothing outside these blocks. No JSON, Markdown or tool calling."


def parse_content_response(text, sources, mode, protected=None, *, allow_missing=False):
    """Validate only addressable nonempty content, never its language/meaning."""
    try:
        raw=decode_response(text,sources,mode,allow_missing=True)
    except ProtocolError:
        body=strip_wrapper(text)
        if mode in {'tagged','auto'}:
            # Tolerate commentary and the common >>>END>>> typo without a call.
            raw={m[1]:m[2] for m in re.finditer(r'<<<([A-Za-z0-9_-]+)>>>\s*(.*?)(?=<<<[A-Za-z0-9_-]+>>>|>>>END>>>|$)',body,re.S) if m[1]!='END'}
        elif mode in {'json','json_schema'}:
            try:raw=json.loads(body)
            except (TypeError,ValueError):raise ProtocolError('Response cannot be mapped to target IDs') from None
            if not isinstance(raw,dict):raise ProtocolError('Response cannot be mapped to target IDs')
        else:raise
    values={}
    for key,source in sources.items():
        value=raw.get(key)
        if not isinstance(value,str) or not value.strip():continue
        value=strip_wrapper(value)
        if protected and key in protected:
            try:value=restore_plain_tokens(value,protected[key],source)
            except ProtocolError:pass
        # Line breaks/indentation and wrapper Markdown are owned by the renderer.
        if '**' not in source:value=re.sub(r'\*\*([^*\n]+)\*\*',r'\1',value)
        value=re.sub(r'\s+',' ',value).strip()
        if value:values[key]=value
    if not allow_missing and set(values)!=set(sources):
        raise ProtocolError('Missing/empty target IDs: '+', '.join(sorted(set(sources)-set(values))))
    return values


def response_format(sources: dict[str, str], mode: str) -> dict | None:
    if mode == "json":
        return {"type": "json_object"}
    if mode == "json_schema":
        return {"type": "json_schema", "json_schema": {"name": "translations", "strict": True, "schema": {
            "type": "object", "properties": {k: {"type": "string"} for k in sources},
            "required": list(sources), "additionalProperties": False,
        }}}
    return None
