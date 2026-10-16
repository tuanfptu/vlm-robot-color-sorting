#!/usr/bin/env python3
"""
Lightweight helpers for language-conditioned object selection.

This module is deliberately free of torch / pybullet / transformers imports so
it can be unit-tested without a CLIP checkpoint or a PyBullet GUI.

Pipeline (see color_sorting_vlm.py):

    user text -> parse_command() -> camera + segmentation crops
              -> validate_crops() -> CLIP similarity (panda_vision_simulation)
              -> rank_scores() / select_target() -> sim.pick_object(name)
              -> placement zone -> verify_in_zone()

Notes on scores: the CLIP scores produced by
VisionLanguagePandaSimulation.compute_object_similarity() are a softmax over the
crops visible in *this* frame. They are relative rankings, not calibrated
confidences, and are always labelled that way here.
"""

import math
import os
import re
import sys
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

SUPPORTED_COLORS = ("red", "blue", "green", "yellow")

# Colors CLIP could be asked about but no object / zone in the simulator has.
UNSUPPORTED_COLORS = (
    "orange", "purple", "violet", "pink", "black", "white", "brown", "gray",
    "grey", "cyan", "magenta", "gold", "silver", "teal", "navy", "beige",
)

# Canonical shape -> words that refer to it.
SHAPE_SYNONYMS = {
    "cube": ("cube", "cubes", "box", "boxes", "block", "blocks", "square"),
    "sphere": ("sphere", "spheres", "ball", "balls", "round", "orb"),
    "cylinder": ("cylinder", "cylinders", "cylindrical", "can", "tube"),
}

UNSUPPORTED_SHAPES = (
    "cone", "cones", "pyramid", "pyramids", "triangle", "triangular", "star",
    "torus", "ring", "prism", "disk", "disc", "capsule",
)

# Words that introduce a placement zone, e.g. "... in the blue zone".
ZONE_PREPOSITIONS = ("in", "into", "to", "onto", "on", "at", "inside")
ZONE_NOUNS = ("zone", "area", "bin", "region")
THROW_WORDS = ("throw", "trash", "discard", "bin it")

# Placement menu shown in interactive mode. Keys are what the user may type.
ZONE_MENU = (
    ("1", "red_zone"),
    ("2", "blue_zone"),
    ("3", "green_zone"),
    ("4", "yellow_zone"),
    ("5", "throw"),
    ("s", "safe"),
)
SPECIAL_DESTINATIONS = ("throw", "safe")

# Status codes returned by parse_command() / select_target().
OK = "OK"
EMPTY = "EMPTY"
UNSUPPORTED = "UNSUPPORTED"
NOT_PRESENT = "NOT_PRESENT"          # no such object in the simulator catalog
NOT_VISIBLE = "NOT_VISIBLE"          # in catalog, but not detected in this frame
AMBIGUOUS = "AMBIGUOUS"
UNGROUNDED = "UNGROUNDED"            # no supported color/shape word at all
INVALID_ZONE = "INVALID_ZONE"
NO_OBJECTS = "NO_OBJECTS"
NO_SCORES = "NO_SCORES"

FREEFORM_PREFIX = "clip:"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ParsedCommand:
    raw: str
    status: str
    message: str = ""
    color: str = None
    shape: str = None
    zone: str = None                 # 'red_zone' ... | 'throw' | None
    clip_text: str = ""              # text actually sent to CLIP
    candidates: list = field(default_factory=list)
    freeform: bool = False           # 'clip:' prefix -> raw CLIP ranking

    @property
    def ok(self):
        return self.status in (OK, AMBIGUOUS)


@dataclass
class Selection:
    status: str
    selected: str = None
    ranking: list = field(default_factory=list)          # [(name, score)] desc
    candidate_ranking: list = field(default_factory=list)
    clip_top: str = None
    clip_agrees: bool = None
    message: str = ""


# ---------------------------------------------------------------------------
# Object catalog helpers
# ---------------------------------------------------------------------------

def split_object_name(name):
    """'red_cube' -> ('red', 'cube'). Unknown parts are returned as None."""
    parts = name.lower().split("_")
    color = next((p for p in parts if p in SUPPORTED_COLORS), None)
    shape = next((p for p in parts if p in SHAPE_SYNONYMS), None)
    return color, shape


def matching_objects(catalog, color=None, shape=None):
    """Objects in `catalog` consistent with the requested color and/or shape."""
    out = []
    for name in catalog:
        obj_color, obj_shape = split_object_name(name)
        if color and obj_color != color:
            continue
        if shape and obj_shape != shape:
            continue
        out.append(name)
    return out


def describe_catalog(catalog):
    return ", ".join(n.replace("_", " ") for n in catalog) or "(none)"


def example_commands(catalog):
    return [f"Pick up the {n.replace('_', ' ')}" for n in catalog]


# ---------------------------------------------------------------------------
# Command parsing
# ---------------------------------------------------------------------------

def _tokens(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).split()


def _find_zone_clause(tokens):
    """
    Locate '<prep> [the] <word> zone' or 'throw away'.

    Returns (start_index_of_clause or None, zone_value or None, bad_word or None).
    """
    joined = " ".join(tokens)
    for w in THROW_WORDS:
        m = re.search(rf"\b{w}\b", joined)
        if m:
            start = len(joined[:m.start()].split())
            return start, "throw", None

    for i, tok in enumerate(tokens):
        if tok not in ZONE_NOUNS:
            continue
        # walk back to the preposition that opens the clause
        start = i
        j = i - 1
        while j >= 0 and i - j <= 3:
            if tokens[j] in ZONE_PREPOSITIONS:
                start = j
                break
            j -= 1
        word = tokens[i - 1] if i > 0 else None
        if word in SUPPORTED_COLORS:
            return start, f"{word}_zone", None
        # 'zone 3' / 'purple zone' / bare 'zone'
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        bad = word if word not in ZONE_PREPOSITIONS + ("the", "a", None) else nxt
        return start, None, bad or "zone"
    return None, None, None


def parse_command(text, catalog):
    """
    Parse a natural-language pick command against the simulator's object catalog.

    Never picks an object on its own: it only narrows the candidate set.
    CLIP ranking (select_target) decides among visible candidates.
    """
    raw = text if text is not None else ""
    stripped = raw.strip()
    if not stripped:
        return ParsedCommand(raw, EMPTY, "Empty prompt - nothing to do.")

    if stripped.lower().startswith(FREEFORM_PREFIX):
        free = stripped[len(FREEFORM_PREFIX):].strip()
        if not free:
            return ParsedCommand(raw, EMPTY, "Empty prompt after 'clip:'.")
        return ParsedCommand(
            raw, OK, "Free-form CLIP ranking over all visible objects.",
            clip_text=free, candidates=list(catalog), freeform=True)

    tokens = _tokens(stripped)
    if not tokens:
        return ParsedCommand(raw, EMPTY, "Prompt has no words.")

    # --- placement zone (validated before the robot moves) ------------------
    zone_start, zone, bad_zone = _find_zone_clause(tokens)
    if zone_start is not None and zone is None:
        return ParsedCommand(
            raw, INVALID_ZONE,
            f"Unknown placement zone '{bad_zone}'. Valid zones: "
            + ", ".join(f"{c} zone" for c in SUPPORTED_COLORS) + ", or 'throw away'.")
    obj_tokens = tokens[:zone_start] if zone_start is not None else tokens

    # --- color --------------------------------------------------------------
    bad_colors = [t for t in obj_tokens if t in UNSUPPORTED_COLORS]
    if bad_colors:
        return ParsedCommand(
            raw, UNSUPPORTED,
            f"Unsupported color '{bad_colors[0]}'. Supported: {', '.join(SUPPORTED_COLORS)}.")
    colors = list(dict.fromkeys(t for t in obj_tokens if t in SUPPORTED_COLORS))

    # --- shape --------------------------------------------------------------
    bad_shapes = [t for t in obj_tokens if t in UNSUPPORTED_SHAPES]
    if bad_shapes:
        return ParsedCommand(
            raw, UNSUPPORTED,
            f"Unsupported shape '{bad_shapes[0]}'. Supported: {', '.join(SHAPE_SYNONYMS)}.")
    shapes = list(dict.fromkeys(
        canon for t in obj_tokens
        for canon, words in SHAPE_SYNONYMS.items() if t in words))

    if len(colors) > 1 or len(shapes) > 1:
        what = " / ".join(colors) if len(colors) > 1 else " / ".join(shapes)
        cands = [n for n in catalog
                 if (not colors or split_object_name(n)[0] in colors)
                 and (not shapes or split_object_name(n)[1] in shapes)]
        return ParsedCommand(
            raw, AMBIGUOUS, f"Conflicting description ({what}). Please name one object.",
            zone=zone, candidates=cands,
            clip_text=" ".join(obj_tokens))

    color = colors[0] if colors else None
    shape = shapes[0] if shapes else None

    if color is None and shape is None:
        return ParsedCommand(
            raw, UNGROUNDED,
            "No supported color or shape found. Name a color "
            f"({', '.join(SUPPORTED_COLORS)}) and/or shape ({', '.join(SHAPE_SYNONYMS)}), "
            f"or prefix with '{FREEFORM_PREFIX}' for raw CLIP ranking.")

    phrase = " ".join(x for x in (color, shape or "object") if x)
    candidates = matching_objects(catalog, color, shape)
    parsed = ParsedCommand(raw, OK, "", color=color, shape=shape, zone=zone,
                           clip_text=f"a {phrase}", candidates=candidates)

    if not candidates:
        parsed.status = NOT_PRESENT
        parsed.message = (f"There is no {phrase} in this simulation. "
                          f"Available: {describe_catalog(catalog)}.")
    elif len(candidates) > 1:
        parsed.status = AMBIGUOUS
        parsed.message = (f"'{phrase}' matches {len(candidates)} objects: "
                          f"{describe_catalog(candidates)}.")
    else:
        parsed.message = f"Target: {candidates[0]}"
    return parsed


# ---------------------------------------------------------------------------
# Crop validation and score ranking
# ---------------------------------------------------------------------------

def validate_crops(crops, bounding_boxes, catalog):
    """
    Keep only usable crops: known object, non-empty image.

    Returns (valid_crops: dict, problems: list[str]).
    """
    valid, problems = {}, []
    crops = crops or {}
    for name in (bounding_boxes or {}):
        if name not in crops:
            problems.append(f"{name}: detected but no crop was produced")
    for name, img in crops.items():
        if name not in catalog:
            problems.append(f"{name}: not in object catalog, ignored")
            continue
        if img is None:
            problems.append(f"{name}: crop is None")
            continue
        size = getattr(img, "size", None)
        if isinstance(size, tuple) and (len(size) < 2 or min(size[:2]) <= 1):
            problems.append(f"{name}: crop too small {size}")
            continue
        valid[name] = img
    return valid, problems


def _is_valid_score(v):
    if isinstance(v, bool):
        return False
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


def rank_scores(scores):
    """
    Sort CLIP scores descending. Non-numeric / NaN / inf entries are dropped.

    Returns (ranking: list[(name, float)], invalid: list[name]).
    Ties are broken by name so output is deterministic.
    """
    ranking, invalid = [], []
    for name, v in (scores or {}).items():
        if _is_valid_score(v):
            ranking.append((name, float(v)))
        else:
            invalid.append(name)
    ranking.sort(key=lambda kv: (-kv[1], kv[0]))
    return ranking, invalid


def select_target(parsed, scores):
    """
    Choose the object to pick, using the original CLIP ranking restricted to the
    candidates allowed by the command. Never falls back to an arbitrary object.
    """
    ranking, _ = rank_scores(scores)
    if not ranking:
        return Selection(NO_SCORES, message="No valid CLIP scores - nothing selected.")

    clip_top = ranking[0][0]
    allowed = set(parsed.candidates)
    cand_rank = [(n, s) for n, s in ranking if n in allowed]
    sel = Selection(OK, ranking=ranking, candidate_ranking=cand_rank, clip_top=clip_top)

    if not cand_rank:
        sel.status = NOT_VISIBLE
        sel.message = (f"Requested object ({describe_catalog(parsed.candidates)}) "
                       "is not visible to the camera.")
        return sel

    if parsed.freeform:
        sel.selected = cand_rank[0][0]
        sel.clip_agrees = True
        sel.message = "Free-form: CLIP top-1 among visible objects."
        return sel

    if parsed.status == AMBIGUOUS or len(cand_rank) > 1:
        sel.status = AMBIGUOUS
        sel.message = (parsed.message or "Several objects match.") + \
            f" CLIP suggests {cand_rank[0][0]}; please confirm or be more specific."
        return sel

    sel.selected = cand_rank[0][0]
    sel.clip_agrees = (sel.selected == clip_top)
    sel.message = ("CLIP agrees with the command." if sel.clip_agrees else
                   f"CLIP ranked {clip_top} first; the command names {sel.selected}.")
    return sel


# ---------------------------------------------------------------------------
# Placement zones
# ---------------------------------------------------------------------------

def parse_zone_choice(text, zones):
    """
    Map user input to a destination: a key of `zones`, 'throw', 'safe' or None.

    Accepts menu numbers ('1'..'5', 's'), colors ('red', 'red zone') and
    'throw' / 'safe'. Anything else returns None (caller re-prompts).
    """
    t = (text or "").strip().lower()
    if not t:
        return None
    for key, dest in ZONE_MENU:
        if t == key and (dest in zones or dest in SPECIAL_DESTINATIONS):
            return dest
    words = _tokens(t)
    if any(w in ("throw", "trash", "discard", "away") for w in words):
        return "throw"
    if words and words[0] in ("safe", "default"):
        return "safe"
    for w in words:
        if f"{w}_zone" in zones:
            return f"{w}_zone"
    return None


def zone_side(position):
    """Human description of a zone in the robot frame (base at origin, facing +x)."""
    x, y = position[0], position[1]
    depth = "near" if x < 0.3 else "far"
    side = "left" if y > 0 else "right"
    return f"{depth}-{side}"


def verify_in_zone(position, zone_cfg, tolerance=0.02, min_z=0.55):
    """
    Objective check: is the object's XY inside the zone footprint and on the table?

    zone_cfg uses the simulator format: {'position': [x,y,z], 'size': [sx,sy,sz]}
    where size is the full extent.
    """
    if position is None:
        return False
    cx, cy = zone_cfg["position"][0], zone_cfg["position"][1]
    hx = zone_cfg["size"][0] / 2.0 + tolerance
    hy = zone_cfg["size"][1] / 2.0 + tolerance
    x, y, z = position[0], position[1], position[2]
    return abs(x - cx) <= hx and abs(y - cy) <= hy and z >= min_z


def is_lifted(before, after, min_rise=0.08):
    """Grasp check: did the object rise with the gripper?"""
    if before is None or after is None:
        return False
    return (after[2] - before[2]) >= min_rise


# ---------------------------------------------------------------------------
# Terminal UI (ANSI, no extra dependencies; honours NO_COLOR and non-TTY)
# ---------------------------------------------------------------------------

def _color_enabled():
    if os.environ.get("NO_COLOR"):
        return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


class Style:
    enabled = _color_enabled()
    CODES = {
        "bold": "1", "dim": "2", "red": "31", "green": "32", "yellow": "33",
        "blue": "34", "magenta": "35", "cyan": "36", "white": "37", "gray": "90",
    }

    @classmethod
    def s(cls, text, *names):
        if not cls.enabled or not names:
            return str(text)
        codes = ";".join(cls.CODES[n] for n in names if n in cls.CODES)
        return f"\033[{codes}m{text}\033[0m"


OBJECT_COLOR = {"red": "red", "blue": "blue", "green": "green", "yellow": "yellow"}
_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def visible_len(text):
    return len(_ANSI_RE.sub("", text))


def paint_object(name):
    color, _ = split_object_name(name or "")
    return Style.s(name, "bold", OBJECT_COLOR[color]) if color else Style.s(name, "bold")


def rule(char="─", width=64, color="gray"):
    return Style.s(char * width, color)


def banner(title, subtitle="", width=64):
    top = "╭" + "─" * (width - 2) + "╮"
    bot = "╰" + "─" * (width - 2) + "╯"
    lines = [Style.s(top, "cyan")]
    for text, styles in ((title, ("bold", "white")), (subtitle, ("gray",))):
        if not text:
            continue
        pad = width - 4 - len(text)
        left = pad // 2
        lines.append(Style.s("│ ", "cyan") + " " * left + Style.s(text, *styles)
                     + " " * (pad - left) + Style.s(" │", "cyan"))
    lines.append(Style.s(bot, "cyan"))
    return "\n".join(lines)


PANEL_KIND = {
    "info": ("cyan", "ℹ"), "ok": ("green", "✔"), "warn": ("yellow", "!"),
    "error": ("red", "✖"), "step": ("magenta", "›"),
}


def panel(title, lines=(), kind="info", width=64):
    color, icon = PANEL_KIND.get(kind, PANEL_KIND["info"])
    head = f"{icon} {title}"
    out = [Style.s("┌─ ", color) + Style.s(head, "bold", color) + " "
           + Style.s("─" * max(0, width - 4 - len(head)), color)]
    for ln in lines:
        out.append(Style.s("│ ", color) + str(ln))
    out.append(Style.s("└" + "─" * (width - 1), color))
    return "\n".join(out)


def score_table(ranking, selected=None, candidates=(), bar_width=24):
    """
    Render CLIP scores sorted descending with bars.
    ★ = selected, ◆ = allowed by the command, · = other.
    """
    if not ranking:
        return [Style.s("(no scores)", "gray")]
    lines = [Style.s(f"   {'object':<18}{'score':>7}  relative softmax over visible crops", "gray")]
    top = ranking[0][1] or 1.0
    for i, (name, score) in enumerate(ranking, 1):
        mark = "★" if name == selected else ("◆" if name in candidates else "·")
        mark_c = "yellow" if mark == "★" else ("cyan" if mark == "◆" else "gray")
        n = int(round(bar_width * max(score, 0.0) / top)) if top > 0 else 0
        color, _ = split_object_name(name)
        bar = Style.s("█" * n, OBJECT_COLOR.get(color, "white")) + Style.s("░" * (bar_width - n), "gray")
        label = paint_object(name) + " " * max(0, 18 - len(name))
        lines.append(f" {Style.s(mark, mark_c)} {label}{score:7.3f}  {bar}")
    return lines
