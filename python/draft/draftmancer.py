"""Draftmancer cube-format pack loader.

Draftmancer files (https://draftmancer.com) are multi-section text files
that describe a custom draft cube. The OMN_Draft_3.5.txt that ships
with this repo is in that format. Sections we care about:

``[Settings]``    — JSON blob with cube-level options (we honour
                    ``withReplacement``).
``[CustomCards]`` — JSON array of card metadata (name, rarity, mana_cost,
                    collector_number, type, image_uris). The collector
                    number gives us a stable set-prefixed id (e.g.
                    "OMN004") and the type field lets us infer
                    pitch/equipment slot.
``[Layouts]``     — Pack layout templates, each weighted. A layout is a
                    name + counts of named sections:
                        - layoutkey (weight)
                            3 LightningWizard
                            2 LightningRuneblade
                            ...
                            2 rare
                            1 RFCommon
``[<SectionName>]`` — One per named slot used by layouts. Each line is
                    "<count> <card name>". Counts are the relative
                    weights inside that slot.

Important: the OMN cube ships with **no heroes** and **no signature
weapons**. Players are assumed to pick a hero outside the draft. This
loader therefore returns *just* the booster contents; assigning a hero
and weapon to each seat is the pipeline orchestrator's job.

Card-name normalisation
-----------------------
Cards in the file have human-readable names like
``"Comet Collision (red)"``. The rest of FAB_Sim_Headless uses
Talishar-style lowercase-underscored slugs (``comet_collision_red``).
This loader normalises every emitted card id so the deck builder /
adapter / analytics layers all speak the same language. The mapping is
fully deterministic — see :func:`slugify` for the algorithm.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .pack_loader import Pack, PackPool


# ---------------------------------------------------------------------------
# Slug normalisation
# ---------------------------------------------------------------------------
_PITCH_PAREN_RE = re.compile(r"\s*\((red|yellow|blue)\)\s*$", re.IGNORECASE)
_NON_SLUG_RE    = re.compile(r"[^a-z0-9]+")


def slugify(card_name: str) -> str:
    """Convert a Draftmancer display name into a Talishar slug.

    Examples
    --------
    >>> slugify("Comet Collision (red)")
    'comet_collision_red'
    >>> slugify("Fingers of Fragmentation")
    'fingers_of_fragmentation'
    >>> slugify("Helm of Astral Sanctuary")
    'helm_of_astral_sanctuary'
    """
    name = card_name.strip()
    colour: str | None = None
    m = _PITCH_PAREN_RE.search(name)
    if m:
        colour = m.group(1).lower()
        name = name[: m.start()]
    slug = _NON_SLUG_RE.sub("_", name.lower()).strip("_")
    if colour:
        slug = f"{slug}_{colour}"
    return slug


# ---------------------------------------------------------------------------
# Parsed-cube representation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SectionEntry:
    """One row inside a section: a card and its weight in that section."""
    weight: int
    slug:   str
    display_name: str


@dataclass
class CubeLayout:
    name:   str
    weight: int                         # relative weight across all layouts
    slots:  list[tuple[int, str]]       # (count, section_name)

    def total_cards(self) -> int:
        return sum(c for c, _ in self.slots)


@dataclass
class DraftmancerCube:
    """Parsed in-memory cube ready for sampling."""
    settings: dict
    custom_cards: list[dict]
    layouts:  list[CubeLayout]
    sections: dict[str, list[SectionEntry]]
    source_path: Path | None = None

    def card_universe(self) -> set[str]:
        return {e.slug for entries in self.sections.values() for e in entries}

    def display_name_for(self, slug: str) -> str | None:
        for entries in self.sections.values():
            for e in entries:
                if e.slug == slug:
                    return e.display_name
        return None

    def class_map(self) -> dict[str, set[str]]:
        """Return ``{card_slug: {class, ...}}`` parsed from CustomCards.

        See :func:`class_map_from_cube` for the implementation. Cached
        per cube instance for cheap repeated access.
        """
        if not hasattr(self, "_class_map_cache"):
            self._class_map_cache = class_map_from_cube(self)  # type: ignore[attr-defined]
        return self._class_map_cache  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
_SECTION_HEADER_RE = re.compile(r"^\[(?P<name>[^\]]+)\]\s*$")
_LAYOUT_HEADER_RE  = re.compile(r"^\s*-\s+(?P<name>\S+)\s*\((?P<weight>\d+)\)\s*$")
_SLOT_LINE_RE      = re.compile(r"^\s*(?P<count>\d+)\s+(?P<sec>\S+)\s*$")
_CARD_LINE_RE      = re.compile(r"^\s*(?P<count>\d+)\s+(?P<name>.+?)\s*$")


def parse_draftmancer(path: str | Path) -> DraftmancerCube:
    """Parse ``OMN_Draft_3.5.txt``-style files into a :class:`DraftmancerCube`.

    The parser is small but tolerant: blank lines are skipped, sections
    are case-insensitive on lookup but preserve their original casing in
    the cube object so error messages match the file.
    """
    p = Path(path)
    lines = p.read_text(encoding="utf-8").splitlines()

    sections_raw: dict[str, list[str]] = {}
    current: str | None = None
    for line in lines:
        m = _SECTION_HEADER_RE.match(line)
        if m:
            current = m.group("name")
            sections_raw.setdefault(current, [])
            continue
        if current is None:
            continue
        sections_raw[current].append(line)

    if "Settings" not in sections_raw:
        raise ValueError(f"Draftmancer file missing [Settings] section: {p}")
    settings_text = "\n".join(sections_raw["Settings"]).strip()
    settings = json.loads(settings_text) if settings_text else {}

    custom_cards: list[dict] = []
    if "CustomCards" in sections_raw:
        cc_text = "\n".join(sections_raw["CustomCards"]).strip()
        if cc_text:
            try:
                custom_cards = json.loads(cc_text)
            except json.JSONDecodeError as e:  # pragma: no cover
                raise ValueError(f"[CustomCards] is not valid JSON: {e}") from e

    layouts: list[CubeLayout] = []
    if "Layouts" in sections_raw:
        layout_name: str | None = None
        layout_weight: int | None = None
        layout_slots: list[tuple[int, str]] = []

        def _flush() -> None:
            nonlocal layout_name, layout_weight, layout_slots
            if layout_name is not None and layout_weight is not None and layout_slots:
                layouts.append(CubeLayout(layout_name, layout_weight, layout_slots))
            layout_name = None
            layout_weight = None
            layout_slots = []

        for line in sections_raw["Layouts"]:
            if not line.strip():
                continue
            m = _LAYOUT_HEADER_RE.match(line)
            if m:
                _flush()
                layout_name = m.group("name")
                layout_weight = int(m.group("weight"))
                continue
            m = _SLOT_LINE_RE.match(line)
            if m and layout_name is not None:
                layout_slots.append((int(m.group("count")), m.group("sec")))
                continue
        _flush()

    if not layouts:
        raise ValueError(f"Draftmancer file has no usable [Layouts] entries: {p}")

    # Every named section referenced by a layout must be present.
    referenced = {s for layout in layouts for _, s in layout.slots}
    sections: dict[str, list[SectionEntry]] = {}
    for sec in referenced:
        if sec not in sections_raw:
            raise ValueError(f"Layout references missing section [{sec}]")
        entries: list[SectionEntry] = []
        for line in sections_raw[sec]:
            stripped = line.strip()
            if not stripped:
                continue
            m = _CARD_LINE_RE.match(stripped)
            if not m:
                continue
            count = int(m.group("count"))
            name  = m.group("name").strip()
            entries.append(SectionEntry(count, slugify(name), name))
        if not entries:
            raise ValueError(f"Section [{sec}] has no usable card lines")
        sections[sec] = entries

    return DraftmancerCube(
        settings=settings,
        custom_cards=custom_cards,
        layouts=layouts,
        sections=sections,
        source_path=p,
    )


# ---------------------------------------------------------------------------
# Sampling -> PackPool
# ---------------------------------------------------------------------------
def make_pack_pool(
    cube: DraftmancerCube,
    *,
    n_packs: int,
    seed: int,
    with_replacement: bool | None = None,
) -> PackPool:
    """Generate ``n_packs`` boosters from a parsed cube.

    Parameters
    ----------
    cube : DraftmancerCube
    n_packs : int
        Total packs to produce. The orchestrator typically calls this
        with ``n_players * packs_per_player``.
    seed : int
        Seed for the local :class:`random.Random`. Re-seeding with the
        same value yields bit-identical packs.
    with_replacement : optional bool
        Override the [Settings] ``withReplacement`` flag. When false
        (default for the OMN cube), each pack is built from a fresh
        copy of the section weights — the supply within a single pack is
        sampled without replacement but the cube as a whole is reusable
        for the next pack.

        Note: "without replacement *across the entire cube*" would force
        n_packs <= cube card total, which isn't how FAB cubes work in
        practice. We respect Draftmancer's convention: WR=false means
        no duplicates *inside a single pack*.
    """
    if with_replacement is None:
        with_replacement = bool(cube.settings.get("withReplacement", False))

    rng = random.Random(seed)
    layout_weights = [layout.weight for layout in cube.layouts]
    layout_total   = sum(layout_weights)
    if layout_total <= 0:
        raise ValueError("Cube layouts have non-positive total weight")

    packs: list[Pack] = []
    for pack_id in range(n_packs):
        # Pick a layout (weighted).
        roll = rng.randrange(layout_total)
        cum = 0
        layout = cube.layouts[-1]
        for L, w in zip(cube.layouts, layout_weights):
            cum += w
            if roll < cum:
                layout = L
                break

        # Build the pack.
        pack_cards: list[str] = []
        for count, section_name in layout.slots:
            section = cube.sections[section_name]
            taken = _sample_section(
                section, count=count,
                rng=rng, with_replacement=with_replacement,
            )
            pack_cards.extend(taken)

        rng.shuffle(pack_cards)
        packs.append(Pack(pack_id=pack_id, cards=tuple(pack_cards)))

    pool = PackPool()
    for p in packs:
        pool.add(p)
    return pool


def _sample_section(
    entries: list[SectionEntry],
    *,
    count: int,
    rng: random.Random,
    with_replacement: bool,
) -> list[str]:
    """Sample ``count`` slugs from ``entries`` respecting per-row weights.

    Each ``SectionEntry`` has an integer weight (the "<count>" prefix in
    the file). We expand into a weighted pool, then sample without
    replacement *within this draw* unless ``with_replacement`` is true.
    """
    weighted: list[str] = []
    for e in entries:
        weighted.extend([e.slug] * e.weight)

    if count <= 0:
        return []
    if with_replacement:
        return [rng.choice(weighted) for _ in range(count)]
    if count > len(weighted):
        # Not enough distinct rows — fall back to with-replacement to avoid hard fail.
        return [rng.choice(weighted) for _ in range(count)]
    return rng.sample(weighted, count)


# ---------------------------------------------------------------------------
# Convenience: parse + sample in one call
# ---------------------------------------------------------------------------
def load_pack_pool_draftmancer(
    path: str | Path,
    *,
    n_packs: int,
    seed: int,
) -> PackPool:
    """Top-level helper. Use this from pipeline configs."""
    return make_pack_pool(parse_draftmancer(path), n_packs=n_packs, seed=seed)


# ---------------------------------------------------------------------------
# Class map (for the hero-assignment cascade)
# ---------------------------------------------------------------------------
def class_map_from_cube(cube: DraftmancerCube) -> dict[str, set[str]]:
    """Build ``{card_slug: {class, ...}}`` from the cube's CustomCards.

    Each CustomCards entry carries a ``type`` field like
    ``"Lightning, Wizard, Action, Attack"`` or
    ``"Lightning, Illusionist, Equipment, Arms"``. We tokenise on
    commas, strip whitespace, and treat every comma-separated token as
    a potential class tag. Downstream callers filter to the classes
    they care about (e.g. ``HERO_CLASS.values()``).

    Cards present in the cube's sections but missing from CustomCards
    yield an empty set — the cascade will treat them as uninformative.
    """
    out: dict[str, set[str]] = {}
    for entry in cube.custom_cards:
        name = entry.get("name") or ""
        if not name:
            continue
        slug = slugify(name)
        type_field = (entry.get("type") or "").strip()
        if not type_field:
            out.setdefault(slug, set())
            continue
        tags = {tok.strip() for tok in type_field.split(",") if tok.strip()}
        out.setdefault(slug, set()).update(tags)
    return out
