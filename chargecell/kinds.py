"""Scan kinds and what their models predict.

ChargeCell follows HRL's tune-up pipeline, with one model per kind of charge-stability scan:

  PvP     plunger vs plunger: find the (1,1) cell of a pair of dots
  PvT     plunger vs reservoir tunnel gate: find the loading lines of an edge dot, the
          tunnel-gate range where electrons load cleanly, and the one-electron operating point
  tiebar  plunger vs plunger zoomed on the (1,1)-(2,0) transition (the "tie bar"): find its
          triple points, a rough interdot tunnel coupling, and a first readout point

Every kind shares the three statuses (FOUND / NOT_IN_WINDOW / UNINTERPRETABLE). Reasons, line
families and dense (per-pixel) heads are per kind. **Never reorder a kind's reasons or line
families**: trained networks depend on the order. Append only, then retrain.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import schema


@dataclass(frozen=True)
class ClassHead:
    """A per-pixel classification head (labels -1 = ignore)."""
    name: str
    classes: int
    ref_index: int | None = None     # supervise only when this anchoring flag is true


@dataclass(frozen=True)
class KindSpec:
    name: str
    title: str
    reasons: tuple[str, ...]
    not_in_window: tuple[str, ...]
    uninterpretable: tuple[str, ...]
    reason_text: dict
    line_families: tuple[str, ...]
    class_heads: tuple[ClassHead, ...]
    n_ref: int
    transpose_augment: bool          # is swapping the axes a symmetry of this kind?
    # plain-language names for the GUI, the CLI and messages
    short: str = ""                  # e.g. "Plunger vs plunger"
    goal: str = ""                   # what a scan of this kind is for
    goal_noun: str = ""              # what "found" refers to

    @property
    def n_dense(self) -> int:
        return sum(h.classes for h in self.class_heads) + len(self.line_families)

    def reasons_for(self, status: str) -> tuple[str, ...]:
        return {schema.FOUND: ("none",), schema.NOT_IN_WINDOW: self.not_in_window,
                schema.UNINTERPRETABLE: self.uninterpretable}[status]


_COMMON_TEXT = {k: schema.REASON_TEXT[k] for k in (
    "none", "no_transitions", "low_snr", "sensor_insensitive", "dots_merged",
    "charge_instability", "resolution_too_coarse")}

PVP = KindSpec(
    name="PvP", title="Plunger vs plunger", short="Plunger vs plunger",
    goal="find the (1,1) cell of a dot pair", goal_noun="(1,1) cell",
    reasons=tuple(schema.REASONS),
    not_in_window=tuple(schema.NOT_IN_WINDOW_REASONS),
    uninterpretable=tuple(schema.UNINTERPRETABLE_REASONS),
    reason_text=dict(schema.REASON_TEXT),
    line_families=tuple(schema.LINE_FAMILIES),
    class_heads=(ClassHead("occ_a", schema.N_OCC_CLASSES, 0),
                 ClassHead("occ_b", schema.N_OCC_CLASSES, 1)),
    n_ref=2, transpose_augment=True,
)

PVT_REASONS = (
    "none",
    # NOT_IN_WINDOW
    "no_transitions",        # no loading lines in view
    "occupancy_too_low",     # empty dot and first loading line visible, second not
    "no_reference",          # loading lines visible, but not the empty dot
    "tunnel_rate_too_low",   # lines fade or latch: tunnel gate too closed
    "reservoir_too_open",    # lines smeared: tunnel gate too open
    # UNINTERPRETABLE
    "low_snr",
    "sensor_insensitive",
    "charge_instability",
    "resolution_too_coarse",
)
PVT = KindSpec(
    name="PvT", title="Plunger vs tunnel gate", short="Plunger vs tunnel gate",
    goal="load exactly one electron at a good tunnel rate", goal_noun="one-electron point",
    reasons=PVT_REASONS, not_in_window=PVT_REASONS[1:6], uninterpretable=PVT_REASONS[6:],
    reason_text={
        **_COMMON_TEXT,
        "none": "The empty dot, its first two loading lines and a tunnel-gate range where "
                "electrons load cleanly are all in the window.",
        "no_transitions": "No loading lines are visible in this window.",
        "occupancy_too_low": "The empty dot and its first loading line are visible, but the "
                             "window stops before the second electron loads.",
        "no_reference": "Loading lines are visible, but not the empty dot, so electrons cannot "
                        "be counted.",
        "tunnel_rate_too_low": "The loading lines fade out: the tunnel gate is too closed for "
                               "electrons to load during the sweep.",
        "reservoir_too_open": "The loading lines are smeared out: the tunnel gate is so open "
                              "that the dot is poorly isolated from the reservoir.",
        "resolution_too_coarse": "Too few points per electron to resolve the loading lines.",
    },
    line_families=("load", "spectator", "sensor"),
    class_heads=(ClassHead("occ", schema.N_OCC_CLASSES, 0),
                 ClassHead("regime", 3, None)),
    n_ref=1, transpose_augment=False,
)
# tunnel-rate regimes of the PvT "regime" head
REGIMES = ("slow", "good", "open")

TIEBAR_REASONS = (
    "none",
    # NOT_IN_WINDOW
    "no_tiebar",             # the (1,1)-(2,0) transition is not in view
    "partially_visible",     # the tie bar runs off the window edge
    # UNINTERPRETABLE
    "low_snr",
    "sensor_insensitive",
    "dots_merged",
    "charge_instability",
    "resolution_too_coarse",
)
TIEBAR = KindSpec(
    name="tiebar", title="Tie bar ((1,1)-(2,0) zoom)", short="Tie bar",
    goal="find the tie bar and its two triple points for readout", goal_noun="tie bar",
    reasons=TIEBAR_REASONS, not_in_window=TIEBAR_REASONS[1:3],
    uninterpretable=TIEBAR_REASONS[3:],
    reason_text={
        **_COMMON_TEXT,
        "none": "The (1,1)-(2,0) line (the tie bar) and both of its triple points are in "
                "the window.",
        "no_tiebar": "The (1,1)-(2,0) line is not in this window.",
        "partially_visible": "The tie bar runs off the edge of the window.",
        "dots_merged": "The two dots are so strongly coupled that the tie bar has no distinct "
                       "ends.",
        "resolution_too_coarse": "Too few points across the tie bar to resolve it.",
    },
    line_families=("a", "b", "tiebar", "interdot", "sensor"),
    class_heads=(ClassHead("region", 5, None),),
    n_ref=0, transpose_augment=False,
)
# classes of the tie-bar "region" head: the four cells around the tie bar, and anything else
REGIONS = ("(1,1)", "(2,0)", "(1,0)", "(2,1)", "other")

KINDS: dict[str, KindSpec] = {k.name: k for k in (PVP, PVT, TIEBAR)}
DEFAULT_KIND = "PvP"


def get(kind: str | None) -> KindSpec:
    try:
        return KINDS[kind or DEFAULT_KIND]
    except KeyError:
        raise ValueError(f"unknown scan kind '{kind}' (known: {', '.join(KINDS)})") from None
