"""Feature semantics provenance shared by training, inference and dataset compatibility checks.

Two input encodings exist (wm.features.layout gives their columns):
  per-agent-key-tiles-v1  tile patch channels full, half (no orientation), slope ((open corner + 1) / 4),
                          special; the encoding of every checkpoint before tile-shapes-v1
  tile-shapes-v1          full, half, slope, special, open x, open y (realpath.tile_shape)
A checkpoint's trained protocol ("feature_protocol") selects the encoding its inputs are built with
(input_protocol); legacy-unspecified weights were trained on the per-agent-key-tiles-v1 layout (the same
columns; key masks were added without changing the dimension). Fresh training uses FEATURE_PROTOCOL unless
it asks for another one explicitly (wm.train --feature-protocol). Nothing guesses an encoding from data.
"""

KEY_TILES_PROTOCOL = "per-agent-key-tiles-v1"
TILE_SHAPES_PROTOCOL = "tile-shapes-v1"
INPUT_PROTOCOLS = (KEY_TILES_PROTOCOL, TILE_SHAPES_PROTOCOL)
FEATURE_PROTOCOL = KEY_TILES_PROTOCOL        # fresh training's default (tile shapes: explicit opt-in)
LEGACY_FEATURE_PROTOCOL = "legacy-unspecified"
UNTRAINED_FEATURE_PROTOCOL = "untrained"


def trained_protocol(info):
    return (info or {}).get("feature_protocol") or LEGACY_FEATURE_PROTOCOL


def input_protocol(info):
    """The input encoding a checkpoint's weights expect (one of INPUT_PROTOCOLS), from its recorded trained
    protocol; an untrained checkpoint's from the encoding it was built with ("active_feature_protocol"). A
    recorded build encoding that disagrees, or an unknown marker, is an error."""
    info = info or {}
    trained = trained_protocol(info)
    active = info.get("active_feature_protocol")
    if trained == UNTRAINED_FEATURE_PROTOCOL:
        if active not in INPUT_PROTOCOLS:
            raise ValueError(f"untrained checkpoint records no known input feature protocol ({active!r})")
        return active
    if trained == LEGACY_FEATURE_PROTOCOL:
        proto = KEY_TILES_PROTOCOL
    elif trained in INPUT_PROTOCOLS:
        proto = trained
    else:
        raise ValueError(f"unknown checkpoint feature protocol {trained!r}")
    if active is not None and active != proto:
        raise ValueError(f"checkpoint feature protocol mismatch: trained {trained!r}, inputs built as {active!r}")
    return proto


def feature_usage(info):
    return {"active": input_protocol(info), "checkpoint_trained": trained_protocol(info)}


def require_feature_protocol(info):
    if trained_protocol(info) not in INPUT_PROTOCOLS:
        raise ValueError("strict checkpoint feature protocol mismatch; legacy feature weights are not relabelled")
    input_protocol(info)


# ---- physics prior protocols ------------------------------------------------------------------------------
# The continuous outputs are corrections to a physics prior (wm.features.prior); a checkpoint's recorded
# "prior_protocol" selects the one its targets were built against and its rollouts add back:
#   gravity-v1           same velocity, gravity in the air, moved by the new velocity x the time step (every
#                        checkpoint before gravity-fan-wind-v1; none of them records a marker)
#   gravity-fan-wind-v1  gravity-v1 plus the measured fan wind force in the air, frame by frame
#                        (wm.features.fan_force; runs/experiments/wind_prior_v1)
# Fresh training uses PRIOR_PROTOCOL unless it asks for another (wm.train --prior-protocol); --init and the
# learner keep the parent's. Nothing guesses a prior from data.
GRAVITY_PRIOR = "gravity-v1"
FAN_WIND_PRIOR = "gravity-fan-wind-v1"
PRIOR_PROTOCOLS = (GRAVITY_PRIOR, FAN_WIND_PRIOR)
PRIOR_PROTOCOL = GRAVITY_PRIOR              # fresh training's default (fan wind: explicit opt-in)


def prior_protocol(info):
    """The physics prior a checkpoint's continuous outputs correct: its recorded "prior_protocol"; no marker
    is gravity-v1 (the only prior before markers existed). An unknown or empty marker is an error."""
    info = info or {}
    if "prior_protocol" not in info:
        return GRAVITY_PRIOR
    p = info["prior_protocol"]
    if p not in PRIOR_PROTOCOLS:
        raise ValueError(f"unknown checkpoint prior protocol {p!r}")
    return p
