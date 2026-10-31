from typing import TypedDict

WRAM_BASE = 0x7E0000
WRAM_SIZE = 0x20000


class RAMAnchor(TypedDict):
    address: int
    type: str
    obs: float | str | dict[str, int] | tuple[str, float] | None


ANCHORS: dict[str, RAMAnchor] = {
    "x": {
        "address": WRAM_BASE + 0x94,
        "type": "<u2",
        "obs": 4096.0,
    },
    "y": {
        "address": WRAM_BASE + 0x96,
        "type": "<u2",
        "obs": 4096.0,
    },
    "layer1_x": {
        "address": WRAM_BASE + 0x1462,
        "type": "<u2",
        "obs": None,
    },
    "x_speed": {
        "address": WRAM_BASE + 0x7B,
        "type": "|i1",
        "obs": 128.0,
    },
    "y_speed": {
        "address": WRAM_BASE + 0x7D,
        "type": "|i1",
        "obs": 128.0,
    },
    "direction": {
        "address": WRAM_BASE + 0x76,
        "type": "|u1",
        "obs": "flag",
    },
    "blocked": {
        "address": WRAM_BASE + 0x77,
        "type": "|u1",
        "obs": {
            "middle": 0x10,
            "up": 0x08,
            "down": 0x04,
            "left": 0x02,
            "right": 0x01,
        },
    },
    "in_water": {
        "address": WRAM_BASE + 0x75,
        "type": "|u1",
        "obs": "flag",
    },
    "climbing": {
        "address": WRAM_BASE + 0x74,
        "type": "|u1",
        "obs": None,
    },
    "ducking": {
        "address": WRAM_BASE + 0x73,
        "type": "|u1",
        "obs": "flag",
    },
    # $7E:0072: 0 on ground, 11 jump, 12 P-speed takeoff, 36 falling.
    # Airborne flag only; y_speed already carries rising vs falling.
    "player_in_air": {
        "address": WRAM_BASE + 0x72,
        "type": "|u1",
        "obs": "flag",
    },
    # $7E:13E4 P-meter: #$70 means max run speed and cape flight available.
    "pmeter": {
        "address": WRAM_BASE + 0x13E4,
        "type": "|u1",
        "obs": None,
    },
    # $7E:0071 non-zero: an animation is eating player input (pipe, death,
    # powerup, door, cutscene). Binary on purpose, the 14 codes are not ordinal.
    "input_blocked": {
        "address": WRAM_BASE + 0x71,
        "type": "|u1",
        "obs": "flag",
    },
    "carrying": {
        "address": WRAM_BASE + 0x1470,
        "type": "|u1",
        "obs": None,
    },
    "flash_timer": {
        "address": WRAM_BASE + 0x1497,
        "type": "|u1",
        "obs": None,
    },
    # powerup
    # valid values:
    # 00 small, 01 big, 02 cape, 03 fire flower
    "powerup": {
        "address": WRAM_BASE + 0x19,
        "type": "|u1",
        "obs": 3.0,
    },
    # item box
    # valid values:
    # 00 none, 01 mushroom, 02 fire flower, 03 star, 04 feather
    "item": {
        "address": WRAM_BASE + 0xDC2,
        "type": "|u1",
        "obs": None,
    },
    # timer
    "timer_hundreds": {
        "address": WRAM_BASE + 0xF31,
        "type": "|u1",
        "obs": ("timer", 400.0),
    },
    "timer_tens": {
        "address": WRAM_BASE + 0xF32,
        "type": "|u1",
        "obs": None,
    },
    "timer_ones": {
        "address": WRAM_BASE + 0xF33,
        "type": "|u1",
        "obs": None,
    },
    # counters
    # lives is signed: -1 means the death animation has taken over
    "lives": {
        "address": WRAM_BASE + 0xDBE,
        "type": "|i1",
        "obs": None,
    },
    "coins": {
        "address": WRAM_BASE + 0xDBF,
        "type": "|u1",
        "obs": None,
    },
    "score": {
        "address": WRAM_BASE + 0xF34,
        "type": "<u4",
        "obs": None,
    },
    # completion
    "level_end_timer": {
        "address": WRAM_BASE + 0x1493,
        "type": "|u1",
        "obs": None,
    },
    # star timer
    "star_timer": {
        "address": WRAM_BASE + 0x1490,
        "type": "|u1",
        "obs": None,
    },
}
