from typing import TypedDict

WRAM_BASE = 0x7E0000
WRAM_SIZE = 0x20000


class RAMAnchor(TypedDict):
    address: int
    type: str


ANCHORS: dict[str, RAMAnchor] = {
    "x": {
        "address": WRAM_BASE + 0x94,
        "type": "<u2",
    },
    "y": {
        "address": WRAM_BASE + 0x96,
        "type": "<u2",
    },
    "layer1_x": {
        "address": WRAM_BASE + 0x1462,
        "type": "<u2",
    },
    "x_speed": {
        "address": WRAM_BASE + 0x7B,
        "type": "<u1",
    },
    "y_speed": {
        "address": WRAM_BASE + 0x7D,
        "type": "<u1",
    },
    "direction": {
        "address": WRAM_BASE + 0x76,
        "type": "<u1",
    },
    "blocked": {
        "address": WRAM_BASE + 0x77,
        "type": "<u1",
    },
    "in_water": {
        "address": WRAM_BASE + 0x75,
        "type": "<u1",
    },
    "climbing": {
        "address": WRAM_BASE + 0x74,
        "type": "<u1",
    },
    "carrying": {
        "address": WRAM_BASE + 0x1470,
        "type": "<u1",
    },
    "flash_timer": {
        "address": WRAM_BASE + 0x1497,
        "type": "<u1",
    },
    # powerup
    # valid values:
    # 00 small, 01 big, 02 cape, 03 fire flower
    "powerup": {
        "address": WRAM_BASE + 0x19,
        "type": "<u1",
    },
    # item box
    # valid values:
    # 00 none, 01 mushroom, 02 fire flower, 03 star, 04 feather
    "item": {
        "address": WRAM_BASE + 0xDC2,
        "type": "<u1",
    },
    # timer
    "timer_hundreds": {
        "address": WRAM_BASE + 0xF31,
        "type": "<u1",
    },
    "timer_tens": {
        "address": WRAM_BASE + 0xF32,
        "type": "<u1",
    },
    "timer_ones": {
        "address": WRAM_BASE + 0xF33,
        "type": "<u1",
    },
    # counters
    # lives is signed: -1 means the death animation has taken over
    "lives": {
        "address": WRAM_BASE + 0xDBE,
        "type": "<i1",
    },
    "coins": {
        "address": WRAM_BASE + 0xDBF,
        "type": "<u1",
    },
    "score": {
        "address": WRAM_BASE + 0xF34,
        "type": "<u4",
    },
    # completion
    "level_end_timer": {
        "address": WRAM_BASE + 0x1493,
        "type": "<u1",
    },
    # star timer
    "star_timer": {
        "address": WRAM_BASE + 0x1490,
        "type": "<u1",
    },
}
