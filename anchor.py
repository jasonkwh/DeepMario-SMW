from typing import TypedDict


class RAMAnchor(TypedDict):
    address: int
    type: str


ANCHORS: dict[str, RAMAnchor] = {
    "x": {
        "address": 8257684,
        "type": "<u2",
    },
    "y": {
        "address": 8257686,
        "type": "<u2",
    },
    # powerup
    # valid values:
    # 00 small, 01 big, 02 cape, 03 fire flower
    "powerup": {
        "address": 8257561,
        "type": "<u1",
    },
    # item box
    # valid values:
    # 00 none, 01 mushroom, 02 fire flower, 03 star, 04 feather
    "item": {
        "address": 8261058,
        "type": "<u1",
    },
    # timer
    "timer_hundreds": {
        "address": 8261425,
        "type": "<u1",
    },
    "timer_tens": {
        "address": 8261426,
        "type": "<u1",
    },
    "timer_ones": {
        "address": 8261427,
        "type": "<u1",
    },
    # completion
    "level_end_timer": {
        "address": 8262803,
        "type": "<u1",
    },
    # star timer
    "star_timer": {
        "address": 8262800,
        "type": "<u1",
    },
}
