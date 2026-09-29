"""pytest の共通設定（重いテストの切り分け・プロトコル文書の読み取り・最小のマップ）。"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from app.contracts import (
    Bounds,
    MapBuilding,
    MapData,
    MapEdge,
    MapNode,
    MapSign,
    MapSignal,
)

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BACKEND_DIR.parent
PROTOCOL_TS = REPO_DIR / "frontend" / "src" / "types" / "protocol.ts"
PROTOCOL_MD = REPO_DIR / "docs" / "protocol.md"

_INTERFACE = re.compile(
    r"^export interface (\w+)(?: extends (\w+))? \{\n(.*?)^\}", re.MULTILINE | re.DOTALL
)
_FIELD = re.compile(r"^  (\w+)(\?)?:", re.MULTILINE)
_TYPE_LITERAL = re.compile(r"^  type: '(\w+)'", re.MULTILINE)
_UNION = re.compile(r"^export type (\w+) =\n((?:  \| \w+\n)+)", re.MULTILINE)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="キャッシュ済みのマップを読む重いテスト（slow）も回す",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="重いテスト。--runslow を付けると回る")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@dataclass(frozen=True)
class TsInterface:
    """protocol.ts の interface 1 つ。`required` は `?` の付かない欄。"""

    name: str
    fields: frozenset[str]
    required: frozenset[str]
    #: `type: 'frame'` のような判別用の値。メッセージ以外は None
    type_literal: str | None = None


def parse_ts_interfaces(source: str) -> dict[str, TsInterface]:
    """`export interface` の直下（2 字下げ）の欄を読む。`extends` は親の欄を足す。"""
    raw: dict[str, tuple[str | None, set[str], set[str], str | None]] = {}
    for m in _INTERFACE.finditer(source):
        body = m.group(3)
        fields: set[str] = set()
        required: set[str] = set()
        for f in _FIELD.finditer(body):
            fields.add(f.group(1))
            if not f.group(2):
                required.add(f.group(1))
        literal = _TYPE_LITERAL.search(body)
        raw[m.group(1)] = (m.group(2), fields, required, literal.group(1) if literal else None)

    def resolve(name: str) -> TsInterface:
        parent, fields, required, literal = raw[name]
        if parent is not None:
            base = resolve(parent)
            fields = fields | base.fields
            required = required | base.required
            literal = literal or base.type_literal
        return TsInterface(name, frozenset(fields), frozenset(required), literal)

    return {name: resolve(name) for name in raw}


def parse_ts_unions(source: str) -> dict[str, list[str]]:
    """`export type X =` に 1 行ずつ `| Name` を並べたユニオンを読む。"""
    return {
        m.group(1): [line.strip()[2:] for line in m.group(2).splitlines()]
        for m in _UNION.finditer(source)
    }


@pytest.fixture(scope="session")
def protocol_ts_source() -> str:
    return PROTOCOL_TS.read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def protocol_md_source() -> str:
    return PROTOCOL_MD.read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def ts_interfaces(protocol_ts_source: str) -> dict[str, TsInterface]:
    return parse_ts_interfaces(protocol_ts_source)


@pytest.fixture(scope="session")
def ts_unions(protocol_ts_source: str) -> dict[str, list[str]]:
    return parse_ts_unions(protocol_ts_source)


@pytest.fixture
def tiny_map() -> MapData:
    """交差点 2 つ・道路 1 本・建物 1 棟・信号と標識 1 基ずつの最小のマップ。"""
    return MapData(
        preset_id="tiny",
        name="最小のマップ",
        center_lat=35.0,
        center_lon=139.0,
        radius_m=50.0,
        bounds=Bounds(-50.0, -50.0, 50.0, 50.0),
        nodes=[MapNode(0, 0.0, 0.0), MapNode(1, 40.0, 0.0)],
        edges=[
            MapEdge(
                id=0,
                u=0,
                v=1,
                lanes=2,
                width=6.5,
                oneway=False,
                speed_limit=8.33,
                polyline=[(0.0, 0.0), (20.0, 0.0), (40.0, 0.0)],
                length=40.0,
            )
        ],
        buildings=[MapBuilding(0, 9.0, [(5.0, 6.0), (15.0, 6.0), (15.0, 16.0), (5.0, 16.0)])],
        signals=[
            MapSignal(
                id=0,
                node_id=1,
                x=33.0,
                y=-1.6,
                heading=0.0,
                group=0,
                road_width=6.5,
                phase_key=1,
                edge_id=0,
            )
        ],
        signs=[
            MapSign(id=0, node_id=0, edge_id=0, x=2.0, y=-3.5, heading=0.0, speed_limit=8.33)
        ],
    )
