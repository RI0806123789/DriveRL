"""`build_map_index` が前提にするノード ID（0 からの連番）と、A* を使う範囲の契約テスト（#91）。"""

from __future__ import annotations

import random

import networkx as nx
import pytest

from app.contracts import Bounds, MapData, MapEdge, MapNode
from app.map.index import build_map_index

K = 3
STEP = 120.0


def _grid(ids: list[int], shuffle: bool = False) -> MapData:
    """K×K の碁盤の目。`ids[i]` が i 番目の交差点の ID。"""
    nodes = [MapNode(ids[i * K + j], j * STEP, i * STEP) for i in range(K) for j in range(K)]
    edges: list[MapEdge] = []
    for i in range(K):
        for j in range(K):
            for di, dj in ((0, 1), (1, 0)):
                if i + di >= K or j + dj >= K:
                    continue
                a, b = nodes[i * K + j], nodes[(i + di) * K + j + dj]
                edges.append(MapEdge(len(edges), a.id, b.id, 1, 7.0, False, 11.1, [(a.x, a.y), (b.x, b.y)], STEP))
    if shuffle:
        random.Random(0).shuffle(nodes)
    edge = (K - 1) * STEP
    return MapData("grid", "grid", 0.0, 0.0, edge, Bounds(-20.0, -20.0, edge + 20.0, edge + 20.0), nodes, edges, [])


@pytest.mark.parametrize(
    "ids",
    [
        [i * 100 for i in range(K * K)],
        [i + 1 for i in range(K * K)],
        [0, 0, 1, 2, 3, 4, 5, 6, 7],
    ],
    ids=["sparse", "one-based", "duplicate"],
)
def test_non_dense_ids_are_rejected(ids: list[int]) -> None:
    with pytest.raises(ValueError, match="0 からの連番"):
        build_map_index(_grid(ids))


def test_shuffled_dense_ids_use_astar(monkeypatch: pytest.MonkeyPatch) -> None:
    index = build_map_index(_grid(list(range(K * K)), shuffle=True))
    assert index._reachable_mask.all()

    calls = {"astar": 0, "dijkstra": 0}
    astar, dijkstra = nx.astar_path, nx.shortest_path

    def count_astar(*args, **kwargs):
        calls["astar"] += 1
        return astar(*args, **kwargs)

    def count_dijkstra(*args, **kwargs):
        calls["dijkstra"] += 1
        return dijkstra(*args, **kwargs)

    monkeypatch.setattr(nx, "astar_path", count_astar)
    monkeypatch.setattr(nx, "shortest_path", count_dijkstra)

    path = index.shortest_path(0, K * K - 1)
    assert path is not None and path[0] == 0 and path[-1] == K * K - 1
    assert calls == {"astar": 1, "dijkstra": 0}
    x, y = index._node_pts[K * K - 1]
    assert index.nearest_node(x + 1.0, y - 1.0) == K * K - 1
