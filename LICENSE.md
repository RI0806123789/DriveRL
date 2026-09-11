# ライセンス

## 本ソフトウェアについて

MIT License

Copyright (c) 2026 伊藤涼真 (Ryoma Ito)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## 地図データについて

本ソフトウェアが取得・表示する道路網および建物のデータは
[OpenStreetMap](https://www.openstreetmap.org/) の提供によるもので、
**Open Data Commons Open Database License (ODbL) 1.0** の下で利用できます。

- データの著作権: © OpenStreetMap contributors
- ライセンス全文: <https://opendatacommons.org/licenses/odbl/1-0/>
- 帰属表示の指針: <https://www.openstreetmap.org/copyright>

地図データはこのリポジトリには含まれません。実行時に Overpass API から取得され、
`backend/data/` 以下にキャッシュされます（`.gitignore` で除外）。
取得したデータやそれを加工したものを再配布する場合は、ODbL の帰属表示および
継承（share-alike）の条件に従ってください。

上記 MIT License は本ソフトウェアのソースコードにのみ適用され、
地図データには適用されません。

---

## 第三者ソフトウェアについて

本ソフトウェアは以下の主要な依存ライブラリを利用しています。
いずれも各ライブラリのライセンスに従って利用してください。

| ライブラリ | ライセンス |
|---|---|
| PyTorch | BSD-3-Clause |
| FastAPI / Starlette / Uvicorn | MIT / BSD-3-Clause |
| OSMnx | MIT |
| GeoPandas / Shapely / NetworkX / NumPy | BSD-3-Clause |
| React / React DOM | MIT |
| three.js | MIT |
| @react-three/fiber / @react-three/drei | MIT |
| Zustand / Vite / TypeScript | MIT / Apache-2.0 |
