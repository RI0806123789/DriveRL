# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

DriveRL は、OSM の実地図上でマルチエージェント強化学習（自作 PPO）の車両を走らせ、
Three.js で 3D 描画・介入できるシミュレーターです。**車両は擬似カメラ画像を CNN で認識した
結果だけを見て走ります。**

コード内のコメント・ドキュメント・ユーザーとのやり取りはすべて**日本語**です。
変数名・識別子は英語。

---

## コマンド

Python は `backend/.venv`、Node は `frontend/`。ルートの `run.ps1` / `run.cmd` は
venv のパスを埋めるだけの薄いラッパーで、中身は `backend/run.py` です。

```powershell
# 起動（-Dev で Vite も子プロセスとして面倒を見る → localhost:5173）
.\run.ps1 -Dev          # 開発中。フロントの変更が保存だけで反映される
.\run.ps1               # バックエンドのみ（frontend/dist を配信）→ 127.0.0.1:8000
.\run.ps1 -Build        # ビルドしてから上

# フロントエンド
cd frontend
npm run typecheck       # tsc --noEmit
npm run build           # typecheck + vite build
npm run verify          # 幾何検証 7 本をまとめて実行（ブラウザ不要）
npm run verify:signals  # 1 本だけ（他: camera / colors / vehicles / sun / detections / speedsigns）

# バックエンド（backend/ から。`python -m app.x` なら PYTHONPATH は不要）
.venv\Scripts\python.exe -m app.map.prefetch          # OSM の事前ダウンロード（初回は数十秒/エリア）
.venv\Scripts\python.exe train_detector.py --samples 2400 --epochs 12   # 認識器の学習
.venv\Scripts\python.exe train_detector.py --collect-only              # 教師データ収集だけ
# ↑ どちらも操作パネルの「モデル作成」タブから同じことができる（中身は同じ実装）
.venv\Scripts\python.exe verify_log_std.py           # 方策分布の健全性チェック
```

**`-Dev` でリロードするのは Vite だけです。** バックエンドを直したら Ctrl+C で止めて
起動し直す必要があります（`--reload` を使わないのは、学習スレッドが作り直されるため）。

### テスト

**自動テストのフレームワークは入っていません**（pytest も vitest も無い）。確認手段は
`npm run verify` の幾何検証スクリプト群と、型チェック・ビルド、そして実測です。
バックエンドの変更を検証するときは、スクラッチにベンチ／比較スクリプトを書いて
**新旧の結果が一致することと速度**を数値で確かめるのが、このリポジトリで機能している作法です。

`npm run verify` が存在する理由: **3D の向きは間違っていても型チェックもビルドも通る**ため、
幾何計算を React から切り離した純粋モジュールへ置き、数値で不変条件を検査しています
（実際に灯火の並びが左右逆になっていたバグをこれが検出しました）。
Node 22.18 以降が要るのは `node scripts/verify-*.ts` で TS を直接実行するためです。

---

## アーキテクチャ

### 2 つの契約ファイル

| ファイル | 何の契約か |
|---|---|
| `docs/protocol.md` | **フロント ↔ バックの唯一の契約。** WebSocket メッセージ・座標系。破壊的変更時は `protocolVersion` を上げる |
| `backend/app/contracts.py` | **バックエンド内部の契約。** 依存を持たない中立地帯で、`map` / `sim` / `rl` / `runtime` はここ経由でのみやり取りする |

`frontend/src/types/protocol.ts` は `docs/protocol.md` の TypeScript 版です。
**3 つはセットで直すこと。**

### 観測が作られる経路（ここが一番間違えやすい）

```
world（真値）
  ├─ percep/camera.py     擬似カメラ 192×144 を numpy で描く（運転席視点）
  │      ↓ 画像
  ├─ percep/detector.py   CNN（Keras 3 / torch バックエンド。TensorFlow は入れない）
  │      ↓ 検出結果 (percep/types.py)
  ├─ percep/encoder.py    → 観測 57 次元
  └─ percep/groundtruth.py  真値から作る「理想の検出結果」
                            ★ 教師データ兼フォールバックの二役。分けてはいけない
```

認識器の**学習**は `percep/trainer.py` にあり、**CLI（`train_detector.py`）と
Web UI（`runtime/detector_job.py` →「モデル作成」タブ）が同じ関数を呼ぶ**。
収集・損失・検証を片方にだけ足すと「コマンドでは学習できるのに画面からだと違う」
という切り分け不能な食い違いになるので、**アルゴリズムは必ず `trainer.py` へ書くこと。**

**不変条件**:

- **観測はカメラ由来、報酬と終了条件は真値由来。** 認識を誤ればそのまま赤信号に突っ込むが、
  信号無視・衝突の判定は真値で行う。「認識できなかったから見逃す」は起こらない。
- **`camera.py` と `groundtruth.py` は同じものを見ていなければならない。**
  描いた物体にラベルが付かない（またはその逆）と、認識器から見てタスクが定義できません。
  正対判定は `percep/types.py` の `facing_viewer()` に一本化してあり、投影式・3D 寸法も
  両者で一致させること。`groundtruth.py` の冒頭にこの約束が書いてあります。
- 3D の寸法は `frontend/src/scene/signalGeometry.ts` / `signGeometry.ts` とも一致させること。
  **片方だけ変えると画面の見た目と検出枠がずれます。**
- `groundtruth._STATIC_CACHE` は `map_index` の identity 比較 1 件だけのキャッシュです。
  **複数マップを行き来するコードでは切り替えのたびに `clear_static_cache()` を呼ぶこと。**

### 実行時のスレッド構成

`runtime/engine.py` が**専用 OS スレッド**で物理と PPO を回し、asyncio（WebSocket）とは
`queue.Queue`（コマンド）とロック付きスナップショット（状態）だけでやり取りします。
だから「一時停止は描画だけ止まり、学習は止まらない」が自然に成立します。

- 1 ステップの予算は 50ms（20Hz）。**エンジンスレッドを止める処理を入れないこと。**
  過去に `_install_route()` が金沢で 1 秒止めていた実例があります（`memo/code_review.md` M-01）。
- PPO の勾配更新は 1 ステップ 1 回ずつに分割済み（まとめると数百 ms 止まる）。

認識器の学習（`runtime/detector_job.py`）は**さらに別のスレッド**で回り、その間だけ
`engine.suspend_sim()` で**物理と PPO を止めます**（`_run_loop` が `_step_once()` を飛ばす）。
利用者の「一時停止」（描画だけ止める）とは別物なので混ぜないこと。止める理由は 2 つ:

1. 収集も学習も CPU を使い切るので、同時に回すと 50ms の予算を守れない
2. `groundtruth._STATIC_CACHE` は `map_index` の identity 比較 1 件だけのキャッシュで、
   別々のマップを持つ 2 つの env が交互に呼ぶと**毎回作り直しになる**

再開（`resume_sim()`）は**inbox 経由**。直前に積む `reload_detector` より先に再開すると、
古い認識器のまま 1 ステップ進んでしまうためです（止めるのは即座、再開は順序つき）。

### フロントエンドの状態管理

**20Hz の `frame` は zustand に入れません。** `frontend/src/store/frameBuffer.ts` の
モジュールスコープの mutable オブジェクトに置き、`useFrame` から直接読みます。
zustand に入れると毎秒 20 回ツリーが再レンダリングされて 3D が重くなるためです。
React の再レンダリングは「マップ変更」「パラメータ変更」「メトリクス更新(1Hz)」「UI 操作」だけ。

サーバー 20Hz / ブラウザ 60fps なので、フロント側でフレーム間を補間します。

### 3D 描画で壊しやすいところ

- **`polygonOffset` の重ね順**: 路面 -1/-2 ＜ 中央線 -2/-4 ＜ 白線 -3/-6 ＜ リボン -4/-8
  ＜ 矢羽根 -5/-10 ＜ 認識車線 -6/-12。崩すと Z ファイティングで点滅します。
- **配色は `scene/palette.ts` が単一の出典**（`usePalette()` 経由）。昼と夜の 2 組を持ち、
  走らせている街の**日の出・日の入りで自動的に切り替わります**（`store/autoTheme.ts`。
  設定項目は無い）。信号の灯火と標識の板の色だけは昼夜で変えません。
- **palette を `<instancedMesh args={[...]}>` の依存にしないこと。** R3F は `args` を
  要素ごとに比較し、違えばオブジェクトごと作り直します。作り直された `InstancedMesh` は
  `count={0}` に戻るのに行列を書く `useEffect` は走らないので、**配色が切り替わった瞬間に
  物体が消えます**。マテリアルは作り直さず色だけ差し替え、`args` に渡したものは
  行列書き込み側の deps にも入れること（`Vehicles.tsx` が手本）。

### モデルの入出力

- **チェックポイントは必ず `weights_only=True` で読む。フォールバックしない。**
  `rl/importer.py` がこれを宣言しており、`SECURITY.md` にも書いてあります。
- 書き出し（`.pt` / TorchScript / `.keras`）には観測レイアウトと行動スケールを
  メタデータとして必ず埋めます。無いと受け取った側が 57 次元に何を入れるか分かりません。
- 観測次元を変えると過去のモデルは読み込めなくなります（50 → 54 → 56 → 57 の履歴あり）。

### マップのキャッシュ

`map/loader.py` の `CACHE_VERSION`（現在 5）を **生成物の中身を変えたら必ず上げること**。
上げないと古いキャッシュが読まれ続けます。

---

## プリセットの規模差に注意

銀座・梅田・栄は 0.8〜0.9km 四方ですが、**金沢だけ 12.3km 四方で 2 桁違います**
（ノード 19,493 / 標識 15,719 / 建物 35,607）。性能に関わる変更は**必ず金沢でも測ること。**
銀座で 2ms の処理が金沢で 1 秒になる、という差が実際に出ます。

金沢は「経路の 80% が物理的に到達不能」（無作為な 2 点が平均 6km 離れるのに 1 エピソードで
2,780m しか進めない）という既知の問題を抱えており、学習題材としては銀座を使うのが既定です。

---

## コードの書き方

- コメントは日本語。**`★` は「ここを変えると気づけない形で壊れる」という印**として
  既存コードが一貫して使っています。新しく不変条件を作ったら同じ印で残すこと。
- 既存コードは「なぜそうなっているか」と実測値を本文に書き込むスタイルです。
  周囲のコメント密度・語り口に合わせてください。
- 失敗を握りつぶすときは**必ず初回だけログを残す**（`code_review B-15`）。
  黙らせると「衝突しない世界」「速度超過 0 件」のように**成績が良くなる方向**に症状が出て、
  外から絶対に気づけなくなります。
- `memo/code_review.md` に既知の指摘が番号付き（`B` / `F` / `X` / `P` / `Q` / `R` / `W` / `L` /
  `M` / `C` / `S`）で溜まっています。該当箇所を直すときは番号を引いて参照すること。
