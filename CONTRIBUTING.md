# DriveRL コントリビューションガイドライン

DriveRL への貢献をご検討いただきありがとうございます。
バグ報告、新機能提案、強化学習モデルの改善、ドキュメント修正など、あらゆる貢献を歓迎します。

開発に着手する前に、本ガイドラインと内部設計書（[`CLAUDE.md`](CLAUDE.md)、[`docs/protocol.md`](docs/protocol.md)）を
お読みください。コメント・ドキュメント・やり取りはすべて**日本語**、変数名・識別子は英語です。
行動規範は [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)、脆弱性の報告は [`SECURITY.md`](SECURITY.md) を見てください
（脆弱性は公開の Issue にしないでください）。

---

## 1. 開発環境のセットアップ

前提: **Python 3.13** と **Node.js 22.18 以降**（`npm test` / `npm run verify` が TypeScript を直接実行するため）。
CUDA 対応 GPU は不要です。

```powershell
# バックエンド（リポジトリ直下から）
py -3.13 -m venv backend\.venv
backend\.venv\Scripts\python.exe -m pip install --upgrade pip
backend\.venv\Scripts\python.exe -m pip install -r requirements.txt
backend\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt   # pytest（テストを回すなら）

# フロントエンド
cd frontend
npm install
```

詳しい手順と初回のマップ取得（`app.map.prefetch`）は [`README.md`](README.md) を参照してください。

---

## 2. 起動方法

```powershell
.\run.ps1 -Dev    # 開発中。Vite も起動し http://localhost:5173 で開く
.\run.ps1         # バックエンドのみ（frontend/dist を配信）→ 127.0.0.1:8000
.\run.ps1 -Build  # ビルドしてから起動
```

- `-Dev` でリロードされるのは **Vite（フロントエンド）だけ**です。バックエンドは学習スレッドが作り直されるため
  `--reload` を使いません。Python を変更したら `Ctrl+C` で止めて起動し直してください。
- バックエンド無しでフロントだけ確かめるときは `cd frontend; npm run dev` で起動し、`http://localhost:5173/?mock=1` を開きます
  （内蔵のモックサーバー `src/store/mockServer.ts` に接続します）。

---

## 3. 守ってほしい不変条件

### ① 契約はセットで直す

フロント ↔ バックの通信や型を変えるときは、次の 3 つを**必ず同時に**直してください。

1. `docs/protocol.md`（唯一の契約。破壊的変更は `protocolVersion` を上げる）
2. `frontend/src/types/protocol.ts`
3. `backend/app/contracts.py`（バックエンド内部の契約）

片方だけ直すと型チェックもビルドも通りますが、実行時に食い違います。`backend/tests/test_protocol_sync.py` が一部を検査します。

### ② 3D の幾何は数値で検査する

3D の向き・灯火・法線は、間違っていても型チェックもビルドも通ります。幾何に触れたら `npm run verify` を通してください。

### ③ 性能に関わる変更は金沢でも測る

銀座・梅田・栄は 0.8〜0.9km 四方ですが、**金沢だけ 12.3km 四方**で規模が 2 桁違います。
銀座で 2ms の処理が金沢で 1 秒になることがあります。マップ探索・空間索引・描画ループなどを変えたら、
**金沢でも測った結果**を PR に書いてください。

### ④ 依存ライブラリを足さない

このプロジェクトは依存を増やさない方針です（PWA・グラフ・アイコン生成も自前）。足したい場合は先に Issue で相談してください。

### ⑤ コメントは最小限

「何をしているか」を言い直すコメントは書きません。設計の意図・経緯・実測値は `CLAUDE.md` / `README.md` へ書きます。
詳細は `CLAUDE.md` の「コードの書き方」を参照してください（`npm run verify:conventions` が一部を検査します）。

---

## 4. テストと検証

PR を出す前に、次をすべて通してください。

```powershell
# フロントエンド
cd frontend
npm run typecheck
npm run build
npm test            # verify:* の全本 + 単体・描画テスト
npm run verify      # 幾何検証

# バックエンド
cd backend
.venv\Scripts\python.exe -m pytest               # 契約テストと数秒で終わる検証
.venv\Scripts\python.exe -m pytest --runslow     # verify_*.py もすべて（経路・信号・地図・安全ギミックに触れたとき）
```

- 新しい `verify_*.py` は `backend/verify/` に置き、`backend/tests/test_verify_wrappers.py` の `PLANS` に読むマップを登録します。
- `verify:*` の本数など、手で書くと古くなる数字は `frontend/package.json` の `scripts` が唯一の出典です。

---

## 5. プルリクエスト

1. `main` へ直接コミットせず、`feat/...` / `fix/...` / `docs/...` のブランチを切ります。
2. コミットメッセージは `feat:` / `fix:` / `docs:` + **日本語**の要約にします。本文には「なぜそうしたか」と壊れやすい点を書いてください。
3. PR を作ると `.github/pull_request_template.md` が挿入されるので、各項目を埋めてください。PR 本文には、変更の概要・確認したマップ（性能に関わるなら金沢を含める）・実行した検証と結果を書きます。
4. 対応する Issue があれば、PR 本文の末尾に `Closes #N` を書きます（`#N` と書くだけではマージしても閉じません）。
5. 個人情報・鍵・ローカルの絶対パスを含めないでください。

### 提出前のチェックリスト

- [ ] 契約（`protocol.md` / `protocol.ts` / `contracts.py`）に矛盾がない（通信を変えた場合）
- [ ] `npm run typecheck` / `npm run build` / `npm test` / `npm run verify` が通る
- [ ] `python -m pytest` が通る（必要なら `--runslow` も）
- [ ] 計算量に影響する変更は金沢でも測った
- [ ] デバッグ用の `console.log`・一時スクリプト・コメントアウトを残していない
