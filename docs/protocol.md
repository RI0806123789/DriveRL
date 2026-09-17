# WebSocket プロトコル仕様 v1

バックエンド（FastAPI）とフロントエンド（React/Three.js）の間の**唯一の契約**。
両側の実装はこの文書に厳密に従うこと。破壊的変更を行う場合は `protocolVersion` を上げる。

- エンドポイント: `ws://localhost:8000/ws`
- 形式: JSON（1 WebSocket メッセージ = 1 JSON オブジェクト）
- 全メッセージは `type: string` を必ず持つ

---

## 1. 座標系

### 1.1 バックエンド（シミュレーション座標 / ENU 平面）

- 単位: **メートル**
- 原点: 選択中プリセットの中心（緯度経度）
- `x` = 東方向（East）が正
- `y` = 北方向（North）が正
- 高さは `z`（ただしシミュレーションは 2D なので車両は常に z=0 平面上）
- 実体は OSMnx が投影した **UTM 座標から中心点の UTM 座標を引いたもの**。
  400m 圏ではスケール歪みは 0.1% 未満なので ENU とみなして扱う。

### 1.2 車両の姿勢

- `heading`: ラジアン。**+x 軸（東）から反時計回り**。
- 進行方向ベクトルは `(cos(heading), sin(heading))`。
- 値域は正規化せず送る場合があるため、フロント側は `atan2(sin, cos)` で `(-pi, pi]` に畳むこと。

### 1.3 フロントエンド（Three.js / 右手系 Y-up）への変換

```
three.position.x =  enu.x
three.position.y =  height        // 地面は 0
three.position.z = -enu.y         // 符号反転に注意
```

車両メッシュは**前方を +X 軸**に向けて作成すること。そうすれば：

```
mesh.rotation.y = heading         // 追加の符号反転は不要
```

（理由: Y 軸回転 φ は +X を `(cos φ, 0, -sin φ)` に写す。上の座標変換で ENU の
`(cos θ, sin θ)` は Three の `(cos θ, 0, -sin θ)` になるため φ = θ で一致する。）

---

## 2. サーバー → クライアント

### 2.1 `init` — 接続確立直後に 1 回だけ送信

```jsonc
{
  "type": "init",
  "protocolVersion": 1,
  "presets": [
    {
      "id": "ginza",
      "name": "東京・銀座",
      "description": "中央通り〜晴海通り周辺の高密度市街地",
      "centerLat": 35.6717,
      "centerLon": 139.765,
      "radiusM": 400
    }
  ],
  "config": {
    "maxVehicles": 8,       // 事前確保するエージェントスロット数
    "simHz": 20,            // 物理・学習ステップの周波数
    "obsDim": 57,
    "actionDim": 2
  },
  "weatherPresets": [       // 天候の選択肢。**(rain, fog) の数値はここが唯一の出典**
    { "id": "clear",     "rain": 0.0,  "fog": 0.0  },
    { "id": "drizzle",   "rain": 0.35, "fog": 0.0  },
    { "id": "rain",      "rain": 0.85, "fog": 0.0  },
    { "id": "fog",       "rain": 0.0,  "fog": 0.75 },
    { "id": "heavy_fog", "rain": 0.15, "fog": 0.95 }
  ],
  "params": { /* 2.5 の SimParams と同じ形。現在値 */ },
  "status": { /* 2.4 の status ペイロードと同じ形 */ }
}
```

`weatherPresets` はクライアントが選択肢を並べるためのもので、**値は
`backend/app/percep/weather.py` の `PRESETS` をそのまま流している**。
上に書いた数値は例示であって出典ではない（表示名だけはクライアント側が持つ）。
クライアントは選んだプリセットの `rain` / `fog` を `set_params` で送り返す。

### 2.2 `map` — マップ読込完了時に送信（サイズが大きい・低頻度）

```jsonc
{
  "type": "map",
  "presetId": "ginza",
  "name": "東京・銀座",
  "bounds": { "minX": -420.0, "maxX": 420.0, "minY": -420.0, "maxY": 420.0 },
  "nodes": [
    { "id": 0, "x": 12.3, "y": -45.6 }
  ],
  "edges": [
    {
      "id": 0,
      "u": 0, "v": 1,          // nodes[].id への参照
      "lanes": 2,
      "width": 7.0,            // メートル。道路メッシュの全幅
      "oneway": false,
      "speedLimit": 13.9,      // m/s
      "polyline": [[12.3, -45.6], [30.1, -40.2]]   // [x, y] の列。始点=u、終点=v
    }
  ],
  "buildings": [
    {
      "id": 0,
      "height": 9.0,           // メートル。OSM に情報がなければ既定値
      "outline": [[x, y], ...] // 外周（閉じない。最後の点と最初の点は別）。CCW 保証なし
    }
  ],
  "signals": [
    {
      "id": 0,
      "nodeId": 12,            // 交差点ノード（nodes[].id）への参照
      "x": 179.0, "y": -238.3, // 停止線の位置（交差点手前）
      "heading": 2.33,         // 進入車両の進行方向 [rad]。灯器は heading + PI を向く
      "group": 0,              // 同じ値どうしが同時に青。交差する流れは必ず別グループ
      "roadWidth": 9.75        // 進入路の幅 [m]。停止線・横断歩道の寸法に使う
    }
  ],
  "signs": [
    {
      "id": 0,
      "nodeId": 2,             // 標識が立つ交差点ノード（nodes[].id）への参照
      "edgeId": 4,             // この標識が規制する道路（edges[].id）への参照
      "x": 270.673, "y": -122.142, // 支柱の位置。進行方向の左側の路端
      "heading": -0.7141,      // 規制する側の進行方向 [rad]。標示板は heading + PI を向く
      "speedLimit": 8.333      // 規制速度 [m/s]（= 30 km/h）
    }
  ]
}
```

- `buildings[].outline` はサーバー側で簡略化（simplify）済み。
- 建物数が数千件になるため、フロントは **必ずジオメトリをマージして 1 メッシュ**にすること。

**信号機について（日本の設置基準に合わせた点）**

- OSM の `highway=traffic_signals` ノードから作る。**接続道路が 3 本以上**の
  交差点にだけ置く（2 本以下は単路部の押しボタン式とみなす）
- 交差点 1 か所につき、**進入できる方向の数だけ**存在する。日本の信号機は
  進入車両に正対して設置されるため、灯器の姿勢は進入方向で決まる。
  一方通行の出口側には信号機を作らない
- `group` は進入方向の**軸**（向きの正負を無視した方向）で 2 つに分けてある。
  直交する流れが必ず別グループになるので、交差する車線が同時に青にならない
- 描画側は灯器を交差点の**対面側**に置き、灯火を運転者から見て**左から 青・黄・赤**に並べる。
  この向きの計算は `frontend/scripts/verify-signal-geometry.ts` で数値検証している

**最高速度標識について（規制標識「最高速度」）**

- OSM の `traffic_sign` ノードは日本ではほとんど付いていないので、**道路（way）の
  `maxspeed` から生成する**。`edges[].speedLimit` と同じ値で、タグが無い道路は
  `highway` 種別の既定値（`residential` = 30km/h など）で埋めてある
- **規制速度が変わる進入口にだけ**置く。同じ速度が続く直線に同じ数字を並べても
  情報が増えないためで、規制が変わる地点に設置するという実際の運用にも合う。
  行き止まりから出る場合（比較対象が無い）は必ず置く
- 実測の基数: 銀座 136 基 / 梅田 54 基 / 栄 91 基 / **金沢 15,719 基**
  （参考: 信号灯器は 261 / 73 / 151 / 2,736 基）。金沢だけ市街地全域（12.3km 四方）なので桁が違う
- **同じ場所・同じ向きの標識は重複しない**（隣ノード単位で生成しているため）。
  以前は対面通行路が `(u,v)` / `(v,u)` の相互エッジ対で来ることに対応できておらず、
  金沢で 27,583 基中 13,250 基が重複していた
- `heading` は `signals[].heading` と同じ約束で、**その標識が規制する側の進行方向**。
  標示板は運転者に正対するので `heading + PI` を向く
- 左側通行なので支柱は**進行方向の左側**の路端に立つ（`x` / `y` はオフセット済み）
- 描画側は白地の円に赤縁・黒数字、直径 60cm、下端を路面から 1.8m に置く
  （道路標識、区画線及び道路標示に関する命令）。数字は **km/h の整数**で、
  `Math.round(speedLimit * 3.6)` で求める。この寸法と向きは
  `frontend/scripts/verify-sign-geometry.ts` で数値検証している

### 2.3 `frame` — 走行状態（等倍で 20Hz、最大 60Hz）

```jsonc
{
  "type": "frame",
  "tick": 1234,               // マップ読込からの通算ステップ数（切り替えると 0 に戻る）
  "simTime": 61.7,            // シミュレーション内経過秒
  "vehicles": [
    {
      "id": 0,                // 0..maxVehicles-1 のスロット番号で固定
      "active": true,         // false のスロットは描画しない
      "x": 1.0, "y": 2.0,
      "heading": 1.5707,
      "speed": 8.3,           // m/s
      "steer": 0.12,          // rad（前輪舵角）。ホイールの見た目に使う
      "collided": false,      // このステップで衝突したか
      "reachedGoal": false,
      "goal": [120.0, -35.0], // 目的地 [x, y]
      "progress": 0.347,      // 経路の達成度 0.0〜1.0（進行距離 / 経路全長）
      "signalViolations": 0,  // このエピソード中に赤信号で停止線を越えた回数
      "laneDepartures": 3,    // このエピソード中に車線を外れた回数
      "speedLimit": 8.333,    // いま適用されている規制速度 [m/s]。標識が無い区間は 0.0
      "speedViolations": 0,   // このエピソード中に規制速度を超えた回数
      "route": [[x, y], ...]  // 目的地までの経路（省略可。変化時のみ入る場合あり）
    }
  ],
  "obstacles": [
    { "id": 3, "x": 1.0, "y": 2.0, "radius": 0.5 }
  ],
  "weather": {                    // いま効いている天候（weatherAuto の間はサーバーが決める）
    "rain": 0.35,                 // 雨の強さ 0.0〜1.0
    "fog": 0.0,                   // 霧の濃さ 0.0〜1.0
    "visibility": 120.0           // 有効視程 [m]。擬似カメラと正解ラベルが共有する
  },
  "signals": [0, 2, 2, 1, ...],  // map.signals と同じ並び。0=青 / 1=黄 / 2=赤
  "detections": {                 // 画像認識の検出結果（キーは車両スロット番号の文字列）
    "3": [
      {
        "cls": 4, "box": [0.0, 0.51, 0.92, 1.0], "conf": 1.0, "distance": 24.7, "lateral": 2.29,
        "lanePoints": [[0.0, -2.29], [2.9, -2.14], [5.7, -2.0], [8.6, -1.85], [11.5, -1.71], [14.3, -1.56]]
      },
      { "cls": 0, "box": [0.41, 0.22, 0.46, 0.31], "conf": 1.0, "phase": 2, "distance": 38.2 }
    ]
  }
}
```

`signals` は信号機が 1 基も無いマップでは省略される。

**現示の決まり方**: サイクルは 60 秒で、青 25 秒 → 黄 3 秒 → 全赤 2 秒 → （交差方向へ交代）。
黄を挟まずに青から赤へ飛ぶことはなく、交代の前に必ず全赤（クリアランス）が入る。
現示は `simTime` だけの関数なので内部状態を持たず、どのクライアントが見ても同じ時刻に同じ灯色になる。

- `vehicles` は**常に全スロット分**（`maxVehicles` 個）を含む。`active` で描画を切り替える。
- `laneDepartures` は毎ステップの外れ具合ではなく、**外れ始めた瞬間**を 1 回と数える。
  車線中心から 1.75m（一般的な車線幅 3.5m の半分）を超えたら「外れた」、
  1.25m まで戻ったら「戻った」とする（しきい値の出入りで回数が跳ねないようヒステリシスを付けている）。
- `route` は毎フレーム送ると重いので、変化があったフレームのみ含まれる。省略時は前回値を保持。
- **`tick` は連番とは限らない。** 物理は `simHz × simSpeed` で進むが、フレームの生成は
  60Hz（`config.FRAME_HZ`）で頭打ちにしてある。8 倍速では物理が毎秒 160 ステップ進む一方、
  配信は毎秒 60 通なので `tick` はおよそ 3 ずつ飛ぶ。
  **クライアントは `tick` の差ではなく `simTime` の差で補間すること。**
- 一時停止（`set_render_paused`）中は `frame` を作らない。再開すると、止まっている間に
  変化した経路がまとめて次の `frame` に載る。**経路は「変化時のみ」送る仕様なので、
  ここで落とすと二度と届かない。**
- `weather.visibility` は**擬似カメラの描画と正解ラベルの両方が使う唯一の視程**。
  霧で遠方が白く沈むとき、正解ラベル側も同じ距離で打ち切られる
  （見えないものにラベルが付くと、認識器から見てタスクが定義できなくなるため）。
  `weatherAuto` が true の間、`params` の `weatherRain` / `weatherFog` は
  **更新されない**ので、いま出ている天候はこの欄で見ること。

**`detections`（画像認識の可視化）**

擬似カメラ画像から認識器が検出した物体。**PPO が観測として受け取っているのと
同じ検出結果**で、運転席／追従カメラのバウンディングボックス表示に使う。
表示のために別経路で作り直してはいけない（画面と学習が食い違っても外からは
気づけなくなるため）。

- キーは車両スロット番号を文字列化したもの（`vehicles[].id` と同じ番号。JSON
  オブジェクトのキーは文字列でなければならないため）
- 非アクティブな車両はキーに含まれない。1 台もアクティブな車両がいない、または
  認識パイプラインが結果を返せなかったフレームでは `detections` 自体が省略される
- 配列内はクラスごとの枠を守った優先度つきラウンドロビン順
  （車線 → 信号 → 標識 → 車両 → 障害物、各クラス内は信頼度の降順）。
  `config.PERCEP_MAX_DETECTIONS`（12 件）で切り詰めても内訳が欠けないようにするため
- **全スロット分を送るが、通常は追従中の 1 台分しか使われない。**
  バックエンドは追従対象（フロント専用の概念）を知らないため、
  絞るには追従対象を伝えるメッセージが要る。代表的な 1 台 8 件で 639B、
  8 台で 5,153B/frame ＝ 20Hz なら 100.6KB/s のうち 12.5KB/s しか読まれていない
- 座標は擬似カメラの正規化画像座標。**擬似カメラはフロントの運転席カメラと同じ
  内部パラメータ**（視野角 68 度、前方オフセット 0.35m、右オフセット 0.36m、
  視点高さ 1.22m）で描いているため、`box` の値に画面の幅・高さを掛けるだけで
  運転席カメラの映像へそのまま重ねられる

検出オブジェクト 1 個のフィールド:

| フィールド | 型 | 単位・値域 | 説明 |
|---|---|---|---|
| `cls` | number | 0〜4 | 検出クラス。`0`=信号機 / `1`=速度標識 / `2`=車両 / `3`=障害物 / `4`=車線 |
| `box` | `[number, number, number, number]` | 正規化座標 0.0〜1.0 | `[x0, y0, x1, y1]`。左上が `(0,0)`、右下が `(1,1)` |
| `conf` | number | 0.0〜1.0 | 信頼度 |
| `phase` | number（省略可） | 0〜2 | **信号機のみ**。0=青 / 1=黄 / 2=赤。他クラスでは省略される |
| `speedLimit` | number（省略可） | m/s | **速度標識のみ**。規制速度。表示は km/h に換算する（`Math.round(speedLimit * 3.6)`） |
| `distance` | number（省略可） | m | 推定距離。検出できなかった場合は省略される |
| `lateral` | number（省略可） | m | **車線のみ**。車線中心からの横方向偏差（左が正） |
| `lanePoints` | `[number, number][]`（省略可） | **自車座標系** `[前方+x, 左+y]`、単位 m | **車線のみ**。認識した車線中心線の点列（6 点）。起点は自車の真横（前方 0m）でそこから前方へ伸びる。車線幅は中心線から左右 1.6m（計 3.2m）相当。路面へ直接重ねて描くために使う（`frontend/src/scene/LaneDetectionOverlay.tsx`） |

表示名（「信号機：赤」「速度標識：30km/h」など）は転送量を減らすため送られてこない。
組み立て規則は `backend/app/percep/types.py` の `Detection.label` にあり、
フロントの `frontend/src/scene/detectionLabels.ts` が同じ規則で組み立てる
（**どちらかを変えたら両方直すこと**）。

**信号は走行にも影響する。** 各車両は前方の信号までの距離と灯色を観測しており、
赤信号のまま停止線を越えると `rewardSignal` の罰が入る
（道路交通法施行令 2 条「赤色の灯火」）。黄色は「安全に停止できない場合は進行してよい」
ため違反に数えない。守れるようになるかは学習しだいで、学習初期は無視して走る。

### 2.4 `status` — 状態変化時に送信

```jsonc
{
  "type": "status",
  "state": "idle",            // "idle" | "loading_map" | "running" | "error"
  "mapLoaded": true,
  "presetId": "ginza",
  "renderPaused": false,      // 「一時停止」は描画のみ。学習は継続している
  "learning": true,           // state=="running" かつ simSuspended でないとき
  "simSuspended": false,      // ★ 物理と PPO ごと止まっている（認識器の学習中）
  "suspendReason": "",        // simSuspended が true のときの理由
  "message": "マップを読み込みました"
}
```

`renderPaused` と `simSuspended` は**別物**である。

| | 止まるもの | 誰が立てるか |
|---|---|---|
| `renderPaused` | 画面の描画（フレーム配信）だけ。**学習は続く** | 利用者の「一時停止」 |
| `simSuspended` | 物理と PPO。フレームも止まる（画面と通信は生きている） | 認識器の学習（2.9） |

`simSuspended` の間も WebSocket のコマンド・モデルの書き出し・進捗配信は動く。
完了・中断すると**サーバー側が自動で降ろす**ので、利用者が再開させる操作は無い。

### 2.5 `params` — パラメータ変更が反映されたときに送信

```jsonc
{
  "type": "params",
  "params": {
    "vehicleCount": 4,        // アクティブにする車両数 (0..maxVehicles)
    "simSpeed": 1.0,          // 実時間に対する倍率 (0.25..8.0)
    "learningRate": 3e-4,
    "gamma": 0.99,
    "clipRange": 0.2,
    "entropyCoef": 0.001,     // 0.01 は強すぎて方策が潰れる（log_std が上限へ張り付く）
    "rolloutLength": 256,
    "maxSpeed": 13.9,         // m/s
    "rewardGoal": 100.0,
    "rewardCollision": -100.0,
    "rewardProgress": 1.0,
    "rewardOffroad": -1.0,
    "rewardTime": -0.05,
    "rewardSignal": -60.0,    // 赤信号で停止線を越えたときの罰
    "rewardOverspeed": -5.0,  // 規制速度を超え始めたときの罰（-1000.0..0.0）
    "obeySignals": true,      // 信号に従わせるか（false なら罰だけで学習任せ）
    "obeySpeedSigns": true,   // 最高速度標識に従わせるか（false なら罰だけで学習任せ）
    "weatherRain": 0.0,       // 雨の強さ 0.0〜1.0（画を濁らせる。視程は縮めない）
    "weatherFog": 0.0,        // 霧の濃さ 0.0〜1.0（視程を縮める）
    "weatherAuto": false      // true の間はサーバーが simTime から天候を決める
  }
}
```

`rewardOverspeed` は**超え「始めた」ステップに 1 回だけ**入る（超えている間ずっとではない）。
毎ステップ入れると、1 度超えただけで戻るまでの数十ステップぶん罰が積み上がり、
報酬がそれに支配されてしまう。車線逸脱の数え方と同じ考え方。
`rewardSignal`（-60.0）より軽くしてあるのは、速度超過が事故に直結する度合いで
赤信号無視に劣るためで、同格にすると停止挙動の学習を邪魔する。

**天候は観測だけを濁らせる。報酬と終了条件は真値のまま**なので、霧で赤信号を
見落として突っ込めば、そのとおり信号無視として罰が入る。
`weatherAuto` が true のあいだ `weatherRain` / `weatherFog` は**送られてきた値のまま
据え置かれる**（サーバーが上書きしない）。実際に効いている値は `frame.weather` で見ること。

### 2.6 `metrics` — 学習指標（既定 1Hz）

```jsonc
{
  "type": "metrics",
  "tick": 1234,
  "wallTime": 61.7,           // サーバー起動からの実時間秒
  "updates": 42,              // PPO 更新回数
  "episodes": 17,
  "meanEpisodeReward": -12.3,
  "meanEpisodeLength": 143.0,
  "policyLoss": 0.011,
  "valueLoss": 0.52,
  "entropy": 1.13,
  "approxKl": 0.008,
  "collisionRate": 0.21,      // 直近エピソードのうち衝突終了の割合
  "goalRate": 0.44,
  "stepsPerSec": 19.8,
  "signalViolations": 0.16,   // 1 エピソードあたりの信号無視回数
  "speedViolations": 0.03,    // 1 エピソードあたりの速度超過回数
  "laneDeviation": 1.24       // 車線中心からの横方向のずれの平均 [m]
}
```

`signalViolations` と `speedViolations` と `laneDeviation` は、直近 50 エピソードの平均。
`obeySpeedSigns` が true のあいだ `speedViolations` はほぼ 0 になる（環境側が
規制速度を超えさせないため）。**0 が続くこと自体は正常**で、異常の指標としては
`obeySpeedSigns` を false にしたときの値を見る。
`laneDeviation` は経路（＝車線中心線）からの `|横方向偏差|` をエピソード内で平均し、
それをさらにエピソード間で平均した値。市街地の車線幅がおおむね 3m なので、
**1.5m を超えていれば隣の車線や対向車線にはみ出している**とみてよい。

フロントは受信した点をリングバッファに蓄積してグラフ化する（履歴はサーバーから送らない）。

### 2.7 `network` — ネットワークの状態（既定 1Hz、metrics と同じ間隔）

```jsonc
{
  "type": "network",
  "updates": 1234,
  "obsDim": 57,
  "actionDim": 2,
  "hiddenSizes": [128, 128],
  "layers": [
    {
      "name": "policy_trunk.0",   // パラメータ名から `.weight` を除いたもの
      "role": "policy",           // "policy" | "value"
      "inDim": 57,
      "outDim": 128,
      "weightAbsMean": 0.0421,    // |w| の平均。学習が進むと動く
      "weightStd": 0.0688,
      "gradNorm": 0.0135,         // 直近の更新で流れた勾配。0 なら学習していない
      "deltaNorm": 0.0009         // ★ 直前の 1 更新で重みが動いた量
    }
  ],
  "logStd": [-0.4988, -0.5105],   // 探索ノイズの対数標準偏差
  "actionStd": [0.607, 0.600],    // クランプ後に exp したもの（実際に使われる σ）
  "logStdMin": -5.0,
  "logStdMax": 0.0
}
```

`deltaNorm` は**直前の 1 更新**で重みが動いた量。スナップショットの間隔ではなく
**更新ごとに確定する**（PPO の更新はロールアウトが溜まるたび＝実測 2〜3 秒に 1 回
しか起きないので、**更新が無い間は同じ値が続くのが正常**）。
1Hz の差分で出すと「動いていない秒」が多発し、画面が「学習が止まっている」と
誤表示するため、意図してこの定義にしてある。**0 でなければ学習は進んでいる。**

**このスナップショットはシミュレーションスレッドのステップ境界で作る。**
書き出しと同じ理由で、asyncio 側から `state_dict()` を取ると
`optimizer.step()` の途中の中途半端な重みを掴む可能性がある。
重みの複製は約 190KB で、1Hz なら負荷にならない。

`logStd` が `logStdMax` に張り付いている場合、**探索ノイズが行動範囲より広く、
方策が実質ランダムに潰れている**（行動は `[-1, 1]` にクリップされる）。
過去にこの状態で 8,925 更新ぶんの学習が無駄になった。

### 2.8 `error`

```jsonc
{ "type": "error", "code": "MAP_LOAD_FAILED", "message": "Overpass API への接続に失敗しました" }
```

`code` の値は次の 3 つ。

| コード | いつ返るか |
|---|---|
| `MAP_LOAD_FAILED` | 未知のプリセット ID、Overpass API への接続失敗 |
| `INVALID_MESSAGE` | JSON として読めない、オブジェクトでない、未知の `type`、必須項目の欠落、`set_params` の値が非有限または型違い |
| `DETECTOR_TRAINING` | 認識器の学習中に `load_map` が来た（ジョブが握っているマップと画面がずれるため断る） |

**介入（車両追加・障害物設置）の失敗は `error` ではなく `status` メッセージの
`message` で返す。** 「スロットが満杯」「その地点から到達可能な経路が無い」
「障害物が上限」はいずれも利用者の操作に対する説明であって、
接続やプロトコルの異常ではないため。フロントは `status.message` を
そのまま画面に出せばよい。

### 2.9 `detector` — 認識器（CNN）の学習状況

操作パネルの「モデル作成」タブ向け。**接続直後に 1 通**送られ、以後は
**中身が変わったときだけ**（最大 1Hz）送られる。学習中にページをリロードしても、
進行中のジョブがそのまま画面に戻る。

```jsonc
{
  "type": "detector",
  "state": "collecting",   // idle|preparing|evaluating|collecting|training|saving|done|error|cancelled
  "running": true,         // 学習スレッドが走っているか
  "message": "教師データを集めています（1200 / 2400 枚）",
  "progress": 0.5,         // ★ **いまの段階の**進捗。段階をまたいで通算しない
  "collected": 1200,
  "samples": 2400,         // ★ **いまの段階の**分母。採点中は採点に使う枚数（400）になる
  "epoch": 0,
  "epochs": 12,
  "batch": 0,
  "batches": 0,
  "history": [             // エポックごとの損失（完了したエポックだけ）
    { "epoch": 1, "loss": 2.41, "valLoss": 2.88 }
  ],
  "elapsedSec": 31.4,
  "warning": "",           // 写っていないクラスがある等。空なら問題なし
  "paramCount": 152992,
  "presetName": "東京・銀座",
  "request": {             // 受け付けた依頼のエコー
    "mode": "full", "presetId": "ginza",
    "samples": 2400, "epochs": 12, "batchSize": 32, "width": 1.0, "seed": 0,
    "weatherMix": true,    // 晴れ以外の天候も混ぜて集めるか
    "focusWeak": true      // 収集前に採点し、弱点を狙って集めるか
  },
  "evaluation": {          // 収集前の採点。測っていなければ null
    "samples": 400,
    "elapsedSec": 5.2,
    "overallRecall": 0.584,
    "classes": [
      { "cls": 0, "name": "TRAFFIC_LIGHT", "truth": 231, "matched": 143,
        "recall": 0.619, "attributeTotal": 143, "attributeOk": 85,
        "attributeAccuracy": 0.594 }
    ],
    "weathers": [
      { "name": "clear", "truth": 380, "matched": 307, "recall": 0.808 },
      { "name": "fog",   "truth": 318, "matched": 51,  "recall": 0.160 }
    ],
    "weakest": "信号機 の成績が最も低い（37%）。霧でも落ちています"
  },
  "dataset": {             // 集めた教師データの内訳。集めていなければ null
    "samples": 2400,
    "objectCellRatio": 0.107,
    "objectsPerImage": 5.1,
    "classCounts": { "TRAFFIC_LIGHT": 2700, "SPEED_SIGN": 600,
                     "VEHICLE": 1800, "OBSTACLE": 4800, "LANE": 2400 },
    "weatherCounts": { "clear": 720, "drizzle": 300, "rain": 300,
                       "fog": 780, "heavy_fog": 300 }
  },
  "model": {               // data/detector/detector.keras
    "exists": true, "filename": "detector.keras", "sizeBytes": 1965573,
    "modifiedAt": "2026-09-14 01:38:34",
    "inUse": true          // ★ いま観測が CNN 由来か（false なら真値フォールバック）
  },
  "datasetFile": { "exists": true, "sizeBytes": 2513909, "modifiedAt": "..." },
  "limits": {              // サーバーが受け付ける値域。UI のスライダーはこれに合わせる
    "samplesMin": 200, "samplesMax": 4800,
    "epochsMin": 1, "epochsMax": 60,
    "batchMin": 8, "batchMax": 128,
    "widthMin": 0.25, "widthMax": 2.0,
    "seedMin": 0, "seedMax": 999999
  }
}
```

**`classCounts` で件数 0 のクラスは、学習しても検出できるようにならない。**
症状は「走らせてみたら前の車を認識しない」という形でしか出ないので、
サーバーは `warning` にも入れて返す。

`collected` / `samples` は**いまの段階のもの**で、段階をまたいで通算しない。
採点中（`evaluating`）は「200 / 400 枚」、収集中（`collecting`）は「1,200 / 2,400 枚」と
分母が変わる。片方だけを段階に追随させると、採点しているのに収集の枚数が分母に出る。

**`evaluation` は「いまの認識器の成績」であって、これから学習するモデルの成績ではない。**
収集の**前**に現行の `detector.keras` を採点したもので、その結果がそのまま次の収集の
重みになる（成績の低いクラス・天候ほど多く集める）。`focusWeak` が false のとき、
および認識器がまだ 1 つも無いときは `null`。
`classes[].recall` は「真値にある物体のうち検出できた割合」、
`attributeAccuracy` は「検出できたもののうち灯色・規制速度まで合っていた割合」で、
属性を持たないクラス（車両・障害物・車線）では `attributeTotal` が 0 になる。

`model.inUse` は**ファイルがあるか**ではなく**いま実際に使っているか**である。
マップを読み込んだ瞬間（`SimulationEnv` が認識器を読むとき）や、学習完了後の
載せ替えで変わる。**ジョブが動いていなくても変わりうる**ので、配信側は
ジョブの進捗ではなくこのメッセージ全体の中身を比べて送っている。

---

## 3. クライアント → サーバー

```jsonc
{ "type": "load_map",     "presetId": "ginza" }
{ "type": "set_params",   "params": { "vehicleCount": 5 } }   // 部分更新。渡したキーのみ反映
{ "type": "spawn_vehicle","x": 10.0, "y": -20.0 }             // 最寄りの道路上にスナップされる
{ "type": "despawn_vehicle", "id": 2 }
{ "type": "add_obstacle", "x": 10.0, "y": -20.0, "radius": 0.5 }
{ "type": "remove_obstacle", "id": 3 }
{ "type": "clear_obstacles" }
{ "type": "set_render_paused", "paused": true }               // 描画のみ停止。学習は継続
{ "type": "reset_episode" }
{ "type": "save_checkpoint" }
{ "type": "load_checkpoint" }
{ "type": "reset_policy" }                                    // 重みを初期化して学習をやり直す
{ "type": "set_network", "hiddenSizes": [128, 128, 128] }     // 隠れ層の構成を変える
{ "type": "start_detector_training",                          // 認識器（CNN）を学習する
  "request": { "mode": "full", "presetId": "ginza",
               "samples": 2400, "epochs": 12, "batchSize": 32,
               "width": 1.0, "seed": 0,
               "weatherMix": true, "focusWeak": true } }
{ "type": "cancel_detector_training" }                        // 中断（すぐには止まらない）
{ "type": "ping" }                                            // → {"type":"pong","t":<server epoch ms>}
```

`set_network` は隠れ層の構成を変える。**重みは引き継げない**（層の形が変わるので
`load_state_dict` が通らない）ため、学習は 0 からやり直しになる。
1〜4 層・各層 16〜512 の範囲外は `INVALID_MESSAGE` で弾き、**構成は変えない**。
適用はエンジンスレッドのステップ境界で行う（学習器を作り直すため、
asyncio 側から触ると更新中の重みを壊す）。
入力（`obsDim`）と出力（`actionDim`）は変わらない。

`start_detector_training` は擬似カメラ画像から信号・標識・車線・車両・障害物を
検出する CNN を学習する（CLI の `backend/train_detector.py` と**同じ実装**を呼ぶ）。

| `mode` | 何をするか | CLI での相当 |
|---|---|---|
| `full` | 教師データを集めてから学習する | 引数なし |
| `collect` | 集めて保存するだけ | `--collect-only` |
| `train` | 保存済みの教師データで学習する | `--train-only` |

- `presetId` を省略（または `null`）にすると、**いま読み込んでいるマップ**を使う。
  読み込み済みのものと同じなら地図を読み直さない（`groundtruth` の静的キャッシュを
  走行側と取り合わないためでもある）。
- `seed` は収集の乱数種（省略時 0）。`SimulationEnv` と行動のランダム化の両方に
  渡るので、**変えるだけで教師データの多様性が上がる**。逆に同じ種なら何度回しても
  同じ画が集まる（再現性が要るときはここを固定する）。CLI の `--seed` と同じ。
- 値域（2.9 の `limits`）を外れた値は**丸めずに** `INVALID_MESSAGE` で弾く。
  `set_params` と違い、押した瞬間に数十分動き出す操作なので、
  指定と違う値で走り出すほうが危ないため。
- `weatherMix` は晴れ以外の天候（小雨・雨・霧・濃霧）も混ぜて集めるかどうか。
  **走行中の天候（`params.weatherRain` / `weatherFog`）とは無関係**で、
  収集は収集で天候を選び直す。
- `focusWeak` は収集の前に現行の認識器を採点し、**成績の低いクラスと天候を多めに
  集める**かどうか。採点の結果は `detector.evaluation` に載る。認識器がまだ無い
  初回は採点をとばして一様に集める（採点する相手がいないため）。
- **認識器の学習中は `load_map` を受け付けない**（`DETECTOR_TRAINING` で断る）。
  ジョブは開始時に借りたマップの参照を握り続けるので、通すと
  「収集は元のエリアのまま、画面だけ新しいエリア」という食い違いが起きる。
- **実行中は `status.simSuspended` が true になり、物理と PPO が止まる。**
  完了・中断でサーバーが自動的に降ろす。
- 二重起動・教師データ不足は `error` ではなく `status.message` で返す
  （利用者の操作に対する説明であって、プロトコルの異常ではないため）。
- `cancel_detector_training` を送っても**すぐには止まらない**。いま処理中の
  バッチ（またはステップ）の切れ目まで進んでから終わる。
  **中断したモデルは保存しない**ので、それまでの認識器はそのまま残る。

不正なメッセージには `error` (`INVALID_MESSAGE`) を返し、接続は維持する。

---

## 4. HTTP エンドポイント

WebSocket に載せないものはここに置く。バイナリの受け渡しは、進捗・保存先の選択・中断を
ブラウザ本来の仕組みに任せられる HTTP のほうが素直なため。

### `GET /api/health`

```jsonc
{
  "status": "ok",
  "protocolVersion": 1,
  "connections": 1,
  "engine": { /* 2.4 の status ペイロードと同じ形 */ }
}
```

### `GET /api/export/{kind}` — 学習済みモデルの書き出し

`kind` は `checkpoint` / `torchscript` / `keras`。

| kind | 中身 | 用途 |
|---|---|---|
| `checkpoint` | 重み＋オプティマイザ状態＋メタデータ（`torch.save` 形式） | このアプリに読み戻して**続きから学習**する。バックアップ |
| `torchscript` | 推論だけを切り出した TorchScript（`metadata.json` を同梱） | このリポジトリのコード無しで `torch.jit.load()` するだけで動く |
| `keras` | 同じネットワークを Keras 3 のモデルとして組み直したもの（`.keras`）| Keras / TensorFlow 系のツールで扱う |

- 成功時: `200`、`Content-Disposition: attachment; filename="autoware-sim_<preset>_upd<更新回数>_<日時>.pt"`
  - 補助ヘッダ `X-Export-Kind` と `X-Export-Size`（バイト数）も返す
- 未知の `kind`: `400` — `{"error": "...", "supported": ["checkpoint", "torchscript", "keras"]}`
- 書き出し失敗: `500` — `{"error": "..."}`
- 時間内に完了しない: `504` — `{"error": "..."}`

書き出しは**シミュレーションスレッドのステップ境界**で行う。asyncio 側から直接
`state_dict()` を取ると `optimizer.step()` の途中の中途半端な重みを掴む可能性があるため。
学習は止まらず、1 ステップ分だけ余分に時間がかかるだけである。

生成したファイルはサーバー上の `backend/data/exports/` にも残る（ダウンロードに失敗しても
手元に残るようにするため）。

TorchScript の入出力：

```
forward(obs: float32[B, 57]) -> (action: float32[B, 2], value: float32[B])
```

`action` は方策分布の平均を `[-1, 1]` にクリップした決定論的な行動。
学習時と同じ確率的な行動が欲しい場合は、同梱の buffer `log_std` を使って
`Normal(action, exp(log_std))` からサンプリングする。
`log_std` は学習時と同じ可動域（`PPO_LOG_STD_MIN`〜`MAX` = -5.0〜0.0）へ
丸めた値が入る。メタデータの `policy.logStd` も同じ値で、`policy.logStdRange` に
可動域そのものが入っている。

Keras 版の入出力：

```
model(obs: float32[B, 57]) -> [action: float32[B, 2], value: float32[B]]
```

`value` の Dense(1) 出力は素のままだと `[B, 1]` になるが、TorchScript 版
（`squeeze(-1)`）と揃えるため `Reshape` で `[B]` に落としてある
（code_review L-09。以前はここに shape の違いを明記するだけだった）。

`keras.saving.load_model()` で読める。**標準の Dense 層だけで構成しているので
`custom_objects` は不要**。行動のクリップは `hard_tanh` 活性で表しており、
`clip(x, -1, 1)` と厳密に一致する。`.keras` は zip なので、メタデータは
`autoware_sim_metadata.json` として同じ zip に同梱している（Keras は自分が知っている
エントリしか読まないため、追記しても読み込みには影響しない）。
`log_std` は Keras の層として表せないので、このメタデータの `policy.logStd` に入れている。

**`torchscript` と `keras` は推論専用で、オプティマイザ状態を持たないため
`POST /api/import` では学習を再開できない。** 取り違えたときは形式を判定して案内する。

### `POST /api/import` — 書き出したモデルから学習を再開

`multipart/form-data` で `file` に **`checkpoint` 形式の `.pt`** を送る。
TorchScript や Keras 形式を渡した場合は、その旨を説明する `400` を返す。

成功時（`200`）:

```jsonc
{
  "ok": true,
  "filename": "autoware-sim_ginza_upd585_20260905-163218.pt",
  "sizeBytes": 578601,
  "checkpoint": {
    "updates": 585,
    "obsDim": 57,
    "actionDim": 2,
    "hiddenSizes": [128, 128],
    "hasOptimizer": true,          // false だと学習の立ち上がりが鈍る
    "exportedAt": "2026-09-05T16:32:18+09:00",
    "presetId": "ginza",
    "presetName": "東京・銀座",
    "metrics": { /* 書き出し時点の指標 */ }
  },
  "backup": {                       // 上書き前に自動退避した、それまでのモデル
    "filename": "autoware-sim_ginza_upd594_before-import_20260905-163259.pt",
    "sizeBytes": 578601
  },
  "message": "学習回数 585 回の状態から学習を再開します"
}
```

失敗時は `{"ok": false, "error": "<日本語の理由>"}`。
利用者が直せる失敗（形式違い・次元不一致など）は `400`、
大きすぎるファイルは `413`、時間切れは `504`、サーバー内部の失敗は `500`。

**安全上の扱い**: 受け取るのは外部から来たファイルなので、
`torch.load(weights_only=True)` で解析し、テンソルと素の Python 値以外が
含まれていれば読み込みを中止する（ピクルによる任意コード実行を防ぐため）。
`export.py` が `torch.__version__` を `str()` で包んでいるのはこの経路を通すためで、
`TorchVersion` オブジェクトのままだと安全モードに弾かれる。

**上書きの扱い**: 読み込みは現在の学習状態を不可逆に置き換えるため、
載せ替える直前に現在のモデルを `backend/data/exports/` へ
`..._before-import_...pt` という名前で自動退避する。
また、読み込み後ただちに `backend/data/checkpoints/shared_policy.pt` を更新するので、
サーバーを再起動しても読み込んだ状態が残る。

**復元される内容**: 方策と価値関数の重み、`log_std`、Adam のモーメント、更新回数。
ロールアウトバッファは現在の環境と結び付いているためクリアされる。
エピソード統計（到達率・衝突率など）は前のポリシーのものなので破棄される。

---

## 5. 走行のルール（日本の道路交通法）

経路は道路中心線ではなく**車線に沿って**生成される。走行側（強化学習）は
その経路を追うだけでよく、車線判断のロジックを持たない。

| ルール | 根拠 | 実装 |
|---|---|---|
| 左側通行 | 17 条 4 項 | 対面通行路では中心線の左半分の車線だけを使う |
| 一番左の車線を走る | 20 条 1 項 | 経路は既定で lane 0（最も左）を通る |
| 左折時は左側端に寄る | 34 条 1 項 | 左折・直進の手前は lane 0 のまま |
| 右折時は中央に寄る | 34 条 2 項 | 右折の 35m 手前から自分の方向の最も右の車線へ滑らかに移る |
| 赤信号で停止線を越えない | 施行令 2 条 | `obeySignals` が true なら**停止線の手前で必ず止まる**。越えると `rewardSignal` の罰 |
| 最高速度を超えない | 22 条 1 項 | `obeySpeedSigns` が true なら**規制速度を超えない**。超えると `rewardOverspeed` の罰 |

車線変更は「経路の横方向オフセットを手前で滑らかに変える」ことで表現している。
別の判断層を持たせず経路に織り込むので、交差点で急に車線をまたぐことがない。

### 最高速度標識に対する操作（`obeySpeedSigns` が true のとき）

信号と同じく**環境の性質**として扱う。経路を張るときに「弧長 → 規制速度」の区切りを
求めておき、毎ステップ現在位置の規制速度を引いて、カーブ・信号と合わせて
**最も厳しい上限**を採る。エージェントの指令は書き換えず、上限を超えないところまで
加速指令を丸めるだけ（`constrain_accel`）。

規制速度は車両側の上限（`maxSpeed`、既定 13.9 m/s = 50km/h）とは別物で、
**道路ごとに変わる**。銀座では 30km/h が 65 基・40km/h が 35 基・50km/h が 56 基。

実測（銀座・16 台・600 ステップ全開加速）:

| | 規制速度の最大超過 |
|---|---|
| `obeySpeedSigns: true` | **0.00 km/h**（規制速度の 100% まで到達はする） |
| `obeySpeedSigns: false` | 20.04 km/h |

速度超過の判定にはヒステリシスがある。`+0.5 m/s` を超えたら「超過」、
`+0.1 m/s` を下回ったら「復帰」。規制速度ぴったりで走ったときに、
数値誤差で回数が跳ね上がらないようにするため。

### 信号に対する操作（`obeySignals` が true のとき）

灯色ごとに、次のステップの速度に上限を掛ける。エージェントの指令を書き換えるのではなく
**環境の性質**として扱うので、PPO から見れば「赤信号に近づくとアクセルが効かなくなる環境」
でしかなく、学習は成立する。

| 灯色 | 与える操作 |
|---|---|
| **青** | 制限なし。指令をそのまま通す |
| **赤** | 停止線の手前で止まりきれる速度まで抑える。`u^2/(2a) + u·dt <= 停止線までの距離 - 1m` を解いた `u` が上限 |
| **黄** | 安全に止まれるなら赤と同じ。止まれないなら制限なし（施行令 2 条ただし書き）|

黄色で「止まれない」と判断して通した信号は、赤に変わってから停止線を越えても違反に数えない。

制動には最大減速度の 80%（4.8 m/s²）を使い、残りを余裕として残す。
連続時間の `v=sqrt(2·a·d)` ではなく上の離散時間の解を使うのは、1 ステップで進む分
（最大 0.7m）を見込まないと毎ステップ食い込んで停止線を越えてしまうため。

交差点では前後の車線中心線がつながらないため、両端を切り詰めてベジエ曲線で接続する。
これが右左折の軌跡になる。

---

## 6. 設計上の約束（memo/memo.md 5章の反映）

| 約束 | 実装箇所 |
|---|---|
| 一時停止は描画のみ。学習は止めない | `set_render_paused` は `frame` 配信を止めるだけ。物理・学習スレッドは回り続ける |
| ユーザー介入中も学習を継続 | 介入イベントは学習ループの外から `queue.Queue` で投入し、ステップ境界で適用する |
| 建物との衝突判定はバックエンド | `frame.vehicles[].collided` がサーバー判定の結果。フロントは見た目を合わせるだけ |
| 学習と描画の頻度を分離 | サーバー 20Hz 配信 / フロント 60fps 描画。フロントが `frame` 間を補間する |
| 擬似固定エージェント数 | `frame.vehicles` は常に `maxVehicles` 個。`active` フラグでマスク |
