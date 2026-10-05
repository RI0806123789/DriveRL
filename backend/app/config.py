"""アプリ全体で共有する定数と設定。"""

from __future__ import annotations

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BACKEND_DIR.parent

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_DIR / ".env")
except ImportError:
    pass


DATA_DIR = BACKEND_DIR / "data"
MAP_CACHE_DIR = DATA_DIR / "map_cache"
OSMNX_CACHE_DIR = DATA_DIR / "osmnx_cache"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
EXPORT_DIR = DATA_DIR / "exports"
UPLOAD_DIR = DATA_DIR / "uploads"
#: ハイパーパラメータの自動探索（`runtime/autotune.py`）の履歴（SQLite）・最良のパラメータ・試行の一覧
TUNING_DIR = DATA_DIR / "tuning"
#: 学習タブの設定値（学習率・報酬の重み・スイッチ）の控え。起動時に戻す（`runtime/param_store.py`）
LEARNING_PARAMS_PATH = DATA_DIR / "learning_params.json"

for _d in (DATA_DIR, MAP_CACHE_DIR, OSMNX_CACHE_DIR, CHECKPOINT_DIR, EXPORT_DIR, UPLOAD_DIR, TUNING_DIR):
    _d.mkdir(parents=True, exist_ok=True)


CHECKPOINT_PATH = CHECKPOINT_DIR / "shared_policy.pt"
#: 自動探索の最良の試行の重み。本番の `CHECKPOINT_PATH` とは分けて置く（探索中に本番を上書きしない）
BEST_TUNED_POLICY_PATH = CHECKPOINT_DIR / "best_tuned_policy.pt"

HOST = os.getenv("DRIVERL_HOST", "127.0.0.1")
PORT = int(os.getenv("DRIVERL_PORT", "8000"))

#: 実用モードの AI コンシェルジュ（`runtime/concierge.py`）が使う Gemini の API キー。空なら使えない（画面のボタンも押せない）
GEMINI_API_KEY_PLACEHOLDER = "your_gemini_api_key_here"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if GEMINI_API_KEY == GEMINI_API_KEY_PLACEHOLDER:
    # .env.example をそのまま複製しただけなら、キーは無いものとして扱う
    GEMINI_API_KEY = ""
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "").strip() or "gemini-3.1-flash-lite"
#: GEMINI_MODEL が混み合っているとき、再試行の最後の 1 回だけ使うモデル。空なら GEMINI_MODEL のまま試し直す
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "").strip()

PROTOCOL_VERSION = 2

CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

#: IP アドレスと localhost のほかに、Host として受け付ける名前（カンマ区切り。例: mypc.local）。app/host_guard.py
ALLOWED_HOSTS = [h.strip() for h in os.getenv("DRIVERL_ALLOWED_HOSTS", "").split(",") if h.strip()]

SIM_HZ = 20.0
DT = 1.0 / SIM_HZ

METRICS_HZ = 1.0

MAX_VEHICLES = 8
MAX_OBSTACLES = 64
MAX_PEDESTRIANS = 64

MAX_EPISODE_STEPS = 4000

VEHICLE_LENGTH = 4.4
VEHICLE_WIDTH = 1.8
VEHICLE_HEIGHT = 1.45
WHEELBASE = 2.6

MAX_STEER = 0.55
MAX_ACCEL = 3.0
MAX_DECEL = -6.0
MAX_SPEED = 13.9
STEER_RATE = 2.5

MAX_LATERAL_ACCEL = 4.5

OBSTACLE_RADIUS = 0.35
OBSTACLE_HEIGHT = 0.75

PEDESTRIAN_HEIGHT = 1.68
PEDESTRIAN_RADIUS = 0.26
PEDESTRIAN_WIDTH = PEDESTRIAN_RADIUS * 2.0
PEDESTRIAN_SPEED = 1.35
PEDESTRIAN_SPEED_SPREAD = 0.35
PEDESTRIAN_SIDEWALK_MARGIN = 1.7
PEDESTRIAN_CROSS_PROB = 0.28
PEDESTRIAN_CROSS_SETBACK_M = 4.0

OBS_SELF_DIM = 2
OBS_GOAL_DIM = 3
OBS_ROUTE_POINTS = 5
OBS_ROUTE_DIM = OBS_ROUTE_POINTS * 2

OBS_LANE_DIM = 4

OBS_SIGNAL_DIM = 5

OBS_SIGN_DIM = 3

OBS_VEHICLE_COUNT = 3
OBS_VEHICLE_FIELDS = 4
OBS_VEHICLE_DIM = OBS_VEHICLE_COUNT * OBS_VEHICLE_FIELDS

OBS_OBSTACLE_COUNT = 3
OBS_OBSTACLE_FIELDS = 3
OBS_OBSTACLE_DIM = OBS_OBSTACLE_COUNT * OBS_OBSTACLE_FIELDS

OBS_PEDESTRIAN_COUNT = 3
OBS_PEDESTRIAN_FIELDS = 3
OBS_PEDESTRIAN_DIM = OBS_PEDESTRIAN_COUNT * OBS_PEDESTRIAN_FIELDS

OBS_FREESPACE_DIM = 9

# 周囲カメラ（後方・左・右）ごとに、いちばん近い車両・障害物・歩行者 1 つ
OBS_SURROUND_CAMERAS = 3
OBS_SURROUND_FIELDS = 3
OBS_SURROUND_DIM = OBS_SURROUND_CAMERAS * OBS_SURROUND_FIELDS

# 車車間通信（V2X）で近くの車から受け取ったメッセージの平均（車速・右左折の意図・危険・交差点への近さ）。sim/v2x.py
OBS_V2X_DIM = 4

# 4 台のカメラだけから作った見通しと死角（左右の見通し距離・前方の遮蔽率・角までの距離・前後左右の見えている割合）。percep/occlusion.py
OBS_OCCLUSION_DIM = 8

# 観測ベクトルの連結順と区画の大きさ。percep/encoder.py の添字と書き出しのメタデータはここから導く
# 後から足した欄は必ず末尾に置く（旧い重みを 0 埋めで読み込めるのはこのため。rl/ppo.py）
OBS_LAYOUT: tuple[tuple[str, int], ...] = (
    ("self", OBS_SELF_DIM),
    ("goal", OBS_GOAL_DIM),
    ("route", OBS_ROUTE_DIM),
    ("lane", OBS_LANE_DIM),
    ("signal", OBS_SIGNAL_DIM),
    ("sign", OBS_SIGN_DIM),
    ("vehicles", OBS_VEHICLE_DIM),
    ("obstacles", OBS_OBSTACLE_DIM),
    ("pedestrians", OBS_PEDESTRIAN_DIM),
    ("freespace", OBS_FREESPACE_DIM),
    ("surround", OBS_SURROUND_DIM),
    ("v2x", OBS_V2X_DIM),
    ("occlusion", OBS_OCCLUSION_DIM),
)

OBS_DIM = sum(size for _name, size in OBS_LAYOUT)
#: 周囲カメラの欄を足す前（66）・V2X の欄を足す前（75）・死角の欄を足す前（79）の観測の次元
OBS_DIM_BEFORE_OCCLUSION = OBS_DIM - OBS_OCCLUSION_DIM
OBS_DIM_BEFORE_V2X = OBS_DIM_BEFORE_OCCLUSION - OBS_V2X_DIM
OBS_DIM_BEFORE_SURROUND = OBS_DIM_BEFORE_V2X - OBS_SURROUND_DIM
#: この次元のチェックポイントは、入力の末尾に 0 の列を足して読み込める
OBS_WIDENABLE_DIMS: tuple[int, ...] = (
    OBS_DIM_BEFORE_SURROUND,
    OBS_DIM_BEFORE_V2X,
    OBS_DIM_BEFORE_OCCLUSION,
)

ACTION_DIM = 2

OBS_ROUTE_SPACING = 6.0
OBS_VEHICLE_RANGE = 40.0
OBS_OBSTACLE_RANGE = 30.0
OBS_PEDESTRIAN_RANGE = 30.0
OBS_FREESPACE_MAX_DISTANCE = 30.0
OBS_GOAL_RANGE = 300.0
OBS_LATERAL_RANGE = 8.0
OBS_SIGNAL_RANGE = 60.0
OBS_SURROUND_RANGE = 20.0
#: V2X で届く距離 [m] と、メッセージを受け取る相手の数（近い順）
V2X_RANGE_M = 30.0
V2X_MAX_PEERS = 2

#: 後退の速さの上限 [m/s]。後退するのは安全ギミックの切り返しだけ（sim/safety.py）
REVERSE_MAX_SPEED = 2.0

#: 周囲カメラ（後方・左・右）を CNN に通す枚数の上限 [枚/ステップ]。前方 8 台だけで 50ms を使い切るため、
#: 周囲は優先度の高いものから順に回す（真値で走るときは毎ステップ全部作る）
SURROUND_CNN_IMAGES_PER_STEP = 3

DETECTOR_DIR = DATA_DIR / "detector"
DETECTOR_PATH = DETECTOR_DIR / "detector.keras"
DETECTOR_DATASET_DIR = DETECTOR_DIR / "dataset"
DETECTOR_DIR.mkdir(parents=True, exist_ok=True)

#: 認識器の推論を動かす先。auto = NPU があれば NPU・無ければ OpenVINO の CPU / cpu = OpenVINO の CPU /
#: keras = OpenVINO を使わず従来の Keras（torch）の CPU。openvino が入っていなければ auto でも Keras になる
PERCEP_DEVICE = os.getenv("DRIVERL_PERCEP_DEVICE", "auto").strip().lower() or "auto"
#: 1 回の推論に通す画像の最大枚数（前方は全車 + 周囲カメラの予算）。NPU の形の固定（バケット）の上限になる
PERCEP_MAX_BATCH = MAX_VEHICLES + SURROUND_CNN_IMAGES_PER_STEP

PERCEP_MAX_DETECTIONS = 15

PERCEP_CONF_THRESHOLD = 0.35

PERCEP_FALLBACK_GROUND_TRUTH = True

GOAL_RADIUS = 8.0
OFFROAD_LIMIT = 12.0

SIGNAL_STOP_TOLERANCE_M = 1.0

PPO_HIDDEN_SIZES = (128, 128)
PPO_HIDDEN_MIN_LAYERS = 1
PPO_HIDDEN_MAX_LAYERS = 4
PPO_HIDDEN_MIN_WIDTH = 16
PPO_HIDDEN_MAX_WIDTH = 512
PPO_EPOCHS = 4
PPO_MINIBATCHES = 4

PPO_MINIBATCHES_PER_STEP = 1

FRAME_HZ = 60.0
PPO_GAE_LAMBDA = 0.95
PPO_VALUE_COEF = 0.5
PPO_MAX_GRAD_NORM = 0.5
PPO_LOG_STD_INIT = -0.5
PPO_LOG_STD_MIN = -5.0
PPO_LOG_STD_MAX = 0.0
#: エキスパート（経路追従）が運転したステップに掛ける模倣の損失の重み（`rl/online_assist.py`）
PPO_BC_COEF = 0.5
#: 模倣の教師の操作はここで頭打ちにする（方策の平均は tanh なので ±1 には届かない）
PPO_BC_TEACH_LIMIT = 0.98

#: 階層型の方策のマクロな意図（`rl/hierarchical_policy.py`）。並びが上位方策の出力の添字
HRL_OPTIONS = ("CRUISE", "FOLLOW", "YIELD", "STOP")
#: 上位方策が意図を選び直す周期 [ステップ]（20 ステップ = 1 秒）
HRL_OPTION_STEPS = 20
#: 上位方策のエントロピーの重み
HRL_META_ENTROPY_COEF = 0.01
#: 下位方策の報酬の整形。操作の変化の二乗・車線中心からの横ずれ [m] の二乗・意図の速度帯からの外れ
HRL_JERK_COEF = 0.05
HRL_LANE_COEF = 0.01
HRL_CONSISTENCY_COEF = 0.05
#: 意図ごとの速度帯 [m/s]。CRUISE は下限だけ、YIELD と STOP は上限だけ、FOLLOW は前走車の速さに合わせる
HRL_CRUISE_MIN_MPS = 5.0
HRL_YIELD_MAX_MPS = 2.78
HRL_STOP_MAX_MPS = 0.5
#: 速度帯からこれだけ外れると整形の罰が頭打ちになる [m/s]
HRL_CONSISTENCY_SPAN_MPS = 3.0
#: 意図の初期の偏り（アクセルの tanh の前）。CRUISE は前へ、STOP は後ろへ引く
HRL_OPTION_ACCEL_PRIOR = (0.4, 0.1, -0.3, -0.7)

TORCH_NUM_THREADS = 4

LANE_DEPARTURE_M = 1.75
LANE_RETURN_M = 1.25

SIGNAL_GREEN_SEC = 25.0
SIGNAL_YELLOW_SEC = 3.0
SIGNAL_ALL_RED_SEC = 2.0
SIGNAL_GREEN_MIN_SEC = 6.0

SPEED_SIGN_DIAMETER = 0.6
SPEED_SIGN_BOTTOM_HEIGHT = 1.8
SPEED_SIGN_POLE_RADIUS = 0.05
SPEED_SIGN_SIDE_MARGIN = 0.8
SPEED_SIGN_SETBACK_M = 12.0

OVERSPEED_TOLERANCE_MPS = 0.5
OVERSPEED_RETURN_MPS = 0.1

DEFAULT_BUILDING_HEIGHT = 9.0
BUILDING_LEVEL_HEIGHT = 3.0
BUILDING_SIMPLIFY_TOLERANCE = 0.5
BUILDING_MIN_AREA = 8.0

DEFAULT_LANE_WIDTH = 3.25
MIN_ROAD_WIDTH = 5.0

GRID_CELL_SIZE = 1.0
GRID_MARGIN = 40.0

ROUTE_RESAMPLE_M = 2.0
