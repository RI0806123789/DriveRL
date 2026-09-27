/** モックの交通（車両・歩行者・障害物・徒歩キャラ）。格子の道を走らせるだけの擬似的な再現 */

import type { NpcPedestrianState, ObstacleState, SimParams, Vec2, VehicleState } from '../../types/protocol.ts'
import { SIGNAL_GREEN, SIGNAL_RED } from '../../types/protocol.ts'
import { GRID_N, GRID_SPACING, ROAD_WIDTH, SIM_HZ, clampGrid, hopKey, nodeX, nodeY, snapToGrid } from './grid.ts'
import type { Rng } from './grid.ts'
import type { MockSignals } from './map.ts'

/** 停止に使う減速度 [m/s^2]（signals.BRAKE_USE_RATIO 相当の値） */
const BRAKE_ACCEL = 4.8
/** 前走車との車間 [m]。車長 4.4m + 余裕 */
const CAR_GAP_M = 6.0

/** 徒歩キャラの手前で空ける距離 [m]・見る範囲 [m]・車体中心からの横幅 [m]（env.py と同じ値） */
const PLAYER_MARGIN_M = 4.4 / 2 + 0.26 + 3.0
const PLAYER_RANGE_M = 30.0
const PLAYER_HALF_WIDTH_M = 2.0
/** 位置が届かなくなってから街から消すまで [ms]（engine.PLAYER_POSE_TTL_SEC と同じ） */
const PLAYER_POSE_TTL_MS = 1000

/** 歩道が車道中心から離れている距離 [m]（バックエンドの PEDESTRIAN_SIDEWALK_MARGIN 相当） */
const WALK_OFFSET_M = ROAD_WIDTH / 2 + 1.7

/** これより遠い歩行者は車の近くへ回す [m]（バックエンドの RECYCLE_FAR_M 相当） */
const PEDESTRIAN_RECYCLE_M = 220

/** 方向指示器を出し始める距離 [m]（バックエンドの TURN_LOOKAHEAD_M 相当） */
const TURN_LOOKAHEAD_M = 30

/** モックの歩行者。格子の道に沿って歩き、交差点でたまに車道を横断する */
interface MockPedestrian {
  id: number
  gx: number
  gy: number
  /** 次に向かうノード */
  tx: number
  ty: number
  /** 区間の進み具合 0.0〜1.0 */
  t: number
  /** 歩道の左右。+1 が進行方向の左 */
  side: number
  /** 横断の進み具合 0.0〜1.0。0 なら歩道 */
  cross: number
  crossing: boolean
  stride: number
  speed: number
}

export interface MockVehicle {
  id: number
  active: boolean
  gx: number
  gy: number
  /** 次に向かうノード */
  tx: number
  ty: number
  /** 現在エッジ上の進捗 0..1 */
  t: number
  x: number
  y: number
  heading: number
  speed: number
  steer: number
  /** 加速指令 -1..1（ペダルの踏み込み）。`braking` と同じ出どころにする */
  throttle: number
  collided: boolean
  reachedGoal: boolean
  /** いま適用されている規制速度 [m/s]。まだ標識を 1 基も通っていなければ 0 */
  speedLimit: number
  /** このエピソード中に規制速度を超えた回数 */
  speedViolations: number
  /** 前ステップで超過していたか。超え「始めた」瞬間だけを数えるため */
  overspeeding: boolean
  goalGx: number
  goalGy: number
  /** 目的地を決めた時点での残り距離 [m]。progress の分母 */
  routeTotal: number
  /** 経路の達成度 0..1（protocol.md 2.3 の progress） */
  progress: number
  /** route を送るべきか */
  routeDirty: boolean
  /** 制動指令が出ているか（ブレーキランプ） */
  braking: boolean
  /** 方向指示器。-1=左 / 0=消灯 / +1=右 */
  turnSignal: number
}

/** 目的地へ向かう次のノードを貪欲に選ぶ（x 方向を先に詰める） */
export function chooseNext(v: MockVehicle, rng: Rng): void {
  const dx = v.goalGx - v.gx
  const dy = v.goalGy - v.gy
  if (dx === 0 && dy === 0) return
  const preferX = dy === 0 ? true : dx === 0 ? false : rng() < 0.5
  if (dx !== 0 && preferX) {
    v.tx = clampGrid(v.gx + Math.sign(dx))
    v.ty = v.gy
  } else if (dy !== 0) {
    v.tx = v.gx
    v.ty = clampGrid(v.gy + Math.sign(dy))
  } else {
    v.tx = clampGrid(v.gx + Math.sign(dx))
    v.ty = v.gy
  }
}

/** 目的地までの残り距離 [m]。 */
export function remainingDistance(v: MockVehicle): number {
  const segRest = Math.hypot(nodeX(v.tx) - v.x, nodeY(v.ty) - v.y)
  const grid = (Math.abs(v.goalGx - v.tx) + Math.abs(v.goalGy - v.ty)) * GRID_SPACING
  return segRest + grid
}

/** 現在地から目的地までの経路をノード列で作る（x → y の順に詰める） */
export function buildRoute(v: MockVehicle): Vec2[] {
  const pts: Vec2[] = [[v.x, v.y]]
  let gx = v.tx
  let gy = v.ty
  pts.push([nodeX(gx), nodeY(gy)])
  let guard = 0
  while ((gx !== v.goalGx || gy !== v.goalGy) && guard++ < 64) {
    if (gx !== v.goalGx) gx += Math.sign(v.goalGx - gx)
    else gy += Math.sign(v.goalGy - gy)
    pts.push([nodeX(gx), nodeY(gy)])
  }
  return pts
}

/** 1 ステップぶんの外からの条件 */
export interface TrafficStep {
  dt: number
  tick: number
  params: SimParams
  /** いまの灯色（`MockSignals.phases()`） */
  phases: readonly number[]
  signals: MockSignals
  /** 配車に徴用している車。乗降地点で止め、respawn させない（いなければ -1） */
  heldVehicle: number
}

export class MockTraffic {
  readonly vehicles: MockVehicle[] = []
  obstacles: ObstacleState[] = []
  /** 終わったエピソードの数（metrics の episodes） */
  episodes = 0
  private readonly rng: Rng
  private pedestrians: MockPedestrian[] = []
  private obstacleSeq = 1
  /** 徒歩キャラ。届かなくなったら消す（実機の TTL と同じ） */
  private player: Vec2 | null = null
  private playerAt = 0

  constructor(rng: Rng, maxVehicles: number) {
    this.rng = rng
    for (let i = 0; i < maxVehicles; i++) {
      this.vehicles.push(this.makeVehicle(i))
    }
  }

  activeCount(): number {
    return this.vehicles.filter((v) => v.active).length
  }

  applyVehicleCount(n: number): void {
    for (const v of this.vehicles) {
      const shouldBeActive = v.id < n
      if (shouldBeActive && !v.active) {
        v.active = true
        v.routeDirty = true
      } else if (!shouldBeActive && v.active) {
        v.active = false
      }
    }
  }

  applyPedestrianCount(n: number, max: number): void {
    const want = Math.max(0, Math.min(max, Math.round(n)))
    while (this.pedestrians.length < want) {
      this.pedestrians.push(this.makePedestrian(this.pedestrians.length))
    }
    if (this.pedestrians.length > want) this.pedestrians.length = want
  }

  /** 地図を読み直したとき。全車の経路を送り直す */
  markRoutesDirty(): void {
    for (const v of this.vehicles) v.routeDirty = true
  }

  /** 空いているスロットに車を出す。空きが無ければ null */
  spawnAt(x: number, y: number): MockVehicle | null {
    const slot = this.vehicles.find((v) => !v.active)
    if (!slot) return null
    const { gx, gy } = snapToGrid(x, y)
    slot.gx = gx
    slot.gy = gy
    slot.tx = gx
    slot.ty = gy
    slot.t = 0
    slot.x = nodeX(gx)
    slot.y = nodeY(gy)
    slot.speed = 0
    slot.active = true
    slot.routeDirty = true
    slot.goalGx = Math.floor(this.rng() * GRID_N)
    slot.goalGy = Math.floor(this.rng() * GRID_N)
    chooseNext(slot, this.rng)
    return slot
  }

  despawn(id: number): void {
    const v = this.vehicles[id]
    if (v) v.active = false
  }

  /** 全車を街のどこかへ置き直す（reset_episode） */
  resetAll(): void {
    for (const v of this.vehicles) {
      const gx = Math.floor(this.rng() * GRID_N)
      const gy = Math.floor(this.rng() * GRID_N)
      v.gx = gx
      v.gy = gy
      v.tx = gx
      v.ty = gy
      v.t = 0
      v.x = nodeX(gx)
      v.y = nodeY(gy)
      v.speed = 0
      v.goalGx = Math.floor(this.rng() * GRID_N)
      v.goalGy = Math.floor(this.rng() * GRID_N)
      v.speedLimit = 0
      chooseNext(v, this.rng)
      this.beginRoute(v)
    }
    this.episodes += 1
  }

  /** 障害物を置き、置いたあとの個数を返す */
  addObstacle(x: number, y: number, radius: number): number {
    this.obstacles.push({ id: this.obstacleSeq++, x, y, radius })
    return this.obstacles.length
  }

  removeObstacle(id: number): void {
    this.obstacles = this.obstacles.filter((o) => o.id !== id)
  }

  clearObstacles(): void {
    this.obstacles = []
  }

  /** 徒歩キャラの位置。null なら街から消す */
  setPlayer(at: Vec2 | null): void {
    this.player = at
    this.playerAt = at ? Date.now() : 0
  }

  step(s: TrafficStep): void {
    // 届かなくなった徒歩キャラは街から消す（見えない人の前で止まり続ける）
    if (this.player && Date.now() - this.playerAt >= PLAYER_POSE_TTL_MS) {
      this.player = null
      this.playerAt = 0
    }

    for (const v of this.vehicles) this.stepVehicle(v, s)
    this.applyQueueing()
    for (const p of this.pedestrians) this.stepPedestrian(p, s.dt)
    if (s.tick % Math.round(SIM_HZ * 2) === 0) this.recyclePedestrians()
    this.detectCollisions()
  }

  /** frame に載せる車両。経路は変わった車だけ載せる */
  vehicleStates(): VehicleState[] {
    return this.vehicles.map((v) => {
      const state: VehicleState = {
        id: v.id,
        active: v.active,
        x: v.x,
        y: v.y,
        heading: v.heading,
        speed: v.active ? v.speed : 0,
        steer: v.steer,
        collided: v.collided,
        reachedGoal: v.reachedGoal,
        goal: [nodeX(v.goalGx), nodeY(v.goalGy)],
        progress: v.progress,
        signalViolations: 0,
        laneDepartures: 0,
        speedLimit: v.speedLimit,
        speedViolations: v.speedViolations,
        braking: v.braking,
        throttle: v.throttle,
        turnSignal: v.turnSignal,
      }
      if (v.active && v.routeDirty) {
        state.route = buildRoute(v)
        v.routeDirty = false
      }
      return state
    })
  }

  pedestrianStates(): NpcPedestrianState[] {
    return this.pedestrians.map((p) => this.pedestrianAt(p))
  }

  /** (x, y) から `radius` 以内に歩行者がいるか */
  pedestrianNear(x: number, y: number, radius: number): boolean {
    return this.pedestrians.some((p) => {
      const state = this.pedestrianAt(p)
      return Math.hypot(state.x - x, state.y - y) < radius
    })
  }

  private makeVehicle(id: number): MockVehicle {
    const gx = Math.floor(this.rng() * GRID_N)
    const gy = Math.floor(this.rng() * GRID_N)
    const v: MockVehicle = {
      id,
      active: false,
      gx,
      gy,
      tx: gx,
      ty: gy,
      t: 0,
      x: nodeX(gx),
      y: nodeY(gy),
      heading: 0,
      speed: 0,
      steer: 0,
      throttle: 0,
      collided: false,
      reachedGoal: false,
      speedLimit: 0,
      speedViolations: 0,
      overspeeding: false,
      goalGx: Math.floor(this.rng() * GRID_N),
      goalGy: Math.floor(this.rng() * GRID_N),
      routeTotal: 1,
      progress: 0,
      routeDirty: true,
      braking: false,
      turnSignal: 0,
    }
    chooseNext(v, this.rng)
    this.beginRoute(v)
    return v
  }

  /** 新しい目的地を決めた直後に呼ぶ。progress の分母を取り直す */
  private beginRoute(v: MockVehicle): void {
    v.routeTotal = Math.max(1, remainingDistance(v))
    v.progress = 0
    v.routeDirty = true
    v.speedViolations = 0
    v.overspeeding = false
  }

  private makePedestrian(id: number): MockPedestrian {
    const gx = Math.floor(this.rng() * GRID_N)
    const gy = Math.floor(this.rng() * GRID_N)
    const along = this.rng() < 0.5
    return {
      id,
      gx,
      gy,
      tx: along ? Math.min(GRID_N - 1, gx + 1) : gx,
      ty: along ? gy : Math.min(GRID_N - 1, gy + 1),
      t: this.rng(),
      side: this.rng() < 0.5 ? 1 : -1,
      cross: 0,
      crossing: false,
      stride: this.rng() * Math.PI * 2,
      speed: 1.1 + this.rng() * 0.5,
    }
  }

  private stepPedestrian(p: MockPedestrian, dt: number): void {
    const ax = nodeX(p.gx)
    const ay = nodeY(p.gy)
    const bx = nodeX(p.tx)
    const by = nodeY(p.ty)
    const length = Math.max(1e-3, Math.hypot(bx - ax, by - ay))
    p.stride += p.speed * dt * 2

    if (p.crossing) {
      p.cross += (p.speed / (WALK_OFFSET_M * 2)) * dt
      if (p.cross >= 1) {
        p.cross = 0
        p.crossing = false
        p.side = -p.side
      }
      return
    }

    p.t += (p.speed * dt) / length
    if (p.t < 1) return

    p.t = 0
    p.gx = p.tx
    p.gy = p.ty
    if (this.rng() < 0.3) {
      p.crossing = true
      p.cross = 0
    }
    // 次の区間を選ぶ（格子なので上下左右のいずれか）
    const options: Array<[number, number]> = []
    if (p.gx > 0) options.push([p.gx - 1, p.gy])
    if (p.gx < GRID_N - 1) options.push([p.gx + 1, p.gy])
    if (p.gy > 0) options.push([p.gx, p.gy - 1])
    if (p.gy < GRID_N - 1) options.push([p.gx, p.gy + 1])
    const [nx, ny] = options[Math.floor(this.rng() * options.length)]
    p.tx = nx
    p.ty = ny
  }

  /** どの車からも遠い歩行者を車の近くへ回す（実サーバーの `PedestrianCrowd.recycle`）。 */
  private recyclePedestrians(): void {
    const cars = this.vehicles.filter((v) => v.active)
    if (!cars.length) return
    for (const p of this.pedestrians) {
      const at = this.pedestrianAt(p)
      let nearest = Infinity
      for (const v of cars) nearest = Math.min(nearest, Math.hypot(v.x - at.x, v.y - at.y))
      if (nearest <= PEDESTRIAN_RECYCLE_M) continue
      const v = cars[Math.floor(this.rng() * cars.length)]
      const home = snapToGrid(v.x, v.y)
      p.gx = home.gx
      p.gy = home.gy
      p.tx = clampGrid(p.gx + (this.rng() < 0.5 ? 1 : -1))
      p.ty = p.gy
      if (p.tx === p.gx) {
        p.tx = p.gx
        p.ty = clampGrid(p.gy + 1)
      }
      p.t = this.rng()
      p.crossing = false
      p.cross = 0
    }
  }

  private pedestrianAt(p: MockPedestrian): NpcPedestrianState {
    const ax = nodeX(p.gx)
    const ay = nodeY(p.gy)
    const bx = nodeX(p.tx)
    const by = nodeY(p.ty)
    const length = Math.max(1e-3, Math.hypot(bx - ax, by - ay))
    const dx = (bx - ax) / length
    const dy = (by - ay) / length
    const side = p.crossing ? p.side * (1 - 2 * p.cross) : p.side
    const offset = side * WALK_OFFSET_M
    return {
      id: p.id,
      x: ax + dx * (p.t * length) - dy * offset,
      y: ay + dy * (p.t * length) + dx * offset,
      heading: p.crossing ? Math.atan2(dx * -p.side, dy * p.side) : Math.atan2(dy, dx),
      stride: p.stride % (Math.PI * 2),
      crossing: p.crossing,
    }
  }

  /** 次の交差点で曲がるなら方向指示器を出す。 */
  private turnSignalFor(v: MockVehicle): number {
    const rest = Math.hypot(nodeX(v.tx) - v.x, nodeY(v.ty) - v.y)
    if (rest > TURN_LOOKAHEAD_M) return 0
    const dirX = Math.sign(v.tx - v.gx)
    const dirY = Math.sign(v.ty - v.gy)
    const turnY = dirX !== 0 ? Math.sign(v.goalGy - v.ty) : 0
    const turnX = dirY !== 0 ? Math.sign(v.goalGx - v.tx) : 0
    // ENU は反時計回りが正なので、外積が正なら左へ曲がる
    const cross = dirX * turnY - dirY * turnX
    return cross > 0 ? -1 : cross < 0 ? 1 : 0
  }

  private stepVehicle(v: MockVehicle, s: TrafficStep): void {
    if (!v.active) return
    const { dt, params } = s

    const ax = nodeX(v.gx)
    const ay = nodeY(v.gy)
    const bx = nodeX(v.tx)
    const by = nodeY(v.ty)
    const segLen = Math.hypot(bx - ax, by - ay)

    if (segLen < 1e-6) {
      v.goalGx = Math.floor(this.rng() * GRID_N)
      v.goalGy = Math.floor(this.rng() * GRID_N)
      chooseNext(v, this.rng)
      this.beginRoute(v)
      v.reachedGoal = true
      this.episodes += 1
      return
    }

    let targetSpeed = params.maxSpeed * (0.45 + 0.55 * Math.sin(Math.PI * v.t))

    const hop = hopKey(v.gx, v.gy, v.tx, v.ty)
    const limit = s.signals.limitOn(hop)
    if (limit !== undefined) v.speedLimit = limit
    if (params.obeySpeedSigns && v.speedLimit > 0) {
      targetSpeed = Math.min(targetSpeed, v.speedLimit)
    }

    let stopT = 1
    if (params.obeySignals) {
      const idx = s.signals.signalOn(hop)
      const phase = idx === undefined ? SIGNAL_GREEN : s.phases[idx]
      if (idx !== undefined && phase !== SIGNAL_GREEN) {
        const setback = s.signals.setbackOf(idx)
        const stopDist = (1 - v.t) * segLen - setback
        const canStop = (v.speed * v.speed) / (2 * BRAKE_ACCEL) <= Math.max(0, stopDist)
        if (phase === SIGNAL_RED || canStop) {
          stopT = Math.max(0, 1 - setback / segLen)
          targetSpeed = Math.min(targetSpeed, Math.sqrt(2 * BRAKE_ACCEL * Math.max(0, stopDist)))
        }
      }
    }

    const playerGap = this.playerGap(v)
    if (playerGap < Infinity) {
      const room = Math.max(0, playerGap - PLAYER_MARGIN_M)
      targetSpeed = Math.min(targetSpeed, Math.sqrt(2 * BRAKE_ACCEL * room))
    }

    // 実機と同じく「指令が減速側か」で決める（実測の加速度では見ない）。
    const demand = targetSpeed - v.speed
    v.throttle = Math.max(-1, Math.min(1, demand / 3))
    v.braking = targetSpeed < v.speed - 0.2
    v.turnSignal = this.turnSignalFor(v)

    v.speed += (targetSpeed - v.speed) * Math.min(1, dt * 1.6)

    const over = v.speedLimit > 0 && v.speed > v.speedLimit
    if (over && !v.overspeeding) v.speedViolations += 1
    v.overspeeding = over

    const advance = (v.speed * dt) / segLen
    v.t += advance
    if (stopT < 1 && v.t > stopT) {
      v.t = stopT
      v.speed = 0
    }

    const prevHeading = v.heading
    v.heading = Math.atan2(by - ay, bx - ax)
    const dh = Math.atan2(Math.sin(v.heading - prevHeading), Math.cos(v.heading - prevHeading))
    v.steer += (Math.max(-0.5, Math.min(0.5, dh * 4)) - v.steer) * 0.25

    if (v.t >= 1) {
      v.t = 0
      v.gx = v.tx
      v.gy = v.ty
      v.x = nodeX(v.gx)
      v.y = nodeY(v.gy)
      if (v.gx === v.goalGx && v.gy === v.goalGy) {
        // 徴用中の 1 台は乗降地点で止める（実機と同じく respawn しない）
        if (v.id === s.heldVehicle) {
          v.speed = 0
          v.reachedGoal = true
          return
        }
        v.reachedGoal = true
        this.episodes += 1
        v.goalGx = Math.floor(this.rng() * GRID_N)
        v.goalGy = Math.floor(this.rng() * GRID_N)
        chooseNext(v, this.rng)
        this.beginRoute(v)
      } else {
        v.reachedGoal = false
        chooseNext(v, this.rng)
        v.routeDirty = true
      }
      return
    }

    v.x = ax + (bx - ax) * v.t
    v.y = ay + (by - ay) * v.t
    v.reachedGoal = false
  }

  /** 前方の進路上にいる徒歩キャラまでの距離 [m]。いなければ Infinity。 */
  private playerGap(v: MockVehicle): number {
    const at = this.player
    if (!at) return Infinity
    const cos = Math.cos(v.heading)
    const sin = Math.sin(v.heading)
    const dx = at[0] - v.x
    const dy = at[1] - v.y
    const lon = dx * cos + dy * sin
    const lat = -dx * sin + dy * cos
    if (lon <= 0 || lon > PLAYER_RANGE_M) return Infinity
    if (Math.abs(lat) > PLAYER_HALF_WIDTH_M) return Infinity
    return lon
  }

  /** 同じ区間を走る前走車に追突しないよう、後続の位置を後ろへ詰める。 */
  private applyQueueing(): void {
    const lanes = new Map<string, MockVehicle[]>()
    for (const v of this.vehicles) {
      if (!v.active) continue
      const key = hopKey(v.gx, v.gy, v.tx, v.ty)
      const list = lanes.get(key)
      if (list) list.push(v)
      else lanes.set(key, [v])
    }

    for (const list of lanes.values()) {
      if (list.length < 2) continue
      const head = list[0]
      const ax = nodeX(head.gx)
      const ay = nodeY(head.gy)
      const bx = nodeX(head.tx)
      const by = nodeY(head.ty)
      const segLen = Math.hypot(bx - ax, by - ay)
      if (segLen < 1e-6) continue

      const gap = CAR_GAP_M / segLen
      list.sort((a, b) => b.t - a.t)
      for (let i = 1; i < list.length; i++) {
        const ahead = list[i - 1]
        const v = list[i]
        const limit = ahead.t - gap
        if (v.t > limit) {
          v.t = Math.max(0, limit)
          v.speed = Math.min(v.speed, ahead.speed)
          v.x = ax + (bx - ax) * v.t
          v.y = ay + (by - ay) * v.t
        }
      }
    }
  }

  private detectCollisions(): void {
    for (const v of this.vehicles) v.collided = false
    const active = this.vehicles.filter((v) => v.active)
    for (let i = 0; i < active.length; i++) {
      for (let j = i + 1; j < active.length; j++) {
        const a = active[i]
        const b = active[j]
        if (Math.hypot(a.x - b.x, a.y - b.y) < 4.5) {
          a.collided = true
          b.collided = true
        }
      }
      for (const o of this.obstacles) {
        const a = active[i]
        if (Math.hypot(a.x - o.x, a.y - o.y) < o.radius + 2.2) a.collided = true
      }
    }
  }
}
