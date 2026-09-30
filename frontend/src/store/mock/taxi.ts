/** モックの配車（迎車 → 乗車 → 到着 → 降車）。**経路はグリッドに沿った折れ線の擬似的な再現** */

import type { StatusPayload, TaxiMessage, Vec2 } from '../../types/protocol.ts'
import { GRID_N, snapToGrid } from './grid.ts'
import type { Rng } from './grid.ts'
import { buildRoute, chooseNext, remainingDistance } from './traffic.ts'
import type { MockVehicle } from './traffic.ts'

const MOCK_IDLE_TAXI: TaxiMessage = {
  type: 'taxi',
  phase: 'idle',
  vehicleId: -1,
  pickup: null,
  dropoff: null,
  routeRevision: 0,
  route: [],
  etaSeconds: 0,
  remainingDistanceM: 0,
  message: '',
  driveMode: 'normal',
}

/** 乗降地点に「着いた」とみなす距離 [m] と、配車を受け付ける最短距離 [m] */
const MOCK_TAXI_ARRIVE_M = 6
const MOCK_TAXI_MIN_TRIP_M = 50
/** ETA を出すときの速度の下限 [m/s] */
const MOCK_TAXI_MIN_SPEED = 3.5

/** 配車が外へ知らせる手段 */
export interface TaxiLink {
  sendTaxi(msg: TaxiMessage): void
  sendStatus(patch: Partial<StatusPayload>): void
}

export class MockTaxi {
  private state: TaxiMessage = { ...MOCK_IDLE_TAXI }
  private sentAt = 0
  private readonly vehicles: MockVehicle[]
  private readonly rng: Rng
  private readonly link: TaxiLink

  constructor(vehicles: MockVehicle[], rng: Rng, link: TaxiLink) {
    this.vehicles = vehicles
    this.rng = rng
    this.link = link
  }

  /** 車内にいるか。乗車中の徒歩キャラは歩行者として扱わない（実機と同じ） */
  get onboard(): boolean {
    return this.state.phase === 'riding' || this.state.phase === 'arrived'
  }

  /** 徴用している車（乗降地点で止める）。配車していなければ -1 */
  get heldVehicle(): number {
    return this.state.phase === 'idle' ? -1 : this.state.vehicleId
  }

  /** taxi メッセージを送る。`force` でなければ 1Hz に間引く */
  send(force = false): void {
    const now = performance.now()
    if (!force && now - this.sentAt < 1000) return
    this.sentAt = now
    this.link.sendTaxi({ ...this.state, route: [...(this.state.route ?? [])] })
  }

  /** 配車の段階を進める。**実機と同じく、到着は「残り距離」で判定する。** */
  step(): void {
    if (this.state.phase === 'idle') return
    const v = this.vehicles[this.state.vehicleId]
    if (!v || !v.active) {
      this.cancel('（モック）配車していた車両がいなくなりました')
      return
    }

    const target = this.state.phase === 'riding' ? this.state.dropoff : this.state.pickup
    if (!target) return
    const remaining = remainingDistance(v)
    this.state.remainingDistanceM = remaining
    this.state.etaSeconds =
      this.state.phase === 'waiting' || this.state.phase === 'arrived'
        ? 0
        : remaining / Math.max(MOCK_TAXI_MIN_SPEED, v.speed)

    const near = Math.hypot(v.x - target[0], v.y - target[1]) <= MOCK_TAXI_ARRIVE_M
    const before = this.state.phase
    if (this.state.phase === 'approaching' && near) {
      v.speed = 0
      this.state.phase = 'waiting'
      this.state.message = '（モック）乗車地点に到着しました。[Enter] で乗車できます'
    } else if (this.state.phase === 'riding' && near) {
      v.speed = 0
      this.state.phase = 'arrived'
      this.state.message = '（モック）目的地に到着しました。[Enter] で降車できます'
    }
    if (this.state.phase === 'waiting' || this.state.phase === 'arrived') v.speed = 0
    this.send(before !== this.state.phase)
  }

  /** 配車を畳む。徴用していた車は街の走行へ戻す */
  cancel(message: string): void {
    const v = this.vehicles[this.state.vehicleId]
    if (v) {
      v.goalGx = Math.floor(this.rng() * GRID_N)
      v.goalGy = Math.floor(this.rng() * GRID_N)
      chooseNext(v, this.rng)
      v.routeDirty = true
    }
    this.state = {
      ...MOCK_IDLE_TAXI,
      routeRevision: this.state.routeRevision + 1,
      message,
    }
    this.link.sendStatus({ taxiVehicleId: -1 })
    this.send(true)
  }

  /** いちばん近い車を迎えに向かわせる。断るときは status の知らせで理由を返す */
  request(pickup: Vec2, dropoff: Vec2): void {
    if (this.state.phase !== 'idle') {
      this.link.sendStatus({ message: '（モック）すでに配車中です' })
      return
    }
    const pick = snapToGrid(pickup[0], pickup[1])
    const drop = snapToGrid(dropoff[0], dropoff[1])
    if (
      Math.hypot(drop.point[0] - pick.point[0], drop.point[1] - pick.point[1]) <
      MOCK_TAXI_MIN_TRIP_M
    ) {
      this.link.sendStatus({ message: '（モック）乗車地点と降車地点が近すぎます' })
      return
    }
    let slot = -1
    let best = Infinity
    for (const v of this.vehicles) {
      if (!v.active) continue
      const d = Math.hypot(v.x - pick.point[0], v.y - pick.point[1])
      if (d < best) {
        best = d
        slot = v.id
      }
    }
    if (slot < 0) {
      this.link.sendStatus({ message: '（モック）配車できる車両がいません' })
      return
    }
    const v = this.vehicles[slot]
    v.goalGx = pick.gx
    v.goalGy = pick.gy
    chooseNext(v, this.rng)
    v.routeDirty = true
    this.state = {
      ...MOCK_IDLE_TAXI,
      phase: 'approaching',
      vehicleId: slot,
      pickup: pick.point,
      dropoff: drop.point,
      route: buildRoute(v),
      routeRevision: this.state.routeRevision + 1,
      remainingDistanceM: remainingDistance(v),
      // 最初の 1 通から出しておく（次の step まで「まもなく」と出てしまう）
      etaSeconds: remainingDistance(v) / MOCK_TAXI_MIN_SPEED,
      message: `（モック）車両 #${slot} が迎えに向かっています`,
    }
    this.link.sendStatus({ taxiVehicleId: slot })
    this.send(true)
  }

  /** 乗車。迎車の途中（approaching）でも受け付ける（実機の `taxi.board` と同じ） */
  board(): void {
    if (this.state.phase !== 'waiting' && this.state.phase !== 'approaching') return
    const v = this.vehicles[this.state.vehicleId]
    const drop = this.state.dropoff
    if (!v || !drop) return
    const snapped = snapToGrid(drop[0], drop[1])
    v.goalGx = snapped.gx
    v.goalGy = snapped.gy
    chooseNext(v, this.rng)
    v.routeDirty = true
    this.state = {
      ...this.state,
      phase: 'riding',
      route: buildRoute(v),
      routeRevision: this.state.routeRevision + 1,
      message: '（モック）目的地へ向かっています',
    }
    this.send(true)
  }

  alight(): void {
    if (!this.onboard) return
    this.cancel('（モック）降車しました')
  }

  /** 利用者が取り消す。`halt` なら緊急停止（その場で止める） */
  abort(halt: boolean): void {
    if (this.state.phase === 'idle') return
    const v = this.vehicles[this.state.vehicleId]
    if (halt && v) v.speed = 0
    this.cancel(
      halt ? '（モック）緊急停止しました。自動運転を終了します' : '（モック）配車を取り消しました',
    )
  }
}
