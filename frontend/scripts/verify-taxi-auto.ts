/** 配車の自動操作が、段階を飛ばさず 1 周することを検証する（ブラウザ不要）。 */

import {
  ARRIVED_WAIT_SEC,
  IDLE_WAIT_SEC,
  RETRY_SEC,
  TRIP_MAX_M,
  TRIP_MIN_M,
  decideAutoAction,
  pickDropoff,
  resetTaxiAutopilot,
  taxiAutopilot,
} from '../src/store/taxiAutopilot.ts'
import type { AutoAction, AutoInput } from '../src/store/taxiAutopilot.ts'
import type { MapNode, TaxiPhase } from '../src/types/protocol.ts'

let failures = 0

function check(label: string, ok: boolean, detail = ''): void {
  console.log(`  [${ok ? 'OK  ' : 'NG  '}] ${label}${detail ? ` — ${detail}` : ''}`)
  if (!ok) failures++
}

function input(patch: Partial<AutoInput> = {}): AutoInput {
  return {
    phase: 'idle',
    vehicleId: 3,
    aimed: -1,
    placed: true,
    mapReady: true,
    now: 100,
    lastSentAt: 0,
    ...patch,
  }
}

console.log('='.repeat(70))
console.log('前提（街に立っていない / エリアが無いときは動かない）')
console.log('='.repeat(70))

{
  check('街に立っていなければ何もしない', decideAutoAction(input({ placed: false })) === 'none')
  check('エリアが無ければ何もしない', decideAutoAction(input({ mapReady: false })) === 'none')
  // ★ 呼べない状態で呼び続けると、断られる message でスマホ画面が埋まる
  check(
    '立っていなければ段階を問わず何もしない',
    (['idle', 'approaching', 'waiting', 'riding', 'arrived'] as TaxiPhase[]).every(
      (phase) => decideAutoAction(input({ phase, placed: false, aimed: 3 })) === 'none',
    ),
  )
}

console.log()
console.log('='.repeat(70))
console.log('段階ごとの判断')
console.log('='.repeat(70))

{
  check(
    'idle は待ってから呼ぶ',
    decideAutoAction(input({ now: 100, lastSentAt: 100 })) === 'none' &&
      decideAutoAction(input({ now: 100 + IDLE_WAIT_SEC, lastSentAt: 100 })) === 'request',
    IDLE_WAIT_SEC + ' 秒',
  )

  // ★ 迎車の途中では乗らない（到着＝ waiting を待つ）
  check(
    '迎車中（approaching）は照準に入っていても乗らない',
    decideAutoAction(input({ phase: 'approaching', aimed: 3, lastSentAt: 0 })) === 'none',
  )

  check(
    'waiting で照準に入っていなければ歩いて近づく',
    decideAutoAction(input({ phase: 'waiting', aimed: -1 })) === 'approach',
  )
  check(
    'waiting で照準に入ったら乗る',
    decideAutoAction(input({ phase: 'waiting', aimed: 3, lastSentAt: 0 })) === 'board',
  )
  // ★ 別の車を狙っているときに乗ろうとしない（断られ続ける）
  check(
    '違う車に照準が合っていても乗らない',
    decideAutoAction(input({ phase: 'waiting', aimed: 5, lastSentAt: 0 })) === 'approach',
  )

  check('riding は何もしない', decideAutoAction(input({ phase: 'riding' })) === 'none')

  check(
    'arrived は待ってから降りる',
    decideAutoAction(input({ phase: 'arrived', now: 100, lastSentAt: 100 })) === 'none' &&
      decideAutoAction(
        input({ phase: 'arrived', now: 100 + ARRIVED_WAIT_SEC, lastSentAt: 100 }),
      ) === 'alight',
    ARRIVED_WAIT_SEC + ' 秒',
  )
}

console.log()
console.log('='.repeat(70))
console.log('送り直しの間隔（断られても詰まらない）')
console.log('='.repeat(70))

/** 断られ続ける段階に 30 秒置いたとき、指令を何回送るか */
function resendCount(phase: TaxiPhase, aimed: number): number {
  let last = 0
  let count = 0
  for (let i = 0; i < 60 * 30; i++) {
    const now = i / 60
    if (decideAutoAction(input({ phase, aimed, now, lastSentAt: last })) !== 'none') {
      last = now
      count++
    }
  }
  return count
}

{
  // ★ 断られた指令では段階が変わらないので、間隔を空けないと 60fps で送り続ける
  //   （スマホ画面が「いまは乗車できません」で埋まる）
  const calls = resendCount('idle', -1)
  check(
    '呼び直しは 30 秒で間隔ぶんだけ',
    calls <= Math.ceil(30 / IDLE_WAIT_SEC) + 1,
    calls + ' 回（上限 ' + (Math.ceil(30 / IDLE_WAIT_SEC) + 1) + ' 回）',
  )

  const boards = resendCount('waiting', 3)
  check(
    '乗り直しは 30 秒で間隔ぶんだけ',
    boards <= Math.ceil(30 / RETRY_SEC) + 1,
    boards + ' 回（上限 ' + (Math.ceil(30 / RETRY_SEC) + 1) + ' 回）',
  )

  const alights = resendCount('arrived', -1)
  check(
    '降り直しは 30 秒で間隔ぶんだけ',
    alights <= Math.ceil(30 / ARRIVED_WAIT_SEC) + 1,
    alights + ' 回（上限 ' + (Math.ceil(30 / ARRIVED_WAIT_SEC) + 1) + ' 回）',
  )
}

console.log()
console.log('='.repeat(70))
console.log('1 周（呼ぶ → 迎車 → 乗車 → 走行 → 到着 → 降車）')
console.log('='.repeat(70))

{
  // サーバーの段階遷移を最小限に真似て、実際に 1 周するかを見る
  const order: TaxiPhase[] = ['idle', 'approaching', 'waiting', 'riding', 'arrived']
  const log: AutoAction[] = []
  let phase: TaxiPhase = 'idle'
  let lastPhase: TaxiPhase = 'idle'
  let lastSentAt = 0
  let aimed = -1
  let steps = 0
  let laps = 0

  for (let i = 0; i < 60 * 300 && laps < 2; i++) {
    const now = i / 60
    if (lastPhase !== phase) {
      lastPhase = phase
      lastSentAt = now
    }
    const action = decideAutoAction(input({ phase, aimed, now, lastSentAt }))
    if (action !== 'none') log.push(action)

    switch (action) {
      case 'request':
        lastSentAt = now
        phase = 'approaching'
        break
      case 'approach':
        // 歩いて近づくと、いずれ照準に入る
        steps++
        if (steps > 30) aimed = 3
        break
      case 'board':
        lastSentAt = now
        phase = 'riding'
        aimed = -1
        steps = 0
        break
      case 'alight':
        lastSentAt = now
        phase = 'idle'
        laps++
        break
      default:
        break
    }
    // 迎車と走行はサーバー側で進む。ここでは一定時間で次の段階へ送る
    if (phase === 'approaching' && now - lastSentAt > 4) phase = 'waiting'
    if (phase === 'riding' && now - lastSentAt > 4) phase = 'arrived'
    if (!order.includes(phase)) break
  }

  check('2 周とも最後まで回る', laps === 2, laps + ' 周')

  const commands = log.filter((a) => a !== 'approach')
  const cycle = ['request', 'board', 'alight']
  check(
    '指令の順番が 呼ぶ → 乗る → 降りる の繰り返し',
    commands.every((a, i) => a === cycle[i % cycle.length]),
    commands.join(' → '),
  )
  check('1 周につき 3 つだけ送る', commands.length === 6, commands.length + ' 件')
}

console.log()
console.log('='.repeat(70))
console.log('行き先の選び方')
console.log('='.repeat(70))

{
  const nodes: MapNode[] = []
  for (let i = 0; i < 400; i++) {
    nodes.push({ id: i, x: (i % 20) * 120, y: Math.floor(i / 20) * 120 })
  }

  let seed = 12345
  const rng = () => {
    seed = (seed * 1103515245 + 12345) % 2147483648
    return seed / 2147483648
  }

  let worst = 0
  let best = Infinity
  for (let i = 0; i < 200; i++) {
    const at = nodes[Math.floor(rng() * nodes.length)]
    const pick = pickDropoff(nodes, at.x, at.y, rng)
    if (!pick) {
      check('行き先が選べる', false)
      break
    }
    const d = Math.hypot(pick[0] - at.x, pick[1] - at.y)
    worst = Math.max(worst, d)
    best = Math.min(best, d)
  }
  check(
    '行き先は近すぎない',
    best >= TRIP_MIN_M,
    '最短 ' + best.toFixed(0) + 'm（下限 ' + TRIP_MIN_M + 'm）',
  )
  // ★ 金沢は 12.3km 四方あるので、上限を外すと経路が作れない点ばかり引く
  check(
    '行き先は遠すぎない',
    worst <= TRIP_MAX_M,
    '最長 ' + worst.toFixed(0) + 'm（上限 ' + TRIP_MAX_M + 'm）',
  )

  check('ノードが無ければ選ばない', pickDropoff([], 0, 0, rng) === null)
  // 帯に入る候補が 1 つも無くても、いちばん近い候補を返して止まらないこと
  const lonely: MapNode[] = [{ id: 0, x: 0, y: 0 }, { id: 1, x: 5, y: 0 }]
  check('帯に入る候補が無くても返す', pickDropoff(lonely, 0, 0, rng) !== null)
}

console.log()
console.log('='.repeat(70))
console.log('入り直したとき')
console.log('='.repeat(70))

{
  // 代入したリテラルへ絞られないよう、宣言どおりの型で入れてから呼ぶ
  const dirty: { phase: TaxiPhase; walking: boolean } = { phase: 'arrived', walking: true }
  taxiAutopilot.lastSentAt = 0
  taxiAutopilot.lastPhase = dirty.phase
  taxiAutopilot.walking = dirty.walking
  resetTaxiAutopilot(500)
  check('時計を入れ直す', taxiAutopilot.lastSentAt === 500)
  check('段階の記憶を戻す', taxiAutopilot.lastPhase === 'idle')
  // ★ 足を止めないと、切ったあとも前進キーが押しっぱなしになる
  check('足を止める', taxiAutopilot.walking === false)
  check(
    '入った直後はすぐ呼ばない',
    decideAutoAction(input({ now: 500, lastSentAt: taxiAutopilot.lastSentAt })) === 'none',
  )
}

console.log()
console.log('='.repeat(70))
if (failures > 0) {
  console.log(`結果: ${failures} 件の不合格`)
  process.exit(1)
}
console.log('結果: すべて合格')
