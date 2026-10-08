/** 車車間通信（V2X）のリンクの線分の頂点・近傍の選び方・表示の文言と、契約ファイル・バックエンドの定数との突き合わせを検証する（ブラウザ不要）。 */

import { readFileSync } from 'node:fs'

import {
  V2X_LINK_HEIGHT_M,
  V2X_MAX_PEERS,
  V2X_RANGE_M,
  linkPairs,
  nearestPeers,
  v2xStatusText,
  writeLinkSegments,
} from '../src/scene/v2xLinkGeometry.ts'
import type { LinkVehicle } from '../src/scene/v2xLinkGeometry.ts'

let failures = 0

function check(label: string, ok: boolean, detail = ''): void {
  console.log(`  [${ok ? 'OK  ' : 'NG  '}] ${label}${detail ? ` — ${detail}` : ''}`)
  if (!ok) failures++
}

function car(id: number, x: number, y: number, links?: number[], active = true): LinkVehicle {
  return { id, active, x, y, v2xConnectedIds: links }
}

console.log('1. 線で結ぶ組')
{
  const pairs = linkPairs([car(0, 0, 0, [1]), car(1, 10, 0, [0])])
  check('互いに受け取っていても線は 1 本（向きを区別しない）', pairs.length === 1 && pairs[0][0] === 0 && pairs[0][1] === 1, JSON.stringify(pairs))
  const oneWay = linkPairs([car(0, 0, 0, [2]), car(2, 20, 0)])
  check('片方だけが受け取っていても線を引く', oneWay.length === 1 && oneWay[0][0] === 0 && oneWay[0][1] === 2)
  const ghost = linkPairs([car(0, 0, 0, [1, 7]), car(1, 5, 0, [], false)])
  check('走っていない車・見当たらない車への線は引かない', ghost.length === 0, JSON.stringify(ghost))
  const self = linkPairs([car(3, 0, 0, [3])])
  check('自分自身への線は引かない', self.length === 0)
  const none = linkPairs([car(0, 0, 0), car(1, 1, 1)])
  check('誰も受け取っていなければ線は 0 本', none.length === 0)
  const tri = linkPairs([car(0, 0, 0, [1, 2]), car(1, 5, 0, [0, 2]), car(2, 10, 0, [1, 0])])
  check('3 台が互いに受け取れば 3 本（重複なし）', tri.length === 3 && new Set(tri.map((p) => p.join(':'))).size === 3)
}

console.log('\n2. 頂点のバッファ')
{
  const pos = new Map([[0, { x: 1, y: 2 }], [1, { x: 11, y: -3 }]])
  const out = new Float32Array(12)
  const n = writeLinkSegments([[0, 1]], (id) => pos.get(id) ?? null, out)
  check('1 本につき 2 頂点', n === 2)
  const expected = [1, V2X_LINK_HEIGHT_M, -2, 11, V2X_LINK_HEIGHT_M, 3]
  const got = Array.from(out.slice(0, 6))
  check(
    'ENU (x, y) は three の (x, 高さ, -y)（protocol.md 1.3）',
    got.every((v, i) => Math.abs(v - expected[i]) < 1e-6),
    JSON.stringify(got),
  )
  const skip = writeLinkSegments([[0, 5], [0, 1]], (id) => pos.get(id) ?? null, out)
  check('位置が分からない組は飛ばし、次の組を詰めて書く', skip === 2 && Math.abs(out[3] - 11) < 1e-6)
  const tiny = new Float32Array(6)
  const capped = writeLinkSegments([[0, 1], [0, 1]], (id) => pos.get(id) ?? null, tiny)
  check('バッファからあふれる分は書かない', capped === 2)
  check('頂点に NaN を書かない', Array.from(out).every((v) => Number.isFinite(v)))
}

console.log('\n3. 近傍の選び方（モック。バックエンドの route_and_aggregate と同じ規則）')
{
  const two = nearestPeers([car(0, 0, 0), car(1, 15, 0)])
  check('15m 離れた 2 台は互いに受け取る', two.get(0)?.[0] === 1 && two.get(1)?.[0] === 0)
  const far = nearestPeers([car(0, 0, 0), car(1, 50, 0)])
  check('50m 離れていれば届かない', far.size === 0)
  const edge = nearestPeers([car(0, 0, 0), car(1, V2X_RANGE_M, 0)])
  check(`ちょうど ${V2X_RANGE_M}m は届く`, edge.get(0)?.length === 1)
  const crowd = nearestPeers([car(0, 0, 0), car(1, 5, 0), car(2, 12, 0), car(3, 20, 0)])
  check(`近い順に最大 ${V2X_MAX_PEERS} 台`, JSON.stringify(crowd.get(0)) === JSON.stringify([1, 2]), JSON.stringify(crowd.get(0)))
  const idle = nearestPeers([car(0, 0, 0), car(1, 5, 0, undefined, false)])
  check('走っていない車とはつながらない', idle.size === 0)
}

console.log('\n4. 表示の文言')
check('1 台', v2xStatusText([2]) === 'V2X: 車両#2とリンク中')
check('2 台', v2xStatusText([2, 5]) === 'V2X: 車両#2・#5とリンク中')
check('相手がいなければ null（チップを出さない）', v2xStatusText([]) === null && v2xStatusText(undefined) === null)

console.log('\n5. 契約ファイルとバックエンドの定数')
const protocolTs = readFileSync('src/types/protocol.ts', 'utf8')
const protocolMd = readFileSync('../docs/protocol.md', 'utf8')
const configPy = readFileSync('../backend/app/config.py', 'utf8')
const contractsPy = readFileSync('../backend/app/contracts.py', 'utf8')
const simStore = readFileSync('src/store/simStore.ts', 'utf8')
const mock = readFileSync('src/store/mockServer.ts', 'utf8')

function interfaceBody(name: string): string {
  return new RegExp(`export interface ${name} \\{([\\s\\S]*?)\\n\\}`).exec(protocolTs)?.[1] ?? ''
}
check('VehicleState に v2xConnectedIds?: number[] がある', /\n  v2xConnectedIds\?: number\[\]/.test(interfaceBody('VehicleState')))
check('SimParams に v2xComm: boolean がある', /\n  v2xComm: boolean/.test(interfaceBody('SimParams')))
check('docs/protocol.md の例に v2xConnectedIds と v2xComm がある', /"v2xConnectedIds":/.test(protocolMd) && /"v2xComm":/.test(protocolMd))
const pyNumber = (name: string) => Number(new RegExp(`^${name} = ([\\d.]+)`, 'm').exec(configPy)?.[1])
check('V2X_RANGE_M がバックエンドと一致する', pyNumber('V2X_RANGE_M') === V2X_RANGE_M, `backend ${pyNumber('V2X_RANGE_M')}`)
check('V2X_MAX_PEERS がバックエンドと一致する', pyNumber('V2X_MAX_PEERS') === V2X_MAX_PEERS, `backend ${pyNumber('V2X_MAX_PEERS')}`)
const layout = /OBS_LAYOUT[^=]*= \(([\s\S]*?)\n\)/.exec(configPy)?.[1] ?? ''
const names = [...layout.matchAll(/\("(\w+)",/g)].map((m) => m[1])
check(
  '観測の V2X・死角・新標識の欄は周囲カメラの直後へ順に続く',
  names.indexOf('v2x') === names.indexOf('surround') + 1 && names.slice(names.indexOf('v2x') + 1).join() === 'occlusion,traffic_signs',
  names.join(', '),
)
const obsDim = 111
check(`モックと既定の設定の obsDim が ${obsDim}`, simStore.includes(`obsDim: ${obsDim},`) && mock.includes(`obsDim: ${obsDim},`))
const pyDefault = /^\s+v2x_comm: bool = (True|False)/m.exec(contractsPy)?.[1]
const tsDefault = /v2xComm: (true|false),/.exec(simStore)?.[1]
check(
  'v2xComm の既定値がバックエンドと一致する',
  pyDefault !== undefined && tsDefault !== undefined && (pyDefault === 'True') === (tsDefault === 'true'),
  `backend ${pyDefault} / frontend ${tsDefault}`,
)

console.log('')
if (failures > 0) {
  console.log(`NG: ${failures} 件`)
  process.exit(1)
}
console.log('OK: すべての検査に通りました')
