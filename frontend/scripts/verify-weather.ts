/** 天候を 3D の見え方へ落とす計算を検証する（ブラウザ不要）。 */

import { LIGHT_SCENE } from '../src/scene/palette.ts'
import {
  advanceWeather,
  approach,
  blendDistance,
  displayedWeather,
  fogLift,
  smoothstep,
  weatherLook,
} from '../src/scene/weatherView.ts'
import type { WeatherState } from '../src/types/protocol.ts'

let failures = 0

function check(label: string, ok: boolean, detail = ''): void {
  console.log(`  [${ok ? 'OK  ' : 'NG  '}] ${label}${detail ? ` — ${detail}` : ''}`)
  if (!ok) failures += 1
}

/** バックエンド `percep/weather.py` の `visibility_m()` と同じ式。 */
const FAR_M = 120
const MIN_VISIBILITY_M = 15
function visibilityOf(fog: number): number {
  if (fog <= 1e-3) return FAR_M
  return FAR_M * Math.pow(MIN_VISIBILITY_M / FAR_M, fog * fog)
}

function weather(rain: number, fog: number): WeatherState {
  return { rain, fog, visibility: visibilityOf(fog) }
}

const CLEAR_NEAR = LIGHT_SCENE.fogNear
const CLEAR_FAR = LIGHT_SCENE.fogFar

/** 運転席・追従・俯瞰のおおよその視点の高さ [m] */
const EYE_DRIVER = 1.2
const EYE_CHASE = 12
const EYE_OVERHEAD = 320

console.log('='.repeat(70))
console.log('1. 晴れのときは既存の遠景フォグをそのまま使う')
console.log('='.repeat(70))
{
  for (const eye of [EYE_DRIVER, EYE_CHASE, EYE_OVERHEAD]) {
    const look = weatherLook(weather(0, 0), CLEAR_NEAR, CLEAR_FAR, eye)
    check(
      `視点 ${eye}m で near/far が配色どおり`,
      look.fogNear === CLEAR_NEAR && look.fogFar === CLEAR_FAR,
      `${look.fogNear} / ${look.fogFar}`,
    )
  }
  const rainy = weatherLook(weather(1, 0), CLEAR_NEAR, CLEAR_FAR, EYE_DRIVER)
  check(
    '雨だけでは視程を縮めない（雨は画を濁らせるだけ）',
    rainy.fogFar === CLEAR_FAR,
    `far=${rainy.fogFar}`,
  )
  check('雨は暗くする', rainy.dim < 0.7, `dim=${rainy.dim.toFixed(2)}`)
  check('雨粒が降る', rainy.dropOpacity > 0, `opacity=${rainy.dropOpacity.toFixed(2)}`)
  check(
    '晴れでは雨粒を描かない',
    weatherLook(weather(0, 0), CLEAR_NEAR, CLEAR_FAR).dropOpacity === 0,
  )
}

console.log()
console.log('='.repeat(70))
console.log('2. 霧が濃いほど遠くが見えなくなる（単調）')
console.log('='.repeat(70))
{
  let monotonic = true
  let prev = Infinity
  const row: string[] = []
  for (const fog of [0, 0.2, 0.4, 0.6, 0.8, 1.0]) {
    const look = weatherLook(weather(0, fog), CLEAR_NEAR, CLEAR_FAR, EYE_DRIVER)
    if (look.fogFar > prev) monotonic = false
    prev = look.fogFar
    row.push(`${fog.toFixed(1)}→${Math.round(look.fogFar)}m`)
  }
  check('fog を上げると fogFar は縮む一方', monotonic, row.join(' '))

  const heavy = weatherLook(weather(0, 1), CLEAR_NEAR, CLEAR_FAR, EYE_DRIVER)
  check('near < far は常に保たれる', heavy.fogNear < heavy.fogFar,
    `${heavy.fogNear.toFixed(1)} < ${heavy.fogFar.toFixed(1)}`)
  check('濃霧でも 9m 未満まで閉じない', heavy.fogFar >= 9, `${heavy.fogFar.toFixed(1)}m`)
}

console.log()
console.log('='.repeat(70))
console.log('3. 視点が高いほど霧を緩める（俯瞰で街が消えないこと）')
console.log('='.repeat(70))
{
  const driver = weatherLook(weather(0, 0.95), CLEAR_NEAR, CLEAR_FAR, EYE_DRIVER)
  const chase = weatherLook(weather(0, 0.95), CLEAR_NEAR, CLEAR_FAR, EYE_CHASE)
  const overhead = weatherLook(weather(0, 0.95), CLEAR_NEAR, CLEAR_FAR, EYE_OVERHEAD)

  check(
    '運転席では視程どおり近くで閉じる',
    driver.fogFar < 40,
    `far=${driver.fogFar.toFixed(1)}m（視程 ${visibilityOf(0.95).toFixed(1)}m）`,
  )
  check('追従は運転席と同程度', Math.abs(chase.fogFar - driver.fogFar) < 1e-6,
    `far=${chase.fogFar.toFixed(1)}m`)
  check(
    '俯瞰では街の一辺（900m）ぶんは見える',
    overhead.fogFar > 400,
    `far=${overhead.fogFar.toFixed(0)}m`,
  )
  check('高いほど緩む（単調）', overhead.fogFar > chase.fogFar && chase.fogFar >= driver.fogFar)
  check('視点 0m でも 1 倍を下回らない', fogLift(0) === 1 && fogLift(-50) > 1)
}

console.log()
console.log('='.repeat(70))
console.log('4. 補間そのもの')
console.log('='.repeat(70))
{
  check('smoothstep は 0 と 1 を保つ', smoothstep(0) === 0 && smoothstep(1) === 1)
  check('smoothstep は範囲外を丸める', smoothstep(-3) === 0 && smoothstep(9) === 1)
  check('幾何補間は両端で一致する',
    blendDistance(2800, 40, 0) === 2800 && blendDistance(2800, 40, 1) === 40)
  const mid = blendDistance(2800, 40, 0.5)
  check('幾何補間の中点は線形補間より小さい（薄い霧から効かせるため）',
    mid < (2800 + 40) / 2, `${mid.toFixed(0)}m < ${((2800 + 40) / 2).toFixed(0)}m`)
  check('幾何補間は 0 を渡しても NaN にならない',
    Number.isFinite(blendDistance(2800, 0, 0.5)))
}

console.log()
console.log('='.repeat(70))
console.log('5. 明るさの下限')
console.log('='.repeat(70))
{
  const worst = weatherLook(weather(1, 1), CLEAR_NEAR, CLEAR_FAR, EYE_DRIVER)
  check('雨も霧も最大でも真っ暗にはしない', worst.dim >= 0.2, `dim=${worst.dim.toFixed(2)}`)
  let dimMonotonic = true
  let prevDim = Infinity
  for (const r of [0, 0.25, 0.5, 0.75, 1]) {
    const d = weatherLook(weather(r, 0), CLEAR_NEAR, CLEAR_FAR).dim
    if (d > prevDim) dimMonotonic = false
    prevDim = d
  }
  check('雨が強いほど暗い（単調）', dimMonotonic)
}

console.log()
console.log('='.repeat(70))
console.log('6. 路面の濡れと、天候の移り変わり')
console.log('='.repeat(70))
{
  check(
    '雨が強いほど濡れる',
    weatherLook(weather(1, 0), CLEAR_NEAR, CLEAR_FAR).wetness >
      weatherLook(weather(0.3, 0), CLEAR_NEAR, CLEAR_FAR).wetness,
  )
  check('晴れでは乾いている', weatherLook(weather(0, 0), CLEAR_NEAR, CLEAR_FAR).wetness === 0)
  check(
    '濡れ具合は明示された値が優先される（乾きかけを表せる）',
    weatherLook({ ...weather(0, 0), wet: 0.6 }, CLEAR_NEAR, CLEAR_FAR).wetness === 0.6,
  )

  check('approach は目標を追い越さない', approach(0, 1, 0.016, 1.6) < 1)
  check('approach は dt を大きくすると目標へ寄る', approach(0, 1, 100, 1.6) > 0.99)
  check('approach の tau=0 は即座に目標', approach(0, 1, 0.016, 0) === 1)

  // displayedWeather はモジュールの状態なので、ここは最後に動かす
  const dt = 1 / 60
  const rainy = { rain: 1, fog: 0, visibility: FAR_M }
  for (let i = 0; i < 60 * 6; i += 1) advanceWeather(rainy, dt)
  check('数秒で雨に追いつく', displayedWeather.rain > 0.95,
    `rain=${displayedWeather.rain.toFixed(3)}`)
  check('路面も濡れている', displayedWeather.wet > 0.9, `wet=${displayedWeather.wet.toFixed(3)}`)

  const dry = { rain: 0, fog: 0, visibility: FAR_M }
  for (let i = 0; i < 60 * 3; i += 1) advanceWeather(dry, dt)
  check(
    '雨がやんだ直後は空が先に晴れる',
    displayedWeather.rain < 0.2,
    `rain=${displayedWeather.rain.toFixed(3)}`,
  )
  check(
    '路面はまだ濡れている（乾くのは遅い）',
    displayedWeather.wet > displayedWeather.rain,
    `wet=${displayedWeather.wet.toFixed(3)} > rain=${displayedWeather.rain.toFixed(3)}`,
  )

  for (let i = 0; i < 60 * 40; i += 1) advanceWeather(dry, dt)
  check('十分おけば乾く', displayedWeather.wet < 0.02, `wet=${displayedWeather.wet.toFixed(4)}`)
}

console.log()
console.log('='.repeat(70))
if (failures > 0) {
  console.log(`結果: ${failures} 件の不合格`)
  process.exit(1)
}
console.log('結果: すべて合格')
