/** 車両の色が 64 台ぶん見分けられるかを検証する（ブラウザ不要）。 */

import { VEHICLE_COLORS, vehicleColor } from '../src/scene/vehicleColors.ts'

const MAX_VEHICLES = 64

let failures = 0

function check(label: string, ok: boolean, detail = ''): void {
  console.log(`  [${ok ? 'OK  ' : 'NG  '}] ${label}${detail ? ` — ${detail}` : ''}`)
  if (!ok) failures += 1
}

function parseColor(css: string): [number, number, number] {
  const hex = /^#([0-9a-f]{6})$/i.exec(css)
  if (hex) {
    const n = parseInt(hex[1], 16)
    return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255]
  }
  const hsl = /^hsl\(\s*([\d.]+)\s*,\s*([\d.]+)%\s*,\s*([\d.]+)%\s*\)$/.exec(css)
  if (!hsl) throw new Error(`色を解釈できない: ${css}`)
  const h = Number(hsl[1]) / 360
  const s = Number(hsl[2]) / 100
  const l = Number(hsl[3]) / 100
  if (s === 0) return [l, l, l]
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s
  const p = 2 * l - q
  const f = (t: number): number => {
    let x = t
    if (x < 0) x += 1
    if (x > 1) x -= 1
    if (x < 1 / 6) return p + (q - p) * 6 * x
    if (x < 1 / 2) return q
    if (x < 2 / 3) return p + (q - p) * (2 / 3 - x) * 6
    return p
  }
  return [f(h + 1 / 3), f(h), f(h - 1 / 3)]
}

/** sRGB -> CIELAB。差が知覚の差におおよそ比例する空間で比べたいので通す。 */
function toLab(rgb: [number, number, number]): [number, number, number] {
  const lin = rgb.map((c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4))
  const [r, g, b] = lin
  const x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
  const y = 0.2126 * r + 0.7152 * g + 0.0722 * b
  const z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883
  const f = (t: number): number => (t > 216 / 24389 ? Math.cbrt(t) : (841 / 108) * t + 4 / 29)
  const fx = f(x)
  const fy = f(y)
  const fz = f(z)
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)]
}

function labDistance(a: string, b: string): number {
  const la = toLab(parseColor(a))
  const lb = toLab(parseColor(b))
  return Math.hypot(la[0] - lb[0], la[1] - lb[1], la[2] - lb[2])
}

console.log('='.repeat(70))
console.log(`車両の色（${MAX_VEHICLES} 台）`)
console.log('='.repeat(70))

const colors = Array.from({ length: MAX_VEHICLES }, (_, i) => vehicleColor(i))

check(
  `パレットが ${MAX_VEHICLES} 色ある`,
  VEHICLE_COLORS.length >= MAX_VEHICLES,
  `${VEHICLE_COLORS.length} 色`,
)
check(
  'パレットの色をそのまま返す',
  VEHICLE_COLORS.every((c, i) => colors[i] === c),
)

const unique = new Set(colors)
check(`${MAX_VEHICLES} 台すべて違う色`, unique.size === MAX_VEHICLES, `${unique.size} 色`)

let worst = Infinity
let worstPair = [0, 0]
for (let i = 0; i < MAX_VEHICLES; i += 1) {
  for (let j = i + 1; j < MAX_VEHICLES; j += 1) {
    const d = labDistance(colors[i], colors[j])
    if (d < worst) {
      worst = d
      worstPair = [i, j]
    }
  }
}
check(
  'どの 2 台も色が離れている（CIELAB 距離 >= 10）',
  worst >= 10,
  `最小 ${worst.toFixed(1)}（#${worstPair[0]} と #${worstPair[1]}）`,
)

let worstAdj = Infinity
let worstAdjAt = 0
for (let i = 0; i + 1 < MAX_VEHICLES; i += 1) {
  const d = labDistance(colors[i], colors[i + 1])
  if (d < worstAdj) {
    worstAdj = d
    worstAdjAt = i
  }
}
check(
  '隣り合う番号はとくに離れている（CIELAB 距離 >= 20）',
  worstAdj >= 20,
  `最小 ${worstAdj.toFixed(1)}（#${worstAdjAt} と #${worstAdjAt + 1}）`,
)

const lightness = colors.map((c) => toLab(parseColor(c))[0])
const tooDark = lightness.filter((l) => l < 55).length
const tooLight = lightness.filter((l) => l > 95).length
check(
  '暗すぎる色が無い（L* >= 55）',
  tooDark === 0,
  `最小 L* ${Math.min(...lightness).toFixed(0)}`,
)
check(
  '明るすぎる色が無い（L* <= 95）',
  tooLight === 0,
  `最大 L* ${Math.max(...lightness).toFixed(0)}`,
)

const odd = [-1, 0, 999, 1.5, Number.NaN]
let oddOk = true
for (const id of odd) {
  try {
    parseColor(vehicleColor(id))
  } catch {
    oddOk = false
  }
}
check('負・巨大・NaN の id でも色を返す', oddOk, odd.join(' / '))

console.log()
console.log(`  ${MAX_VEHICLES} 台ぶんの色:`)
for (let i = 0; i < MAX_VEHICLES; i += 8) {
  console.log(`    #${String(i).padStart(2)}-${String(i + 7).padStart(2)}: ${colors.slice(i, i + 8).join(' ')}`)
}

console.log()
console.log('='.repeat(70))
if (failures > 0) {
  console.log(`結果: ${failures} 件の不合格`)
  process.exit(1)
}
console.log('結果: すべて合格')
