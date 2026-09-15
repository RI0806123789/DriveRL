/** PWA のアイコン（PNG）を生成する。`npm run icons` で実行する。 */

import { deflateSync } from 'node:zlib'
import { mkdirSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const OUT_DIR = join(dirname(fileURLToPath(import.meta.url)), '..', 'public')

/** 1 ピクセルあたりの分割数（アンチエイリアス用のスーパーサンプリング） */
const SS = 4

type RGB = [number, number, number]

/** index.html のファビコンと同じ色 */
const COLOR = {
  bg: hex('#141a20'),
  road: hex('#3a4550'),
  body: hex('#9ecaff'),
  glass: hex('#0f1b26'),
  wheel: hex('#15181c'),
} satisfies Record<string, RGB>

function hex(s: string): RGB {
  const n = Number.parseInt(s.slice(1), 16)
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
}

type Vec = [number, number]

/** 角丸矩形の内側か */
function inRoundRect(x: number, y: number, w: number, h: number, r: number): boolean {
  if (x < 0 || y < 0 || x > w || y > h) return false
  const cx = Math.min(Math.max(x, r), w - r)
  const cy = Math.min(Math.max(y, r), h - r)
  const dx = x - cx
  const dy = y - cy
  return dx * dx + dy * dy <= r * r
}

/** 線分を太さ width で太らせた領域（両端は丸）の内側か。SVG の stroke-linecap=round 相当 */
function inCapsule(x: number, y: number, a: Vec, b: Vec, width: number): boolean {
  const r = width / 2
  const vx = b[0] - a[0]
  const vy = b[1] - a[1]
  const len2 = vx * vx + vy * vy
  let t = len2 > 0 ? ((x - a[0]) * vx + (y - a[1]) * vy) / len2 : 0
  t = Math.min(1, Math.max(0, t))
  const dx = x - (a[0] + vx * t)
  const dy = y - (a[1] + vy * t)
  return dx * dx + dy * dy <= r * r
}

/** 多角形の内側か（レイキャスト。凹でも正しい） */
function inPolygon(x: number, y: number, pts: Vec[]): boolean {
  let inside = false
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const [xi, yi] = pts[i]
    const [xj, yj] = pts[j]
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside
  }
  return inside
}

function inCircle(x: number, y: number, cx: number, cy: number, r: number): boolean {
  const dx = x - cx
  const dy = y - cy
  return dx * dx + dy * dy <= r * r
}

/** 車体。ファビコンの path をそのまま点列にしたもの（角の 1px の丸めは省く） */
const CAR_BODY: Vec[] = [
  [7, 13.5],
  [25, 13.5],
  [27.2, 17.7],
  [27.2, 21.8],
  [4.8, 21.8],
  [4.8, 17.7],
]

/** 窓（車体より暗い台形） */
const CAR_GLASS: Vec[] = [
  [9.4, 14.2],
  [22.6, 14.2],
  [24.1, 17.1],
  [7.9, 17.1],
]

/** 32x32 のユーザー座標 (x, y) を色に変える。手前の図形が勝つ。 */
function sample(x: number, y: number, rounded: boolean): RGB | null {
  if (inCircle(x, y, 10, 22, 2.4) || inCircle(x, y, 22, 22, 2.4)) return COLOR.wheel
  if (inPolygon(x, y, CAR_GLASS)) return COLOR.glass
  if (inPolygon(x, y, CAR_BODY)) return COLOR.body
  if (inCapsule(x, y, [4, 20], [28, 20], 2)) return COLOR.road
  if (!rounded) return COLOR.bg
  return inRoundRect(x, y, 32, 32, 8) ? COLOR.bg : null
}

/** アイコンを 1 枚描く。 */
function render(size: number, { inset = 0, rounded = true, lift = 0 } = {}): Buffer {
  const rgba = Buffer.alloc(size * size * 4)
  const scale = 32 / (size * (1 - inset * 2))
  const offset = (size * inset * scale * -1)

  for (let py = 0; py < size; py++) {
    for (let px = 0; px < size; px++) {
      let r = 0
      let g = 0
      let b = 0
      let a = 0
      for (let sy = 0; sy < SS; sy++) {
        for (let sx = 0; sx < SS; sx++) {
          const ux = (px + (sx + 0.5) / SS) * scale + offset
          const uy = (py + (sy + 0.5) / SS) * scale + offset + lift
          const c = sample(ux, uy, rounded)
          if (c) {
            r += c[0]
            g += c[1]
            b += c[2]
            a += 255
          }
        }
      }
      const n = SS * SS
      const i = (py * size + px) * 4
      const covered = a / 255
      rgba[i] = covered > 0 ? Math.round(r / covered) : 0
      rgba[i + 1] = covered > 0 ? Math.round(g / covered) : 0
      rgba[i + 2] = covered > 0 ? Math.round(b / covered) : 0
      rgba[i + 3] = Math.round(a / n)
    }
  }
  return encodePng(size, size, rgba)
}

const CRC_TABLE = (() => {
  const table = new Uint32Array(256)
  for (let n = 0; n < 256; n++) {
    let c = n
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
    table[n] = c >>> 0
  }
  return table
})()

function crc32(buf: Buffer): number {
  let c = 0xffffffff
  for (const byte of buf) c = CRC_TABLE[(c ^ byte) & 0xff] ^ (c >>> 8)
  return (c ^ 0xffffffff) >>> 0
}

function chunk(type: string, data: Buffer): Buffer {
  const length = Buffer.alloc(4)
  length.writeUInt32BE(data.length)
  const body = Buffer.concat([Buffer.from(type, 'ascii'), data])
  const crc = Buffer.alloc(4)
  crc.writeUInt32BE(crc32(body))
  return Buffer.concat([length, body, crc])
}

function encodePng(width: number, height: number, rgba: Buffer): Buffer {
  const ihdr = Buffer.alloc(13)
  ihdr.writeUInt32BE(width, 0)
  ihdr.writeUInt32BE(height, 4)
  ihdr[8] = 8
  ihdr[9] = 6

  const raw = Buffer.alloc(height * (width * 4 + 1))
  for (let y = 0; y < height; y++) {
    const src = y * width * 4
    const dst = y * (width * 4 + 1)
    raw[dst] = 0
    rgba.copy(raw, dst + 1, src, src + width * 4)
  }

  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(raw, { level: 9 })),
    chunk('IEND', Buffer.alloc(0)),
  ])
}

mkdirSync(OUT_DIR, { recursive: true })

const files: Array<[string, Buffer]> = [
  ['icon-192.png', render(192)],
  ['icon-512.png', render(512)],
  ['icon-maskable-512.png', render(512, { inset: 0.1, rounded: false, lift: 3 })],
]

for (const [name, buf] of files) {
  writeFileSync(join(OUT_DIR, name), buf)
  console.log(`  ${name}  ${(buf.length / 1024).toFixed(1)} KB`)
}
console.log(`アイコンを ${files.length} 枚生成しました: ${OUT_DIR}`)
