/** ナンバープレート（日本の中型・自家用）の寸法・文字組み・配置。**純粋モジュール。** */

/** 中型プレートの実寸 [m]（330mm × 165mm） */
export const PLATE_W = 0.33
export const PLATE_H = 0.165

/** 取り付け位置。 */
export const PLATE_SLOTS: ReadonlyArray<{
  readonly position: readonly [number, number, number]
  /** 板の向き [rad]。前は車の前方、後ろは真後ろを向く */
  readonly yaw: number
}> = [
  { position: [2.37, 0.44, 0], yaw: 0 },
  { position: [-2.23, 0.46, 0], yaw: Math.PI },
]

export const PLATES_PER_VEHICLE = PLATE_SLOTS.length

/** 自家用（白地・緑文字）の配色。事業用の緑地は使わない */
export const PLATE_BG = '#f4f4ee'
export const PLATE_INK = '#1a6b3c'

/** 分類番号。車体は全幅 1.8m で 5 ナンバー枠（1.7m）を超えるので 3 ナンバー */
export const PLATE_CLASS = '300'

/** ひらがな。自家用に使われる文字から 1 つ（レンタカーの「わ」・事業用は使わない） */
export const PLATE_KANA = 'さ'

/** マップのプリセットに対応する陸運支局の地名。未知のプリセットは「品川」 */
const REGIONS: Record<string, string> = {
  ginza: '品川',
  kanazawa: '石川',
  umeda: 'なにわ',
  sakae: '名古屋',
}

export const DEFAULT_REGION = '品川'

/** プリセット id から地名を引く。 */
export function plateRegion(presetId: string | null | undefined): string {
  if (!presetId) return DEFAULT_REGION
  return REGIONS[presetId] ?? DEFAULT_REGION
}

/** 一連指定番号を日本の様式で組み立てる。 */
export function plateSerial(value: number): string {
  const n = Math.max(0, Math.floor(value))
  const digits = String(n % 10000)
  if (digits.length >= 4) return `${digits.slice(0, 2)}-${digits.slice(2)}`
  return '・'.repeat(4 - digits.length) + digits
}

/** プレート 1 枚ぶんの文字。上段と下段に分けて返す。 */
export interface PlateText {
  /** 上段の地名 */
  region: string
  /** 上段の分類番号 */
  classNumber: string
  /** 下段のひらがな */
  kana: string
  /** 下段の一連指定番号 */
  serial: string
}

export function plateTextFor(vehicleId: number, presetId: string | null | undefined): PlateText {
  return {
    region: plateRegion(presetId),
    classNumber: PLATE_CLASS,
    kana: PLATE_KANA,
    serial: plateSerial(vehicleId),
  }
}

/** プレートに書かれているとおりの 1 行表記（「品川 300 さ ・・・0」）。 */
export function plateLabel(vehicleId: number, presetId: string | null | undefined): string {
  const t = plateTextFor(vehicleId, presetId)
  return `${t.region} ${t.classNumber} ${t.kana} ${t.serial}`
}

/** サーバーが作った文言の「車両 #N」をプレートの表記へ置き換える。 */
export function withPlateNames(
  message: string,
  presetId: string | null | undefined,
): string {
  return message.replace(/車両 #(\d+)/g, (_, digits: string) =>
    plateLabel(Number(digits), presetId),
  )
}

/** アトラス 1 セルの画素数。プレートの縦横比（2:1）に合わせる */
export const CELL_W = 512
export const CELL_H = 256

/** アトラスの何段目を使うかを、テクスチャ座標の v に写す係数。 */
export function plateUvRow(vehicleId: number, count: number): number {
  return Math.max(0, count - 1 - vehicleId)
}
