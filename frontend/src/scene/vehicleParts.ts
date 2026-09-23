/** 車両を描き分ける部品の一覧（instancedMesh 1 つずつ）。**React から切り離した純粋モジュール。** */

import { PLATES_PER_VEHICLE } from './licensePlate.ts'
import { GAUGE_SLOTS, PEDAL_SLOTS, TURN_INDICATOR_SLOTS, WHEEL_OFFSETS } from './vehicleGeometry.ts'
import { LIGHTS_PER_VEHICLE } from './vehicleLights.ts'
import { WIPER_PIVOTS } from './wiper.ts'

export type PartKey =
  | 'bodyNear'
  | 'trim'
  | 'chrome'
  | 'interior'
  | 'steering'
  | 'pedal'
  | 'gauge'
  | 'needle'
  | 'indicator'
  | 'tyre'
  | 'rim'
  | 'caliper'
  | 'wiper'
  | 'bodyFar'
  | 'wheelFar'
  | 'glass'
  | 'lamp'
  | 'plate'
  | 'sign'
  | 'pool'

/** マテリアルの名前（`vehicleMaterials.ts` の `VehicleMaterials` のキー） */
export type PartMaterial =
  | 'paint'
  | 'farPaint'
  | 'trim'
  | 'chrome'
  | 'glass'
  | 'interior'
  | 'controls'
  | 'gauge'
  | 'needle'
  | 'indicator'
  | 'tyre'
  | 'rim'
  | 'caliper'
  | 'farWheel'
  | 'lamp'
  | 'plate'
  | 'sign'
  | 'pool'

/** `tier` は近景だけ・遠景だけ・両方（共通）のどれで出すか */
export interface PartSpec {
  readonly key: PartKey
  readonly per: number
  readonly tier: 'near' | 'far' | 'both'
  readonly material: PartMaterial
  readonly shadow?: boolean
  readonly renderOrder?: number
  /** 実用モードだけ・前照灯を点けたときだけ出す */
  readonly when?: 'practical' | 'headlights'
}

export const PEDALS_PER_VEHICLE = PEDAL_SLOTS.length
export const GAUGES_PER_VEHICLE = GAUGE_SLOTS.length
export const INDICATORS_PER_VEHICLE = TURN_INDICATOR_SLOTS.length
export const WHEELS_PER_VEHICLE = WHEEL_OFFSETS.length
export const WIPERS_PER_VEHICLE = WIPER_PIVOTS.length

/** 部品の一覧。**instancedMesh を足すときは verify:vehicles の上限と CLAUDE.md の表を直すこと** */
export const VEHICLE_PARTS: ReadonlyArray<PartSpec> = [
  { key: 'bodyNear', per: 1, tier: 'near', material: 'paint', shadow: true },
  { key: 'trim', per: 1, tier: 'near', material: 'trim', shadow: true },
  { key: 'chrome', per: 1, tier: 'near', material: 'chrome' },
  { key: 'interior', per: 1, tier: 'near', material: 'interior' },
  { key: 'steering', per: 1, tier: 'near', material: 'controls' },
  { key: 'pedal', per: PEDALS_PER_VEHICLE, tier: 'near', material: 'controls' },
  { key: 'gauge', per: GAUGES_PER_VEHICLE, tier: 'near', material: 'gauge' },
  { key: 'needle', per: GAUGES_PER_VEHICLE, tier: 'near', material: 'needle' },
  { key: 'indicator', per: INDICATORS_PER_VEHICLE, tier: 'near', material: 'indicator' },
  { key: 'tyre', per: WHEELS_PER_VEHICLE, tier: 'near', material: 'tyre', shadow: true },
  { key: 'rim', per: WHEELS_PER_VEHICLE, tier: 'near', material: 'rim' },
  { key: 'caliper', per: WHEELS_PER_VEHICLE, tier: 'near', material: 'caliper' },
  { key: 'wiper', per: WIPERS_PER_VEHICLE, tier: 'near', material: 'trim' },
  { key: 'bodyFar', per: 1, tier: 'far', material: 'farPaint', shadow: true },
  { key: 'wheelFar', per: WHEELS_PER_VEHICLE, tier: 'far', material: 'farWheel', shadow: true },
  // ガラスは depthWrite を切ってあるので、内装より後に描く
  { key: 'glass', per: 1, tier: 'both', material: 'glass', renderOrder: 2 },
  { key: 'lamp', per: LIGHTS_PER_VEHICLE, tier: 'both', material: 'lamp' },
  { key: 'plate', per: PLATES_PER_VEHICLE, tier: 'both', material: 'plate' },
  { key: 'sign', per: 1, tier: 'both', material: 'sign', when: 'practical' },
  { key: 'pool', per: 1, tier: 'both', material: 'pool', renderOrder: 1, when: 'headlights' },
]

export const NEAR_PARTS = VEHICLE_PARTS.filter((p) => p.tier === 'near')
export const FAR_PARTS = VEHICLE_PARTS.filter((p) => p.tier === 'far')
