/** 追従中の車の死角を路面に描く。検出した車両の陰は半透明の赤、建物の陰・視程の外は暗い紫。**Canvas の中に置くこと。** */

import { useMemo } from 'react'
import * as THREE from 'three'

import { OcclusionLayer, type OcclusionWriter } from './OcclusionLayer'
import { DYNAMIC_SHADOW_COLOR, STATIC_SHADOW_COLOR, writeShadows, type Rgb } from './occlusionGeometry'

function linear(hex: string): Rgb {
  const c = new THREE.Color(hex)
  return [c.r, c.g, c.b]
}

export function OcclusionShadowOverlay() {
  const write = useMemo<OcclusionWriter>(() => {
    const dynamic = linear(DYNAMIC_SHADOW_COLOR)
    const still = linear(STATIC_SHADOW_COLOR)
    return (view, positions, colors, capacity) =>
      writeShadows(view, (kind) => (kind === 'dynamic' ? dynamic : still), positions, colors, capacity)
  }, [])
  return <OcclusionLayer toggle="occlusionShadows" write={write} opacity={0.32} renderOrder={19} />
}
