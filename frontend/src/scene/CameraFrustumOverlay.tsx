/** 追従中の車の 4 台のカメラが見えている範囲（視野コーン）を、半透明のエメラルドグリーンで路面に描く。**Canvas の中に置くこと。** */

import { useMemo } from 'react'
import * as THREE from 'three'

import { OcclusionLayer, type OcclusionWriter } from './OcclusionLayer'
import { FRUSTUM_COLOR, writeSeen, type Rgb } from './occlusionGeometry'

export function CameraFrustumOverlay() {
  const write = useMemo<OcclusionWriter>(() => {
    const c = new THREE.Color(FRUSTUM_COLOR)
    const rgb: Rgb = [c.r, c.g, c.b]
    return (view, positions, colors, capacity) => writeSeen(view, rgb, positions, colors, capacity)
  }, [])
  return <OcclusionLayer toggle="cameraFrustums" write={write} opacity={0.2} renderOrder={18} />
}
