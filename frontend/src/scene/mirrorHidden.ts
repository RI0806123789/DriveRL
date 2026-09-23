/** ミラーの映像には描かないもの（カメラに貼った雨の板・追跡の印など）。ミラーを描く間だけ隠す */

import { useEffect, type RefObject } from 'react'
import type * as THREE from 'three'

export const hiddenFromMirrors = new Set<THREE.Object3D>()

/** `ref` の指す物をミラーの映像から外す（アンマウントで戻す） */
export function useHiddenFromMirrors(ref: RefObject<THREE.Object3D | null>): void {
  useEffect(() => {
    const object = ref.current
    if (!object) return
    hiddenFromMirrors.add(object)
    return () => {
      hiddenFromMirrors.delete(object)
    }
  }, [ref])
}
