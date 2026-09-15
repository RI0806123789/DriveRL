/** いまの配色を 3D 側から読むためのフック。 */

import { useSimStore } from '../store/simStore'
import { scenePalette, type ScenePalette } from './palette'

export function usePalette(): ScenePalette {
  return scenePalette(useSimStore((s) => s.theme))
}
