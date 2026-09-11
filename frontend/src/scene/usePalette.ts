/**
 * いまの配色を 3D 側から読むためのフック。
 *
 * `palette.ts` を React に依存させたくない（Node の検証スクリプトから読むため）ので、
 * zustand との接続だけをこちらに切り出してある。
 *
 * 返る値は `DARK_SCENE` / `LIGHT_SCENE` そのものなので、**参照は配色ごとに安定**する。
 * useMemo の依存にそのまま入れてよい。
 */

import { useSimStore } from '../store/simStore'
import { scenePalette, type ScenePalette } from './palette'

export function usePalette(): ScenePalette {
  return scenePalette(useSimStore((s) => s.theme))
}
