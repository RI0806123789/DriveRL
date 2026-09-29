/** node --test で .tsx を読むためのローダー。依存を足さず、devDependencies の typescript で JSX を変換する。 */

import { existsSync, readFileSync } from 'node:fs'
import { fileURLToPath, pathToFileURL } from 'node:url'
import ts from 'typescript'

interface ResolveContext {
  parentURL?: string
}

type NextResolve = (specifier: string, context: ResolveContext) => Promise<{ url: string }>
type NextLoad = (url: string, context: object) => Promise<{ format?: string; source?: unknown }>

const EXTENSIONS = ['.ts', '.tsx']

/** Vite と同じく、拡張子を省いた相対 import を .ts / .tsx へ解決する。 */
export async function resolve(specifier: string, context: ResolveContext, next: NextResolve) {
  const relative = specifier.startsWith('./') || specifier.startsWith('../')
  if (relative && context.parentURL && !/\.[cm]?[jt]sx?$/.test(specifier)) {
    for (const ext of EXTENSIONS) {
      const url = new URL(specifier + ext, context.parentURL)
      if (existsSync(fileURLToPath(url))) return { url: url.href, shortCircuit: true }
    }
  }
  return next(specifier, context)
}

/** .tsx だけ typescript で JS へ変換する（.ts は Node の型の除去に任せる）。 */
export async function load(url: string, context: object, next: NextLoad) {
  if (!url.endsWith('.tsx')) return next(url, context)
  const source = readFileSync(fileURLToPath(url), 'utf8')
  const out = ts.transpileModule(source, {
    fileName: fileURLToPath(url),
    compilerOptions: {
      jsx: ts.JsxEmit.ReactJSX,
      module: ts.ModuleKind.ESNext,
      target: ts.ScriptTarget.ES2022,
      verbatimModuleSyntax: false,
    },
  })
  return { format: 'module', source: out.outputText, shortCircuit: true }
}

export const hooksUrl = pathToFileURL(fileURLToPath(import.meta.url)).href
