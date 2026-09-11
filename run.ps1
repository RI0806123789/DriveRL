<#
.SYNOPSIS
    DriveRL を 1 コマンドで起動する（PowerShell 用）。

.DESCRIPTION
    バックエンド（FastAPI + PPO）を起動する。-Dev を付けると Vite の開発サーバーも
    一緒に立ち上げ、Ctrl+C でまとめて止める。venv のパスを打たずに済ませるための
    薄いラッパーで、中身は backend/run.py に任せている。

    cmd.exe やエクスプローラーからは run.cmd を使うこと（.ps1 は Windows の既定で
    メモ帳に関連付けられているため、PowerShell 以外から叩くと中身が開くだけになる）。

.PARAMETER Dev
    Vite の開発サーバーも起動する。ブラウザでは http://localhost:5173 を開く。
    フロントの変更が即反映される代わりに、メモリを 300MB ほど余計に使う。

.PARAMETER Build
    起動前に `npm run build` を実行する。以降はバックエンド単体（8000 番）で
    画面まで配信できるので、Vite を動かさずに済む。

.EXAMPLE
    .\run.ps1 -Dev
    バックエンドと Vite を一緒に起動する。開発中はこれ。

.EXAMPLE
    .\run.ps1
    ビルド済みのフロントを 8000 番で配信する。メモリが軽い。

.EXAMPLE
    .\run.ps1 -Build
    ビルドしてから 8000 番で配信する。
#>
param(
    [switch]$Dev,
    [switch]$Build
)

$ErrorActionPreference = 'Stop'

$python = Join-Path $PSScriptRoot 'backend\.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    Write-Host '仮想環境が見つかりません: backend\.venv' -ForegroundColor Red
    Write-Host '先に次を実行してください:' -ForegroundColor Yellow
    Write-Host '  py -3.13 -m venv backend\.venv'
    Write-Host '  backend\.venv\Scripts\python.exe -m pip install -r requirements.txt'
    exit 1
}

# run.py 側の引数へ変換する
$runArgs = @()
if ($Dev)   { $runArgs += '--dev' }
if ($Build) { $runArgs += '--build' }

& $python (Join-Path $PSScriptRoot 'backend\run.py') @runArgs
exit $LASTEXITCODE
