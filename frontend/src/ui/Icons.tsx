/**
 * インライン SVG アイコン。
 * アイコンライブラリは導入していないので、必要なものを自前で書いている。
 * currentColor を使うので、親の color がそのまま効く。
 */

import type { ReactNode } from 'react'

export interface IconProps {
  /** 一辺の px */
  size?: number
  className?: string
}

interface PathIconProps extends IconProps {
  children: ReactNode
  /** true なら fill 塗り、false なら stroke 描き */
  filled?: boolean
  viewBox?: string
}

function Svg({ size = 20, className, children, filled = false, viewBox = '0 0 24 24' }: PathIconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox={viewBox}
      className={className}
      aria-hidden="true"
      focusable="false"
      fill={filled ? 'currentColor' : 'none'}
      stroke={filled ? 'none' : 'currentColor'}
      strokeWidth={filled ? undefined : 1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {children}
    </svg>
  )
}

export const MenuIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M4 7h16M4 12h16M4 17h16" />
  </Svg>
)

export const CloseIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M6 6l12 12M18 6L6 18" />
  </Svg>
)

export const PlayIcon = (p: IconProps) => (
  <Svg {...p} filled>
    <path d="M8 5.5v13l11-6.5z" />
  </Svg>
)

export const PauseIcon = (p: IconProps) => (
  <Svg {...p} filled>
    <rect x="7" y="5" width="3.6" height="14" rx="1.4" />
    <rect x="13.4" y="5" width="3.6" height="14" rx="1.4" />
  </Svg>
)

export const RefreshIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M20 12a8 8 0 1 1-2.6-5.9" />
    <path d="M20 4v4h-4" />
  </Svg>
)

export const MapIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M9 4L3.5 6v14L9 18l6 2 5.5-2V4L15 6z" />
    <path d="M9 4v14M15 6v14" />
  </Svg>
)

export const ChartIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M4 19V5" />
    <path d="M4 19h16" />
    <path d="M7.5 15l3.5-4.5 3 2.5L20 7" />
  </Svg>
)

export const EyeIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M2.5 12S6 5.8 12 5.8 21.5 12 21.5 12 18 18.2 12 18.2 2.5 12 2.5 12z" />
    <circle cx="12" cy="12" r="3" />
  </Svg>
)

export const CarIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M4.5 16.5v2a1 1 0 0 0 1 1h1.5a1 1 0 0 0 1-1v-2" />
    <path d="M16 16.5v2a1 1 0 0 0 1 1h1.5a1 1 0 0 0 1-1v-2" />
    <path d="M3 16.5v-4l2-5.2A2 2 0 0 1 6.9 6h10.2a2 2 0 0 1 1.9 1.3l2 5.2v4z" />
    <path d="M3 12.8h18" />
    <circle cx="7" cy="14.6" r="0.9" fill="currentColor" stroke="none" />
    <circle cx="17" cy="14.6" r="0.9" fill="currentColor" stroke="none" />
  </Svg>
)

export const ConeIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M12 4l5.5 14h-11z" />
    <path d="M9.4 12h5.2" />
    <path d="M4 19.5h16" />
  </Svg>
)

export const SaveIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M5 4h11l4 4v12H5z" />
    <path d="M8 4v5h7V4" />
    <rect x="8" y="13" width="8" height="7" />
  </Svg>
)

export const DownloadIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M12 4v11" />
    <path d="M7.5 11.5L12 16l4.5-4.5" />
    <path d="M4.5 19.5h15" />
  </Svg>
)

export const TrashIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M4.5 6.5h15" />
    <path d="M9.5 6.5V4.8a1 1 0 0 1 1-1h3a1 1 0 0 1 1 1v1.7" />
    <path d="M6.5 6.5l1 12.2a1.4 1.4 0 0 0 1.4 1.3h6.2a1.4 1.4 0 0 0 1.4-1.3l1-12.2" />
  </Svg>
)

export const WarningIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M12 4.2L21 19.5H3z" />
    <path d="M12 10v4.2" />
    <circle cx="12" cy="17" r="0.9" fill="currentColor" stroke="none" />
  </Svg>
)

export const InfoIcon = (p: IconProps) => (
  <Svg {...p}>
    <circle cx="12" cy="12" r="8.5" />
    <path d="M12 11v5.5" />
    <circle cx="12" cy="8" r="0.9" fill="currentColor" stroke="none" />
  </Svg>
)

export const CheckIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M5 12.6l4.4 4.4L19 7.5" />
  </Svg>
)

export const ChevronDownIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M6.5 9.5L12 15l5.5-5.5" />
  </Svg>
)

export const CameraIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M3.5 8.5h3.2l1.5-2.3h7.6l1.5 2.3h3.2v10H3.5z" />
    <circle cx="12" cy="13" r="3.3" />
  </Svg>
)

export const TargetIcon = (p: IconProps) => (
  <Svg {...p}>
    <circle cx="12" cy="12" r="7.8" />
    <circle cx="12" cy="12" r="3.4" />
    <path d="M12 1.8v3.2M12 19v3.2M1.8 12H5M19 12h3.2" />
  </Svg>
)

export const TuneIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M4 7.5h9M17.5 7.5H20M4 16.5h3.5M12 16.5h8" />
    <circle cx="15" cy="7.5" r="2.3" />
    <circle cx="9.5" cy="16.5" r="2.3" />
  </Svg>
)

export const BrainIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M12 5.2a2.7 2.7 0 0 0-5 1.4A2.6 2.6 0 0 0 5 9.2a2.7 2.7 0 0 0 .9 2 2.6 2.6 0 0 0 .6 3.5A2.7 2.7 0 0 0 12 18.8z" />
    <path d="M12 5.2a2.7 2.7 0 0 1 5 1.4 2.6 2.6 0 0 1 2 2.6 2.7 2.7 0 0 1-.9 2 2.6 2.6 0 0 1-.6 3.5A2.7 2.7 0 0 1 12 18.8z" />
    <path d="M12 5.2v13.6" />
  </Svg>
)

export const PlugIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M9 3.5v5M15 3.5v5" />
    <path d="M6.5 8.5h11v3a5.5 5.5 0 0 1-11 0z" />
    <path d="M12 17v3.5" />
  </Svg>
)
