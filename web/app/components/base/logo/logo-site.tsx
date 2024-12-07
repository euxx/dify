'use client'
import type { FC } from 'react'
import classNames from '@/utils/classnames'
import { useSelector } from '@/context/app-context'

type LogoSiteProps = {
  className?: string
}

const LogoSite: FC<LogoSiteProps> = ({
  className,
}) => {
  const { theme } = useSelector((s) => {
    return {
      theme: s.theme,
    }
  })

  const src = theme === 'light' ? '/logo/logo-site.png' : `/logo/logo-site-${theme}.png`
  return (
    <div className={classNames('flex items-center', className)}>
      <img
        src={src}
        className={classNames('block w-auto h-10', className)}
        alt='logo'
      />
      <span className='ml-2 text-lg'>西南油气田知识问答工作流平台</span>
    </div>
  )
}

export default LogoSite
