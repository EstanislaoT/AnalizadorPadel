import { render, screen, waitFor } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import VideoPlayer from './VideoPlayer'

describe('VideoPlayer', () => {
  it('renders video element', () => {
    render(<VideoPlayer videoId={1} />)

    const video = document.querySelector('video')
    expect(video).toBeInTheDocument()
  })

  it('displays loading state initially', () => {
    render(<VideoPlayer videoId={1} />)

    const loader = screen.getByText(/loading video/i)
    expect(loader).toBeInTheDocument()
  })

  it('calls onError when video fails to load', () => {
    const onError = vi.fn()
    render(
      <VideoPlayer
        videoId={999}
        onError={onError}
      />
    )

    const video = document.querySelector('video')
    if (video) {
      video.dispatchEvent(new Event('error'))
      expect(onError).toHaveBeenCalled()
    }
  })

  it('renders with correct src attribute', () => {
    const videoId = 1
    render(<VideoPlayer videoId={videoId} />)

    const video = document.querySelector('video')
    expect(video).toHaveAttribute('src', expect.stringContaining(`/videos/${videoId}/stream`))
  })

  it('renders with controls by default', () => {
    render(<VideoPlayer videoId={1} />)

    const video = document.querySelector('video')
    expect(video).toHaveAttribute('controls')
  })

  it('renders without controls when specified', () => {
    render(<VideoPlayer videoId={1} controls={false} />)

    const video = document.querySelector('video')
    expect(video).not.toHaveAttribute('controls')
  })

  it('applies custom className', () => {
    const { container } = render(<VideoPlayer videoId={1} className="custom-video" />)

    expect(container.firstChild).toHaveClass('custom-video')
  })

  it('displays error message when video fails', async () => {
    render(<VideoPlayer videoId={999} />)

    const video = document.querySelector('video')
    if (video) {
      video.dispatchEvent(new Event('error'))
      await waitFor(() => {
        const errorMessage = screen.getByText(/failed to load video/i)
        expect(errorMessage).toBeInTheDocument()
      })
    }
  })
})
