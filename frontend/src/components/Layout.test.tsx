import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, it, expect } from 'vitest'
import { Layout } from './Layout'

describe('Layout', () => {
  it('renders navigation links', () => {
    render(
      <MemoryRouter>
        <Layout>
          <div>Test Content</div>
        </Layout>
      </MemoryRouter>
    )

    expect(screen.getByText('Dashboard')).toBeInTheDocument()
    expect(screen.getByText('Videos')).toBeInTheDocument()
    expect(screen.getByText('Análisis')).toBeInTheDocument()
    expect(screen.getByText('Reportes')).toBeInTheDocument()
  })

  it('renders app title', () => {
    render(
      <MemoryRouter>
        <Layout>
          <div>Test Content</div>
        </Layout>
      </MemoryRouter>
    )

    expect(screen.getByText(/analizador.*padel/i)).toBeInTheDocument()
  })

  it('renders navigation drawer or menu', () => {
    render(
      <MemoryRouter>
        <Layout>
          <div>Test Content</div>
        </Layout>
      </MemoryRouter>
    )

    const appBar = document.querySelector('.MuiAppBar-root')
    expect(appBar).toBeInTheDocument()
  })

  it('renders children content', () => {
    render(
      <MemoryRouter>
        <Layout>
          <div data-testid="test-child">Test Content</div>
        </Layout>
      </MemoryRouter>
    )

    expect(screen.getByTestId('test-child')).toBeInTheDocument()
    expect(screen.getByText('Test Content')).toBeInTheDocument()
  })
})
