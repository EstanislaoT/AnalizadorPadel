import { render, screen, waitFor } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import { MemoryRouter } from 'react-router-dom'
import { Dashboard } from './Dashboard'

describe('Dashboard', () => {
  it('renders dashboard title', async () => {
    render(
      <MemoryRouter>
        <Dashboard />
      </MemoryRouter>
    )

    expect(await screen.findByTestId('dashboard-title')).toBeInTheDocument()
    expect(screen.getByText(/dashboard/i)).toBeInTheDocument()
  })

  it('fetches and displays statistics', async () => {
    render(
      <MemoryRouter>
        <Dashboard />
      </MemoryRouter>
    )

    // Esperar a que termine el loading
    await waitFor(() => {
      expect(screen.queryByRole('progressbar')).not.toBeInTheDocument()
    }, { timeout: 5000 })

    // Verificar que el contenedor de estadísticas está presente
    await waitFor(() => {
      expect(screen.getByTestId('stats-container')).toBeInTheDocument()
    }, { timeout: 5000 })

    // Verificar que aparecen estadísticas (buscar por texto de las cards)
    expect(screen.getByText('Videos')).toBeInTheDocument()
    expect(screen.getByText('Análisis')).toBeInTheDocument()
  }, 10000)

  it('renders recent videos section', async () => {
    render(
      <MemoryRouter>
        <Dashboard />
      </MemoryRouter>
    )

    await waitFor(() => {
      expect(screen.queryByRole('progressbar')).not.toBeInTheDocument()
    }, { timeout: 5000 })

    await waitFor(() => {
      expect(screen.getByTestId('recent-videos')).toBeInTheDocument()
    }, { timeout: 5000 })
  }, 10000)

  it('renders recent analyses section', async () => {
    render(
      <MemoryRouter>
        <Dashboard />
      </MemoryRouter>
    )

    await waitFor(() => {
      expect(screen.queryByRole('progressbar')).not.toBeInTheDocument()
    }, { timeout: 5000 })

    await waitFor(() => {
      expect(screen.getByTestId('recent-analyses')).toBeInTheDocument()
    }, { timeout: 5000 })
  }, 10000)
})
