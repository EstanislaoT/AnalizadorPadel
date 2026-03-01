import { useState, useEffect } from 'react'
import { AnalizadorPadelApiService } from './services/api/generated'

function App() {
  const [videos, setVideos] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)

  useEffect(() => {
    loadVideos()
  }, [])

  const loadVideos = async () => {
    setLoading(true)
    try {
      const response = await AnalizadorPadelApiService.getVideos()
      setVideos(response.data?.data || [])
    } catch (error) {
      console.error('Error loading videos:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0])
    }
  }

  const handleUpload = async () => {
    if (!selectedFile) return

    setUploading(true)
    try {
      await AnalizadorPadelApiService.createVideo({ file: selectedFile as any })
      await loadVideos()
      setSelectedFile(null)
    } catch (error) {
      console.error('Error uploading video:', error)
    } finally {
      setUploading(false)
    }
  }

  const handleAnalyze = async (videoId: number) => {
    try {
      await AnalizadorPadelApiService.startAnalysis(videoId)
      await loadVideos()
    } catch (error) {
      console.error('Error starting analysis:', error)
    }
  }

  return (
    <div className="app">
      <header>
        <h1>🎾 Analizador de Padel</h1>
      </header>

      <main>
        <section className="upload-section">
          <h2>Subir Video</h2>
          <input type="file" accept="video/*" onChange={handleFileChange} />
          <button onClick={handleUpload} disabled={!selectedFile || uploading}>
            {uploading ? 'Subiendo...' : 'Subir'}
          </button>
        </section>

        <section className="videos-section">
          <h2>Videos</h2>
          <button onClick={loadVideos} disabled={loading}>
            {loading ? 'Cargando...' : 'Actualizar'}
          </button>

          <div className="video-list">
            {videos.map((video) => (
              <div key={video.id} className="video-card">
                <h3>{video.name}</h3>
                <p>Status: {video.status}</p>
                <p>Subido: {new Date(video.uploadedAt).toLocaleDateString()}</p>
                {video.status === 'Uploaded' && (
                  <button onClick={() => handleAnalyze(video.id)}>
                    Iniciar Análisis
                  </button>
                )}
              </div>
            ))}
          </div>
        </section>
      </main>
    </div>
  )
}

export default App
