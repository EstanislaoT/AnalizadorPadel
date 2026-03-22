using System.Net;
using System.Net.Http.Json;
using AnalizadorPadel.Api.Data;
using AnalizadorPadel.Api.Models.DTOs;
using AnalizadorPadel.Api.Models.Entities;
using AnalizadorPadel.Api.Tests.Infrastructure;
using Microsoft.EntityFrameworkCore;
using TechTalk.SpecFlow;

namespace AnalizadorPadel.Api.Tests.BDD.StepDefinitions;

[Binding]
public class AnalysisSteps : IntegrationTestBase
{
    private HttpResponseMessage? _response;
    private ApiResponse<AnalysisStats>? _statsResponse;
    private ApiResponse<HeatmapData>? _heatmapResponse;
    private ApiResponse<DashboardStats>? _dashboardResponse;
    private ApiResponse<string>? _reportResponse;

    public AnalysisSteps(CustomWebApplicationFactory factory) : base(factory)
    {
    }

    [Given("existe un análisis completado en el sistema")]
    public async Task GivenExisteUnAnalisisCompletadoEnElSistema()
    {
        await CreateTestAnalysisAsync(AnalysisStatus.Completed);
    }

    [Given("existe un análisis con ID (\\d+) en estado \"(.*)\"")]
    public async Task GivenExisteUnAnalisisConIDEnEstado(int id, string status)
    {
        var analysisStatus = Enum.Parse<AnalysisStatus>(status);
        await CreateTestAnalysisAsync(analysisStatus);
    }

    [Given("no existe un análisis con ID (\\d+)")]
    public void GivenNoExisteUnAnalisisConID(int id)
    {
        // No action needed - database starts empty
    }

    [Given("existen (\\d+) videos en el sistema")]
    public async Task GivenExistenVideosEnElSistema(int count)
    {
        using var scope = Factory.Services.CreateScope();
        var dbContext = scope.ServiceProvider.GetRequiredService<PadelDbContext>();

        for (int i = 0; i < count; i++)
        {
            var video = new VideoEntity
            {
                Name = $"Video {i + 1}",
                FilePath = $"/path/video{i + 1}.mp4",
                FileSizeBytes = 1024,
                FileExtension = ".mp4",
                UploadedAt = DateTime.UtcNow.AddMinutes(-i),
                Status = nameof(VideoStatus.Uploaded)
            };
            dbContext.Videos.Add(video);
        }
        await dbContext.SaveChangesAsync();
    }

    [Given("existen (\\d+) análisis completados")]
    public async Task GivenExistenAnalisisCompletados(int count)
    {
        using var scope = Factory.Services.CreateScope();
        var dbContext = scope.ServiceProvider.GetRequiredService<PadelDbContext>();

        for (int i = 0; i < count; i++)
        {
            var video = new VideoEntity
            {
                Name = $"Video {i + 1}",
                FilePath = $"/path/video{i + 1}.mp4",
                FileSizeBytes = 1024,
                FileExtension = ".mp4",
                UploadedAt = DateTime.UtcNow.AddMinutes(-i),
                Status = nameof(VideoStatus.Completed)
            };
            dbContext.Videos.Add(video);
            await dbContext.SaveChangesAsync();

            var analysis = new AnalysisEntity
            {
                VideoId = video.Id,
                Status = nameof(AnalysisStatus.Completed),
                StartedAt = DateTime.UtcNow.AddHours(-i),
                CompletedAt = DateTime.UtcNow.AddHours(-i).AddMinutes(10),
                TotalFrames = 1000,
                PlayersDetected = 4,
                DetectionRatePercent = 85.0,
                ModelUsed = "yolov8m.pt"
            };
            dbContext.Analyses.Add(analysis);
            await dbContext.SaveChangesAsync();
        }
    }

    [Given("existen (\\d+) análisis fallido")]
    public async Task GivenExistenAnalisisFallido(int count)
    {
        using var scope = Factory.Services.CreateScope();
        var dbContext = scope.ServiceProvider.GetRequiredService<PadelDbContext>();

        for (int i = 0; i < count; i++)
        {
            var video = new VideoEntity
            {
                Name = $"Failed Video {i + 1}",
                FilePath = $"/path/failed{i + 1}.mp4",
                FileSizeBytes = 1024,
                FileExtension = ".mp4",
                UploadedAt = DateTime.UtcNow.AddMinutes(-i),
                Status = nameof(VideoStatus.Failed)
            };
            dbContext.Videos.Add(video);
            await dbContext.SaveChangesAsync();

            var analysis = new AnalysisEntity
            {
                VideoId = video.Id,
                Status = nameof(AnalysisStatus.Failed),
                StartedAt = DateTime.UtcNow.AddHours(-i),
                CompletedAt = DateTime.UtcNow.AddHours(-i).AddMinutes(5),
                ErrorMessage = "Processing timeout"
            };
            dbContext.Analyses.Add(analysis);
            await dbContext.SaveChangesAsync();
        }
    }

    [Given("se han completado (\\d+) análisis")]
    public async Task GivenSeHanCompletadoAnalisis(int count)
    {
        await GivenExistenAnalisisCompletados(count);
    }

    [Given("el usuario ha subido (\\d+) videos en los últimos días")]
    public async Task GivenElUsuarioHaSubidoVideosEnLosUltimosDias(int count)
    {
        await GivenExistenVideosEnElSistema(count);
    }

    [Given("existen (\\d+) análisis en total")]
    public async Task GivenExistenAnalisisEnTotal(int count)
    {
        await GivenExistenAnalisisCompletados(count);
    }

    [Given(@"(\d+) análisis están completados")]
    public async Task GivenAnalisisEstanCompletados(int count)
    {
        // Additional completed analyses - handled in the When step context
    }

    [When("el usuario solicita las estadísticas del análisis (\\d+)")]
    public async Task WhenElUsuarioSolicitaLasEstadisticasDelAnalisis(int id)
    {
        _response = await Client.GetAsync($"/api/analyses/{id}/stats");
        if (_response.IsSuccessStatusCode)
        {
            _statsResponse = await _response.Content.ReadFromJsonAsync<ApiResponse<AnalysisStats>>();
        }
    }

    [When("el usuario solicita los datos del heatmap del análisis (\\d+)")]
    public async Task WhenElUsuarioSolicitaLosDatosDelHeatmapDelAnalisis(int id)
    {
        _response = await Client.GetAsync($"/api/analyses/{id}/heatmap");
        if (_response.IsSuccessStatusCode)
        {
            _heatmapResponse = await _response.Content.ReadFromJsonAsync<ApiResponse<HeatmapData>>();
        }
    }

    [When("el usuario solicita las estadísticas del dashboard")]
    public async Task WhenElUsuarioSolicitaLasEstadisticasDelDashboard()
    {
        _response = await Client.GetAsync("/api/dashboard/stats");
        if (_response.IsSuccessStatusCode)
        {
            _dashboardResponse = await _response.Content.ReadFromJsonAsync<ApiResponse<DashboardStats>>();
        }
    }

    [When("el usuario solicita el reporte del análisis (\\d+)")]
    public async Task WhenElUsuarioSolicitaElReporteDelAnalisis(int id)
    {
        _response = await Client.GetAsync($"/api/analyses/{id}/report");
        if (_response.IsSuccessStatusCode)
        {
            _reportResponse = await _response.Content.ReadFromJsonAsync<ApiResponse<string>>();
        }
    }

    [Then("las estadísticas incluyen frames totales")]
    public void ThenLasEstadisticasIncluyenFramesTotales()
    {
        _statsResponse.Should().NotBeNull();
        _statsResponse!.Data.Should().NotBeNull();
        _statsResponse.Data!.TotalFrames.Should().BeGreaterThan(0);
    }

    [Then("las estadísticas incluyen tasa de detección")]
    public void ThenLasEstadisticasIncluyenTasaDeDeteccion()
    {
        _statsResponse.Should().NotBeNull();
        _statsResponse!.Data.Should().NotBeNull();
        _statsResponse.Data!.DetectionRatePercent.Should().BeGreaterOrEqualTo(0);
    }

    [Then("las estadísticas incluyen tiempo de procesamiento")]
    public void ThenLasEstadisticasIncluyenTiempoDeProcesamiento()
    {
        _statsResponse.Should().NotBeNull();
        _statsResponse!.Data.Should().NotBeNull();
        _statsResponse.Data!.ProcessingTimeSeconds.Should().BeGreaterOrEqualTo(0);
    }

    [Then("las estadísticas incluyen modelo utilizado")]
    public void ThenLasEstadisticasIncluyenModeloUtilizado()
    {
        _statsResponse.Should().NotBeNull();
        _statsResponse!.Data.Should().NotBeNull();
        _statsResponse.Data!.ModelUsed.Should().NotBeNullOrEmpty();
    }

    [Then("el mensaje indica \"(.*)\"")]
    public async Task ThenElMensajeIndica(string expectedMessage)
    {
        var content = await _response!.Content.ReadFromJsonAsync<ApiResponse<object>>();
        content.Should().NotBeNull();
        content!.Message.Should().Contain(expectedMessage);
    }

    [Then("el heatmap contiene (\\d+) puntos")]
    public void ThenElHeatmapContienePuntos(int expectedCount)
    {
        _heatmapResponse.Should().NotBeNull();
        _heatmapResponse!.Data.Should().NotBeNull();
        _heatmapResponse.Data!.Points.Should().HaveCount(expectedCount);
    }

    [Then("cada punto tiene coordenadas X, Y e intensidad")]
    public void ThenCadaPuntoTieneCoordenadasXYEIntensidad()
    {
        _heatmapResponse.Should().NotBeNull();
        _heatmapResponse!.Data.Should().NotBeNull();

        foreach (var point in _heatmapResponse.Data!.Points)
        {
            point.X.Should().BeGreaterOrEqualTo(0);
            point.Y.Should().BeGreaterOrEqualTo(0);
            point.Intensity.Should().BeGreaterOrEqualTo(1);
        }
    }

    [Then("las dimensiones de la cancha son \"(.*)\"")]
    public void ThenLasDimensionesDeLaCanchaSon(string expectedDimensions)
    {
        _heatmapResponse.Should().NotBeNull();
        _heatmapResponse!.Data.Should().NotBeNull();
        _heatmapResponse.Data!.CourtDimensions.Should().Be(expectedDimensions);
    }

    [Then("el dashboard muestra (\\d+) videos totales")]
    public void ThenElDashboardMuestraVideosTotales(int expectedCount)
    {
        _dashboardResponse.Should().NotBeNull();
        _dashboardResponse!.Data.Should().NotBeNull();
        _dashboardResponse.Data!.TotalVideos.Should().Be(expectedCount);
    }

    [Then("el dashboard muestra (\\d+) análisis totales")]
    public void ThenElDashboardMuestraAnalisisTotales(int expectedCount)
    {
        _dashboardResponse.Should().NotBeNull();
        _dashboardResponse!.Data.Should().NotBeNull();
        _dashboardResponse.Data!.TotalAnalyses.Should().Be(expectedCount);
    }

    [Then("el dashboard muestra (\\d+) análisis completados")]
    public void ThenElDashboardMuestraAnalisisCompletados(int expectedCount)
    {
        _dashboardResponse.Should().NotBeNull();
        _dashboardResponse!.Data.Should().NotBeNull();
        _dashboardResponse.Data!.CompletedAnalyses.Should().Be(expectedCount);
    }

    [Then("el dashboard muestra (\\d+) análisis fallido")]
    public void ThenElDashboardMuestraAnalisisFallido(int expectedCount)
    {
        _dashboardResponse.Should().NotBeNull();
        _dashboardResponse!.Data.Should().NotBeNull();
        _dashboardResponse.Data!.FailedAnalyses.Should().Be(expectedCount);
    }

    [Then("la tasa de éxito es (\\d+)%")]
    public void ThenLaTasaDeExitoEs(int expectedRate)
    {
        _dashboardResponse.Should().NotBeNull();
        _dashboardResponse!.Data.Should().NotBeNull();
        _dashboardResponse.Data!.SuccessRatePercent.Should().BeApproximately(expectedRate, 0.01);
    }

    [Then("el dashboard muestra los (\\d+) videos más recientes")]
    public void ThenElDashboardMuestraLosVideosMasRecientes(int expectedCount)
    {
        _dashboardResponse.Should().NotBeNull();
        _dashboardResponse!.Data.Should().NotBeNull();
        _dashboardResponse.Data!.RecentVideos.Should().HaveCount(expectedCount);
    }

    [Then("los videos están ordenados por fecha descendente")]
    public void ThenLosVideosEstanOrdenadosPorFechaDescendente()
    {
        _dashboardResponse.Should().NotBeNull();
        _dashboardResponse!.Data.Should().NotBeNull();
        var videos = _dashboardResponse.Data!.RecentVideos;

        for (int i = 0; i < videos.Count - 1; i++)
        {
            videos[i].UploadedAt.Should().BeOnOrAfter(videos[i + 1].UploadedAt);
        }
    }

    [Then("el dashboard muestra los (\\d+) análisis más recientes")]
    public void ThenElDashboardMuestraLosAnalisisMasRecientes(int expectedCount)
    {
        _dashboardResponse.Should().NotBeNull();
        _dashboardResponse!.Data.Should().NotBeNull();
        _dashboardResponse.Data!.RecentAnalyses.Should().HaveCountLessOrEqualTo(expectedCount);
    }

    [Then("los análisis están ordenados por fecha de inicio descendente")]
    public void ThenLosAnalisisEstanOrdenadosPorFechaDeInicioDescendente()
    {
        _dashboardResponse.Should().NotBeNull();
        _dashboardResponse!.Data.Should().NotBeNull();
        var analyses = _dashboardResponse.Data!.RecentAnalyses;

        for (int i = 0; i < analyses.Count - 1; i++)
        {
            analyses[i].StartedAt.Should().BeOnOrAfter(analyses[i + 1].StartedAt);
        }
    }

    [Then("la respuesta incluye la ruta al archivo PDF")]
    public void ThenLaRespuestaIncluyeLaRutaAlArchivoPDF()
    {
        _reportResponse.Should().NotBeNull();
        _reportResponse!.Data.Should().NotBeNull();
        _reportResponse.Data.Should().Contain(".pdf");
    }

    private async Task<int> CreateTestAnalysisAsync(AnalysisStatus status)
    {
        using var scope = Factory.Services.CreateScope();
        var dbContext = scope.ServiceProvider.GetRequiredService<PadelDbContext>();

        var video = new VideoEntity
        {
            Name = "Test Video",
            FilePath = "/path/test.mp4",
            FileSizeBytes = 1024,
            FileExtension = ".mp4",
            UploadedAt = DateTime.UtcNow,
            Status = status == AnalysisStatus.Completed ? nameof(VideoStatus.Completed) : nameof(VideoStatus.Processing)
        };
        dbContext.Videos.Add(video);
        await dbContext.SaveChangesAsync();

        var analysis = new AnalysisEntity
        {
            VideoId = video.Id,
            StartedAt = DateTime.UtcNow,
            Status = status.ToString()
        };

        if (status == AnalysisStatus.Completed)
        {
            analysis.CompletedAt = DateTime.UtcNow;
            analysis.TotalFrames = 1000;
            analysis.PlayersDetected = 4;
            analysis.DetectionRatePercent = 85.0;
            analysis.AvgDetectionsPerFrame = 3.8;
            analysis.ProcessingTimeSeconds = 60.0;
            analysis.ModelUsed = "yolov8m.pt";
        }

        dbContext.Analyses.Add(analysis);
        await dbContext.SaveChangesAsync();

        return analysis.Id;
    }
}
