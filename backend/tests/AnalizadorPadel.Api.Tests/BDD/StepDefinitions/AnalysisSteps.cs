using System.Net;
using System.Net.Http.Json;
using AnalizadorPadel.Api.Data;
using AnalizadorPadel.Api.Models.DTOs;
using AnalizadorPadel.Api.Models.Entities;
using AnalizadorPadel.Api.Tests.Infrastructure;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using TechTalk.SpecFlow;

namespace AnalizadorPadel.Api.Tests.BDD.StepDefinitions;

[Binding]
public class AnalysisSteps : IntegrationTestBase
{
    private const string ResponseKey = "HttpResponse";
    private const string StatsResponseKey = "AnalysisStatsResponse";
    private const string HeatmapResponseKey = "HeatmapResponse";
    private const string DashboardResponseKey = "DashboardResponse";
    private const string ReportResponseKey = "ReportResponse";
    private const string TotalAnalysesKey = "TotalAnalyses";

    private readonly ScenarioContext _scenarioContext;

    private HttpResponseMessage? Response
    {
        get => _scenarioContext.TryGetValue<HttpResponseMessage>(ResponseKey, out var response) ? response : null;
        set => _scenarioContext[ResponseKey] = value!;
    }

    private ApiResponse<AnalysisStats>? StatsResponse
    {
        get => _scenarioContext.TryGetValue<ApiResponse<AnalysisStats>>(StatsResponseKey, out var response) ? response : null;
        set => _scenarioContext[StatsResponseKey] = value!;
    }

    private ApiResponse<HeatmapData>? HeatmapResponse
    {
        get => _scenarioContext.TryGetValue<ApiResponse<HeatmapData>>(HeatmapResponseKey, out var response) ? response : null;
        set => _scenarioContext[HeatmapResponseKey] = value!;
    }

    private ApiResponse<DashboardStats>? DashboardResponse
    {
        get => _scenarioContext.TryGetValue<ApiResponse<DashboardStats>>(DashboardResponseKey, out var response) ? response : null;
        set => _scenarioContext[DashboardResponseKey] = value!;
    }

    private ApiResponse<string>? ReportResponse
    {
        get => _scenarioContext.TryGetValue<ApiResponse<string>>(ReportResponseKey, out var response) ? response : null;
        set => _scenarioContext[ReportResponseKey] = value!;
    }

    public AnalysisSteps(CustomWebApplicationFactory factory, ScenarioContext scenarioContext) : base(factory)
    {
        _scenarioContext = scenarioContext;
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

        var existingVideos = await dbContext.Videos.CountAsync();
        for (int i = existingVideos; i < count; i++)
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

        var existingCompleted = await dbContext.Analyses.CountAsync(a => a.Status == nameof(AnalysisStatus.Completed));
        var analysesToCreate = count - existingCompleted;
        if (analysesToCreate <= 0)
        {
            return;
        }

        var candidateVideos = await dbContext.Videos
            .Where(v => !dbContext.Analyses.Any(a => a.VideoId == v.Id))
            .OrderBy(v => v.UploadedAt)
            .ToListAsync();

        for (int i = 0; i < analysesToCreate; i++)
        {
            VideoEntity video;
            if (i < candidateVideos.Count)
            {
                video = candidateVideos[i];
                video.Status = nameof(VideoStatus.Completed);
            }
            else
            {
                video = new VideoEntity
                {
                    Name = $"Video {existingCompleted + i + 1}",
                    FilePath = $"/path/video{existingCompleted + i + 1}.mp4",
                    FileSizeBytes = 1024,
                    FileExtension = ".mp4",
                    UploadedAt = DateTime.UtcNow.AddMinutes(-(existingCompleted + i)),
                    Status = nameof(VideoStatus.Completed)
                };
                dbContext.Videos.Add(video);
                await dbContext.SaveChangesAsync();
            }

            var analysis = new AnalysisEntity
            {
                VideoId = video.Id,
                Status = nameof(AnalysisStatus.Completed),
                StartedAt = DateTime.UtcNow.AddHours(-(existingCompleted + i)),
                CompletedAt = DateTime.UtcNow.AddHours(-(existingCompleted + i)).AddMinutes(10),
                TotalFrames = 1000,
                PlayersDetected = 4,
                DetectionRatePercent = 85.0,
                ModelUsed = "yolov8m.pt"
            };
            dbContext.Analyses.Add(analysis);
        }

        await dbContext.SaveChangesAsync();
    }

    [Given("existen (\\d+) análisis fallido")]
    public async Task GivenExistenAnalisisFallido(int count)
    {
        using var scope = Factory.Services.CreateScope();
        var dbContext = scope.ServiceProvider.GetRequiredService<PadelDbContext>();

        var existingFailed = await dbContext.Analyses.CountAsync(a => a.Status == nameof(AnalysisStatus.Failed));
        var analysesToCreate = count - existingFailed;
        if (analysesToCreate <= 0)
        {
            return;
        }

        var candidateVideos = await dbContext.Videos
            .Where(v => !dbContext.Analyses.Any(a => a.VideoId == v.Id))
            .OrderBy(v => v.UploadedAt)
            .ToListAsync();

        for (int i = 0; i < analysesToCreate; i++)
        {
            VideoEntity video;
            if (i < candidateVideos.Count)
            {
                video = candidateVideos[i];
                video.Status = nameof(VideoStatus.Failed);
            }
            else
            {
                video = new VideoEntity
                {
                    Name = $"Failed Video {existingFailed + i + 1}",
                    FilePath = $"/path/failed{existingFailed + i + 1}.mp4",
                    FileSizeBytes = 1024,
                    FileExtension = ".mp4",
                    UploadedAt = DateTime.UtcNow.AddMinutes(-(existingFailed + i)),
                    Status = nameof(VideoStatus.Failed)
                };
                dbContext.Videos.Add(video);
                await dbContext.SaveChangesAsync();
            }

            var analysis = new AnalysisEntity
            {
                VideoId = video.Id,
                Status = nameof(AnalysisStatus.Failed),
                StartedAt = DateTime.UtcNow.AddHours(-(existingFailed + i)),
                CompletedAt = DateTime.UtcNow.AddHours(-(existingFailed + i)).AddMinutes(5),
                ErrorMessage = "Processing timeout"
            };
            dbContext.Analyses.Add(analysis);
        }

        await dbContext.SaveChangesAsync();
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
        _scenarioContext[TotalAnalysesKey] = count;
        await Factory.ResetStateAsync();

        using var scope = Factory.Services.CreateScope();
        var dbContext = scope.ServiceProvider.GetRequiredService<PadelDbContext>();

        for (int i = 0; i < count; i++)
        {
            var video = new VideoEntity
            {
                Name = $"Scenario Video {i + 1}",
                FilePath = $"/path/scenario-video{i + 1}.mp4",
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
                ErrorMessage = "Scenario seeded failure"
            };
            dbContext.Analyses.Add(analysis);
            await dbContext.SaveChangesAsync();
        }
    }

    [Given(@"(\d+) análisis están completados")]
    public async Task GivenAnalisisEstanCompletados(int count)
    {
        var total = _scenarioContext.TryGetValue<int>(TotalAnalysesKey, out var totalAnalyses) ? totalAnalyses : 0;
        count.Should().BeLessThanOrEqualTo(total);

        using var scope = Factory.Services.CreateScope();
        var dbContext = scope.ServiceProvider.GetRequiredService<PadelDbContext>();
        var analyses = await dbContext.Analyses
            .OrderBy(a => a.Id)
            .Take(count)
            .ToListAsync();

        foreach (var analysis in analyses)
        {
            analysis.Status = nameof(AnalysisStatus.Completed);
            analysis.CompletedAt = DateTime.UtcNow;
            analysis.TotalFrames = 1000;
            analysis.PlayersDetected = 4;
            analysis.DetectionRatePercent = 85.0;
            analysis.ModelUsed = "yolov8m.pt";

            var video = await dbContext.Videos.FindAsync(analysis.VideoId);
            if (video != null)
            {
                video.Status = nameof(VideoStatus.Completed);
                video.AnalysisId = analysis.Id;
            }
        }

        await dbContext.SaveChangesAsync();
    }

    [When("el usuario solicita las estadísticas del análisis (\\d+)")]
    public async Task WhenElUsuarioSolicitaLasEstadisticasDelAnalisis(int id)
    {
        Response = await Client.GetAsync($"/api/analyses/{id}/stats");
        if (Response.IsSuccessStatusCode)
        {
            StatsResponse = await Response.Content.ReadFromJsonAsync<ApiResponse<AnalysisStats>>();
        }
    }

    [When("el usuario solicita los datos del heatmap del análisis (\\d+)")]
    public async Task WhenElUsuarioSolicitaLosDatosDelHeatmapDelAnalisis(int id)
    {
        Response = await Client.GetAsync($"/api/analyses/{id}/heatmap");
        if (Response.IsSuccessStatusCode)
        {
            HeatmapResponse = await Response.Content.ReadFromJsonAsync<ApiResponse<HeatmapData>>();
        }
    }

    [When("el usuario solicita las estadísticas del dashboard")]
    public async Task WhenElUsuarioSolicitaLasEstadisticasDelDashboard()
    {
        Response = await Client.GetAsync("/api/dashboard/stats");
        if (Response.IsSuccessStatusCode)
        {
            DashboardResponse = await Response.Content.ReadFromJsonAsync<ApiResponse<DashboardStats>>();
        }
    }

    [When("el usuario solicita el reporte del análisis (\\d+)")]
    public async Task WhenElUsuarioSolicitaElReporteDelAnalisis(int id)
    {
        Response = await Client.GetAsync($"/api/analyses/{id}/report");
        if (Response.IsSuccessStatusCode)
        {
            ReportResponse = await Response.Content.ReadFromJsonAsync<ApiResponse<string>>();
        }
    }

    [Then("las estadísticas incluyen frames totales")]
    public void ThenLasEstadisticasIncluyenFramesTotales()
    {
        StatsResponse.Should().NotBeNull();
        StatsResponse!.Data.Should().NotBeNull();
        StatsResponse.Data!.TotalFrames.Should().BeGreaterThan(0);
    }

    [Then("las estadísticas incluyen tasa de detección")]
    public void ThenLasEstadisticasIncluyenTasaDeDeteccion()
    {
        StatsResponse.Should().NotBeNull();
        StatsResponse!.Data.Should().NotBeNull();
        StatsResponse.Data!.DetectionRatePercent.Should().BeGreaterThanOrEqualTo(0);
    }

    [Then("las estadísticas incluyen tiempo de procesamiento")]
    public void ThenLasEstadisticasIncluyenTiempoDeProcesamiento()
    {
        StatsResponse.Should().NotBeNull();
        StatsResponse!.Data.Should().NotBeNull();
        StatsResponse.Data!.ProcessingTimeSeconds.Should().BeGreaterThanOrEqualTo(0);
    }

    [Then("las estadísticas incluyen modelo utilizado")]
    public void ThenLasEstadisticasIncluyenModeloUtilizado()
    {
        StatsResponse.Should().NotBeNull();
        StatsResponse!.Data.Should().NotBeNull();
        StatsResponse.Data!.ModelUsed.Should().NotBeNullOrEmpty();
    }

    [Then("el mensaje indica \"(.*)\"")]
    public async Task ThenElMensajeIndica(string expectedMessage)
    {
        var content = await Response!.Content.ReadFromJsonAsync<ApiResponse<object>>();
        content.Should().NotBeNull();
        content!.Message.Should().Contain(expectedMessage);
    }

    [Then("el heatmap contiene (\\d+) puntos")]
    public void ThenElHeatmapContienePuntos(int expectedCount)
    {
        HeatmapResponse.Should().NotBeNull();
        HeatmapResponse!.Data.Should().NotBeNull();
        HeatmapResponse.Data!.Points.Should().HaveCount(expectedCount);
    }

    [Then("cada punto tiene coordenadas X, Y e intensidad")]
    public void ThenCadaPuntoTieneCoordenadasXYEIntensidad()
    {
        HeatmapResponse.Should().NotBeNull();
        HeatmapResponse!.Data.Should().NotBeNull();

        foreach (var point in HeatmapResponse.Data!.Points)
        {
            point.X.Should().BeGreaterThanOrEqualTo(0);
            point.Y.Should().BeGreaterThanOrEqualTo(0);
            point.Intensity.Should().BeGreaterThanOrEqualTo(1);
        }
    }

    [Then("las dimensiones de la cancha son \"(.*)\"")]
    public void ThenLasDimensionesDeLaCanchaSon(string expectedDimensions)
    {
        HeatmapResponse.Should().NotBeNull();
        HeatmapResponse!.Data.Should().NotBeNull();
        HeatmapResponse.Data!.CourtDimensions.Should().Be(expectedDimensions);
    }

    [Then("el dashboard muestra (\\d+) videos totales")]
    public void ThenElDashboardMuestraVideosTotales(int expectedCount)
    {
        DashboardResponse.Should().NotBeNull();
        DashboardResponse!.Data.Should().NotBeNull();
        DashboardResponse.Data!.TotalVideos.Should().Be(expectedCount);
    }

    [Then("el dashboard muestra (\\d+) análisis totales")]
    public void ThenElDashboardMuestraAnalisisTotales(int expectedCount)
    {
        DashboardResponse.Should().NotBeNull();
        DashboardResponse!.Data.Should().NotBeNull();
        DashboardResponse.Data!.TotalAnalyses.Should().Be(expectedCount);
    }

    [Then("el dashboard muestra (\\d+) análisis completados")]
    public void ThenElDashboardMuestraAnalisisCompletados(int expectedCount)
    {
        DashboardResponse.Should().NotBeNull();
        DashboardResponse!.Data.Should().NotBeNull();
        DashboardResponse.Data!.CompletedAnalyses.Should().Be(expectedCount);
    }

    [Then("el dashboard muestra (\\d+) análisis fallido")]
    public void ThenElDashboardMuestraAnalisisFallido(int expectedCount)
    {
        DashboardResponse.Should().NotBeNull();
        DashboardResponse!.Data.Should().NotBeNull();
        DashboardResponse.Data!.FailedAnalyses.Should().Be(expectedCount);
    }

    [Then("la tasa de éxito es (\\d+)%")]
    public void ThenLaTasaDeExitoEs(int expectedRate)
    {
        DashboardResponse.Should().NotBeNull();
        DashboardResponse!.Data.Should().NotBeNull();
        DashboardResponse.Data!.SuccessRatePercent.Should().BeApproximately(expectedRate, 0.01);
    }

    [Then("el dashboard muestra los (\\d+) videos más recientes")]
    public void ThenElDashboardMuestraLosVideosMasRecientes(int expectedCount)
    {
        DashboardResponse.Should().NotBeNull();
        DashboardResponse!.Data.Should().NotBeNull();
        DashboardResponse.Data!.RecentVideos.Should().HaveCount(expectedCount);
    }

    [Then("los videos están ordenados por fecha descendente")]
    public void ThenLosVideosEstanOrdenadosPorFechaDescendente()
    {
        DashboardResponse.Should().NotBeNull();
        DashboardResponse!.Data.Should().NotBeNull();
        var videos = DashboardResponse.Data!.RecentVideos;

        for (int i = 0; i < videos.Count - 1; i++)
        {
            videos[i].UploadedAt.Should().BeOnOrAfter(videos[i + 1].UploadedAt);
        }
    }

    [Then("el dashboard muestra los (\\d+) análisis más recientes")]
    public void ThenElDashboardMuestraLosAnalisisMasRecientes(int expectedCount)
    {
        DashboardResponse.Should().NotBeNull();
        DashboardResponse!.Data.Should().NotBeNull();
        DashboardResponse.Data!.RecentAnalyses.Should().HaveCountLessThanOrEqualTo(expectedCount);
    }

    [Then("los análisis están ordenados por fecha de inicio descendente")]
    public void ThenLosAnalisisEstanOrdenadosPorFechaDeInicioDescendente()
    {
        DashboardResponse.Should().NotBeNull();
        DashboardResponse!.Data.Should().NotBeNull();
        var analyses = DashboardResponse.Data!.RecentAnalyses;

        for (int i = 0; i < analyses.Count - 1; i++)
        {
            analyses[i].StartedAt.Should().BeOnOrAfter(analyses[i + 1].StartedAt);
        }
    }

    [Then("la respuesta incluye la ruta al archivo PDF")]
    public void ThenLaRespuestaIncluyeLaRutaAlArchivoPDF()
    {
        ReportResponse.Should().NotBeNull();
        ReportResponse!.Data.Should().NotBeNull();
        ReportResponse.Data.Should().Contain(".pdf");
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
