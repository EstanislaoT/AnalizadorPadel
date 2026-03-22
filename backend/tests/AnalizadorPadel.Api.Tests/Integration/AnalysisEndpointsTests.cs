using System.Net;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using AnalizadorPadel.Api.Data;
using AnalizadorPadel.Api.Models.DTOs;
using AnalizadorPadel.Api.Models.Entities;
using AnalizadorPadel.Api.Tests.Infrastructure;
using Microsoft.EntityFrameworkCore;

namespace AnalizadorPadel.Api.Tests.Integration;

public class AnalysisEndpointsTests : IntegrationTestBase
{
    public AnalysisEndpointsTests(CustomWebApplicationFactory factory) : base(factory)
    {
    }

    #region POST /api/videos/{id}/analyse Tests

    [Fact]
    public async Task StartAnalysis_WithExistingVideo_ShouldReturn202Accepted()
    {
        // Arrange
        var videoId = await CreateTestVideoInDbAsync("test.mp4");

        // Act
        var response = await Client.PostAsync($"/api/videos/{videoId}/analyse", null);

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.Accepted);

        var result = await response.Content.ReadFromJsonAsync<ApiResponse<AnalysisDto>>();
        result.Should().NotBeNull();
        result!.Success.Should().BeTrue();
        result.Data.Should().NotBeNull();
        result.Data!.VideoId.Should().Be(videoId);
        result.Data.Status.Should().Be(AnalysisStatus.Running);

        // Check Location header
        response.Headers.Location.Should().NotBeNull();
        response.Headers.Location!.ToString().Should().Contain($"/api/analyses/{result.Data.Id}");
    }

    [Fact]
    public async Task StartAnalysis_WithNonExistingVideo_ShouldReturn404()
    {
        // Act
        var response = await Client.PostAsync("/api/videos/999/analyse", null);

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.NotFound);

        var result = await response.Content.ReadFromJsonAsync<ApiResponse<object>>();
        result.Should().NotBeNull();
        result!.Success.Should().BeFalse();
        result.Message.Should().Contain("no encontrado");
    }

    [Fact]
    public async Task StartAnalysis_ShouldUpdateVideoStatusToProcessing()
    {
        // Arrange
        var videoId = await CreateTestVideoInDbAsync("test.mp4");

        // Act
        var response = await Client.PostAsync($"/api/videos/{videoId}/analyse", null);
        var analysisResult = await response.Content.ReadFromJsonAsync<ApiResponse<AnalysisDto>>();

        // Assert - Check video status was updated
        var videoResponse = await Client.GetAsync($"/api/videos/{videoId}");
        var videoResult = await videoResponse.Content.ReadFromJsonAsync<ApiResponse<VideoDto>>();

        videoResult.Should().NotBeNull();
        videoResult!.Data.Should().NotBeNull();
        videoResult.Data!.Status.Should().Be(VideoStatus.Processing);
        videoResult.Data.AnalysisId.Should().Be(analysisResult!.Data!.Id);
    }

    #endregion

    #region GET /api/analyses/{id} Tests

    [Fact]
    public async Task GetAnalysis_WithExistingAnalysis_ShouldReturnAnalysis()
    {
        // Arrange
        var analysisId = await CreateTestAnalysisInDbAsync(VideoStatus.Completed);

        // Act
        var response = await Client.GetAsync($"/api/analyses/{analysisId}");

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.OK);

        var result = await response.Content.ReadFromJsonAsync<ApiResponse<AnalysisDto>>();
        result.Should().NotBeNull();
        result!.Success.Should().BeTrue();
        result.Data.Should().NotBeNull();
        result.Data!.Id.Should().Be(analysisId);
    }

    [Fact]
    public async Task GetAnalysis_WithNonExistingAnalysis_ShouldReturn404()
    {
        // Act
        var response = await Client.GetAsync("/api/analyses/999");

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.NotFound);

        var result = await response.Content.ReadFromJsonAsync<ApiResponse<object>>();
        result.Should().NotBeNull();
        result!.Success.Should().BeFalse();
    }

    #endregion

    #region GET /api/analyses/{id}/stats Tests

    [Fact]
    public async Task GetAnalysisStats_WithCompletedAnalysis_ShouldReturnStats()
    {
        // Arrange
        var analysisId = await CreateTestAnalysisInDbAsync(VideoStatus.Completed, new AnalysisEntity
        {
            TotalFrames = 1000,
            FramesWith4Players = 850,
            DetectionRatePercent = 85.0,
            AvgDetectionsPerFrame = 3.9,
            PlayersDetected = 4,
            ProcessingTimeSeconds = 30.5,
            ModelUsed = "yolov8m.pt"
        });

        // Act
        var response = await Client.GetAsync($"/api/analyses/{analysisId}/stats");

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.OK);

        var result = await response.Content.ReadFromJsonAsync<ApiResponse<AnalysisStats>>();
        result.Should().NotBeNull();
        result!.Success.Should().BeTrue();
        result.Data.Should().NotBeNull();
        result.Data!.TotalFrames.Should().Be(1000);
        result.Data.DetectionRatePercent.Should().Be(85.0);
        result.Data.ModelUsed.Should().Be("yolov8m.pt");
    }

    [Fact]
    public async Task GetAnalysisStats_WithRunningAnalysis_ShouldReturn404()
    {
        // Arrange
        var analysisId = await CreateTestAnalysisInDbAsync(VideoStatus.Processing);

        // Act
        var response = await Client.GetAsync($"/api/analyses/{analysisId}/stats");

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.NotFound);
    }

    #endregion

    #region GET /api/analyses/{id}/heatmap Tests

    [Fact]
    public async Task GetAnalysisHeatmap_WithCompletedAnalysis_ShouldReturnHeatmapData()
    {
        // Arrange
        var analysisId = await CreateTestAnalysisInDbAsync(VideoStatus.Completed);

        // Act
        var response = await Client.GetAsync($"/api/analyses/{analysisId}/heatmap");

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.OK);

        var result = await response.Content.ReadFromJsonAsync<ApiResponse<HeatmapData>>();
        result.Should().NotBeNull();
        result!.Success.Should().BeTrue();
        result.Data.Should().NotBeNull();
        result.Data!.Points.Should().HaveCount(100);
        result.Data.CourtDimensions.Should().Be("23.77m x 10.97m");
    }

    [Fact]
    public async Task GetAnalysisHeatmap_WithNonCompletedAnalysis_ShouldReturn404()
    {
        // Arrange
        var analysisId = await CreateTestAnalysisInDbAsync(VideoStatus.Failed);

        // Act
        var response = await Client.GetAsync($"/api/analyses/{analysisId}/heatmap");

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.NotFound);
    }

    #endregion

    #region GET /api/analyses/{id}/report Tests

    [Fact]
    public async Task GetAnalysisReport_WithCompletedAnalysis_ShouldReturnReportPath()
    {
        // Arrange
        var analysisId = await CreateTestAnalysisInDbAsync(VideoStatus.Completed);

        // Act
        var response = await Client.GetAsync($"/api/analyses/{analysisId}/report");

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.OK);

        var result = await response.Content.ReadFromJsonAsync<ApiResponse<string>>();
        result.Should().NotBeNull();
        result!.Success.Should().BeTrue();
        result.Data.Should().Contain($"/api/analyses/{analysisId}/report-placeholder");
    }

    [Fact]
    public async Task GetAnalysisReport_WithNonCompletedAnalysis_ShouldReturn404()
    {
        // Arrange
        var analysisId = await CreateTestAnalysisInDbAsync(VideoStatus.Processing);

        // Act
        var response = await Client.GetAsync($"/api/analyses/{analysisId}/report");

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.NotFound);
    }

    #endregion

    #region GET /api/dashboard/stats Tests

    [Fact]
    public async Task GetDashboardStats_WithNoData_ShouldReturnZeroStats()
    {
        // Act
        var response = await Client.GetAsync("/api/dashboard/stats");

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.OK);

        var result = await response.Content.ReadFromJsonAsync<ApiResponse<DashboardStats>>();
        result.Should().NotBeNull();
        result!.Success.Should().BeTrue();
        result.Data.Should().NotBeNull();
        result.Data!.TotalVideos.Should().Be(0);
        result.Data.TotalAnalyses.Should().Be(0);
    }

    [Fact]
    public async Task GetDashboardStats_WithData_ShouldReturnCorrectStats()
    {
        // Arrange - Create test data
        await CreateTestVideoInDbAsync("video1.mp4");
        await CreateTestVideoInDbAsync("video2.mp4");
        await CreateTestAnalysisInDbAsync(VideoStatus.Completed, new AnalysisEntity
        {
            DetectionRatePercent = 90.0
        });

        // Act
        var response = await Client.GetAsync("/api/dashboard/stats");

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.OK);

        var result = await response.Content.ReadFromJsonAsync<ApiResponse<DashboardStats>>();
        result.Should().NotBeNull();
        result!.Success.Should().BeTrue();
        result.Data!.TotalVideos.Should().Be(2);
        result.Data.TotalAnalyses.Should().Be(1);
        result.Data.RecentVideos.Should().HaveCount(2);
    }

    #endregion

    #region GET /api/health Tests

    [Fact]
    public async Task HealthCheck_ShouldReturnHealthyStatus()
    {
        // Act
        var response = await Client.GetAsync("/api/health");

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.OK);

        var result = await response.Content.ReadFromJsonAsync<ApiResponse<object>>();
        result.Should().NotBeNull();
        result!.Success.Should().BeTrue();
        result.Message.Should().Contain("healthy");
    }

    #endregion

    #region Helper Methods

    private async Task<int> CreateTestVideoInDbAsync(string fileName, VideoStatus status = VideoStatus.Uploaded)
    {
        using var scope = Factory.Services.CreateScope();
        var dbContext = scope.ServiceProvider.GetRequiredService<PadelDbContext>();

        var video = new VideoEntity
        {
            Name = Path.GetFileNameWithoutExtension(fileName),
            FilePath = $"/tmp/test/{fileName}",
            FileSizeBytes = 1024,
            FileExtension = Path.GetExtension(fileName),
            UploadedAt = DateTime.UtcNow,
            Status = status.ToString()
        };

        dbContext.Videos.Add(video);
        await dbContext.SaveChangesAsync();

        return video.Id;
    }

    private async Task<int> CreateTestAnalysisInDbAsync(VideoStatus videoStatus, AnalysisEntity? analysisData = null)
    {
        using var scope = Factory.Services.CreateScope();
        var dbContext = scope.ServiceProvider.GetRequiredService<PadelDbContext>();

        var video = new VideoEntity
        {
            Name = "Test Video",
            FilePath = "/tmp/test/video.mp4",
            FileSizeBytes = 1024,
            FileExtension = ".mp4",
            UploadedAt = DateTime.UtcNow,
            Status = videoStatus.ToString()
        };
        dbContext.Videos.Add(video);
        await dbContext.SaveChangesAsync();

        var analysis = analysisData ?? new AnalysisEntity();
        analysis.VideoId = video.Id;
        analysis.StartedAt = DateTime.UtcNow;

        if (videoStatus == VideoStatus.Completed)
        {
            analysis.Status = nameof(AnalysisStatus.Completed);
            analysis.CompletedAt = DateTime.UtcNow;
            analysis.TotalFrames ??= 1000;
            analysis.PlayersDetected ??= 4;
            analysis.DetectionRatePercent ??= 85.0;
            analysis.ModelUsed ??= "yolov8m.pt";
        }
        else if (videoStatus == VideoStatus.Failed)
        {
            analysis.Status = nameof(AnalysisStatus.Failed);
            analysis.CompletedAt = DateTime.UtcNow;
            analysis.ErrorMessage = "Test error";
        }
        else
        {
            analysis.Status = nameof(AnalysisStatus.Running);
        }

        dbContext.Analyses.Add(analysis);
        await dbContext.SaveChangesAsync();

        // Update video with analysis ID
        video.AnalysisId = analysis.Id;
        video.Status = analysis.Status;
        await dbContext.SaveChangesAsync();

        return analysis.Id;
    }

    #endregion
}
