using AnalizadorPadel.Api.Data;
using AnalizadorPadel.Api.Models.DTOs;
using AnalizadorPadel.Api.Models.Entities;
using AnalizadorPadel.Api.Services;
using AnalizadorPadel.Api.Tests.Infrastructure;
using Microsoft.AspNetCore.Hosting;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Configuration;
using Moq;
using Microsoft.EntityFrameworkCore.Storage;

namespace AnalizadorPadel.Api.Tests.Unit.Services;

public class AnalysisServiceTests : TestBase
{
    private readonly Mock<IWebHostEnvironment> _envMock;
    private readonly IConfiguration _configuration;
    private readonly Mock<ILogger<AnalysisService>> _loggerMock;
    private readonly Mock<IVideoService> _videoServiceMock;
    private readonly DbContextOptions<PadelDbContext> _dbOptions;
    private readonly InMemoryDatabaseRoot _databaseRoot;
    private readonly string _databaseName;

    public AnalysisServiceTests()
    {
        _envMock = new Mock<IWebHostEnvironment>();
        _envMock.Setup(e => e.ContentRootPath).Returns(Path.GetTempPath());
        _configuration = new ConfigurationBuilder().AddInMemoryCollection().Build();
        _loggerMock = new Mock<ILogger<AnalysisService>>();
        _videoServiceMock = new Mock<IVideoService>();
        _databaseRoot = new InMemoryDatabaseRoot();
        _databaseName = $"TestDb_{Guid.NewGuid()}";

        _dbOptions = new DbContextOptionsBuilder<PadelDbContext>()
            .UseInMemoryDatabase(_databaseName, _databaseRoot)
            .Options;
    }

    private async Task<(AnalysisService service, PadelDbContext dbContext)> CreateServiceAsync()
    {
        var dbContext = new PadelDbContext(_dbOptions);
        await dbContext.Database.EnsureCreatedAsync();

        var factory = new Mock<IDbContextFactory<PadelDbContext>>();
        factory.Setup(f => f.CreateDbContextAsync(It.IsAny<CancellationToken>()))
               .ReturnsAsync(() => new PadelDbContext(_dbOptions));

        var service = new AnalysisService(factory.Object, _envMock.Object, _configuration, _videoServiceMock.Object, _loggerMock.Object);
        return (service, dbContext);
    }

    #region StartAnalysisAsync Tests

    [Fact]
    public async Task StartAnalysisAsync_WithExistingVideo_ShouldCreateAnalysisAndReturnDto()
    {
        // Arrange
        var (service, dbContext) = await CreateServiceAsync();
        var videoId = 1;

        _videoServiceMock.Setup(v => v.GetByIdAsync(videoId))
            .ReturnsAsync(new VideoDto(
                Id: videoId,
                Name: "Test Video",
                Description: null,
                FilePath: "/path/test.mp4",
                UploadedAt: DateTime.UtcNow,
                Status: VideoStatus.Uploaded,
                AnalysisId: null));

        // Act
        var result = await service.StartAnalysisAsync(videoId, null);

        // Assert
        result.Should().NotBeNull();
        result.VideoId.Should().Be(videoId);
        result.Status.Should().Be(AnalysisStatus.Running);
        result.Id.Should().BeGreaterThan(0);
        result.StartedAt.Should().BeCloseTo(DateTime.UtcNow, TimeSpan.FromSeconds(1));

        _videoServiceMock.Verify(v => v.UpdateStatusAsync(videoId, VideoStatus.Processing, It.IsAny<int>()), Times.AtLeastOnce);
    }

    [Fact]
    public async Task StartAnalysisAsync_WithNonExistingVideo_ShouldThrowArgumentException()
    {
        // Arrange
        var (service, dbContext) = await CreateServiceAsync();
        var videoId = 999;

        _videoServiceMock.Setup(v => v.GetByIdAsync(videoId))
            .ReturnsAsync((VideoDto?)null);

        // Act & Assert
        var exception = await Assert.ThrowsAsync<ArgumentException>(() => service.StartAnalysisAsync(videoId, null));
        exception.Message.Should().Contain($"Video {videoId} not found");
    }

    #endregion

    #region GetByIdAsync Tests

    [Fact]
    public async Task GetByIdAsync_WithExistingAnalysis_ShouldReturnAnalysis()
    {
        // Arrange
        var (service, dbContext) = await CreateServiceAsync();
        var analysis = new AnalysisEntity
        {
            VideoId = 1,
            StartedAt = DateTime.UtcNow,
            Status = nameof(AnalysisStatus.Completed),
            CompletedAt = DateTime.UtcNow,
            TotalFrames = 1000,
            PlayersDetected = 4,
            DetectionRatePercent = 85.5,
            ModelUsed = "yolov8m.pt"
        };
        dbContext.Analyses.Add(analysis);
        await dbContext.SaveChangesAsync();

        // Act
        var result = await service.GetByIdAsync(analysis.Id);

        // Assert
        result.Should().NotBeNull();
        result!.Id.Should().Be(analysis.Id);
        result.Status.Should().Be(AnalysisStatus.Completed);
        result.Result.Should().NotBeNull();
        result.Result!.TotalFrames.Should().Be(1000);
        result.Result.DetectionRatePercent.Should().Be(85.5);
    }

    [Fact]
    public async Task GetByIdAsync_WithPendingAnalysis_ShouldReturnNullResult()
    {
        // Arrange
        var (service, dbContext) = await CreateServiceAsync();
        var analysis = new AnalysisEntity
        {
            VideoId = 1,
            StartedAt = DateTime.UtcNow,
            Status = nameof(AnalysisStatus.Running)
        };
        dbContext.Analyses.Add(analysis);
        await dbContext.SaveChangesAsync();

        // Act
        var result = await service.GetByIdAsync(analysis.Id);

        // Assert
        result.Should().NotBeNull();
        result!.Status.Should().Be(AnalysisStatus.Running);
        result.Result.Should().BeNull();
    }

    [Fact]
    public async Task GetByIdAsync_WithNonExistingAnalysis_ShouldReturnNull()
    {
        // Arrange
        var (service, dbContext) = await CreateServiceAsync();

        // Act
        var result = await service.GetByIdAsync(999);

        // Assert
        result.Should().BeNull();
    }

    #endregion

    #region GetStatsAsync Tests

    [Fact]
    public async Task GetStatsAsync_WithCompletedAnalysis_ShouldReturnStats()
    {
        // Arrange
        var (service, dbContext) = await CreateServiceAsync();
        var analysis = new AnalysisEntity
        {
            VideoId = 1,
            StartedAt = DateTime.UtcNow,
            Status = nameof(AnalysisStatus.Completed),
            TotalFrames = 1500,
            FramesWith4Players = 1200,
            DetectionRatePercent = 80.0,
            AvgDetectionsPerFrame = 3.8,
            PlayersDetected = 4,
            ProcessingTimeSeconds = 45.5,
            ModelUsed = "yolov8m.pt"
        };
        dbContext.Analyses.Add(analysis);
        await dbContext.SaveChangesAsync();

        // Act
        var result = await service.GetStatsAsync(analysis.Id);

        // Assert
        result.Should().NotBeNull();
        result!.TotalFrames.Should().Be(1500);
        result.FramesWith4Players.Should().Be(1200);
        result.DetectionRatePercent.Should().Be(80.0);
        result.AvgDetectionsPerFrame.Should().Be(3.8);
        result.PlayersDetected.Should().Be(4);
        result.ProcessingTimeSeconds.Should().Be(45.5);
        result.ModelUsed.Should().Be("yolov8m.pt");
    }

    [Fact]
    public async Task GetStatsAsync_WithNonCompletedAnalysis_ShouldReturnNull()
    {
        // Arrange
        var (service, dbContext) = await CreateServiceAsync();
        var analysis = new AnalysisEntity
        {
            VideoId = 1,
            StartedAt = DateTime.UtcNow,
            Status = nameof(AnalysisStatus.Running)
        };
        dbContext.Analyses.Add(analysis);
        await dbContext.SaveChangesAsync();

        // Act
        var result = await service.GetStatsAsync(analysis.Id);

        // Assert
        result.Should().BeNull();
    }

    [Fact]
    public async Task GetStatsAsync_WithNonExistingAnalysis_ShouldReturnNull()
    {
        // Arrange
        var (service, dbContext) = await CreateServiceAsync();

        // Act
        var result = await service.GetStatsAsync(999);

        // Assert
        result.Should().BeNull();
    }

    #endregion

    #region GetHeatmapAsync Tests

    [Fact]
    public async Task GetHeatmapAsync_WithCompletedAnalysis_ShouldReturnHeatmapData()
    {
        // Arrange
        var (service, dbContext) = await CreateServiceAsync();
        var analysis = new AnalysisEntity
        {
            VideoId = 1,
            StartedAt = DateTime.UtcNow,
            Status = nameof(AnalysisStatus.Completed)
        };
        dbContext.Analyses.Add(analysis);
        await dbContext.SaveChangesAsync();

        // Act
        var result = await service.GetHeatmapAsync(analysis.Id);

        // Assert
        result.Should().NotBeNull();
        result!.Points.Should().HaveCount(100);
        result.CourtDimensions.Should().Be("23.77m x 10.97m");

        // Verify points have valid coordinates
        result.Points.All(p => p.X >= 0 && p.X <= 23.77).Should().BeTrue();
        result.Points.All(p => p.Y >= 0 && p.Y <= 10.97).Should().BeTrue();
        result.Points.All(p => p.Intensity >= 1 && p.Intensity <= 10).Should().BeTrue();
    }

    [Fact]
    public async Task GetHeatmapAsync_WithNonCompletedAnalysis_ShouldReturnNull()
    {
        // Arrange
        var (service, dbContext) = await CreateServiceAsync();
        var analysis = new AnalysisEntity
        {
            VideoId = 1,
            StartedAt = DateTime.UtcNow,
            Status = nameof(AnalysisStatus.Running)
        };
        dbContext.Analyses.Add(analysis);
        await dbContext.SaveChangesAsync();

        // Act
        var result = await service.GetHeatmapAsync(analysis.Id);

        // Assert
        result.Should().BeNull();
    }

    [Fact]
    public async Task GetHeatmapAsync_WithDifferentAnalysisIds_ShouldGenerateDifferentData()
    {
        // Arrange
        var (service, dbContext) = await CreateServiceAsync();

        var analysis1 = new AnalysisEntity
        {
            VideoId = 1,
            StartedAt = DateTime.UtcNow,
            Status = nameof(AnalysisStatus.Completed)
        };
        var analysis2 = new AnalysisEntity
        {
            VideoId = 2,
            StartedAt = DateTime.UtcNow,
            Status = nameof(AnalysisStatus.Completed)
        };
        dbContext.Analyses.AddRange(analysis1, analysis2);
        await dbContext.SaveChangesAsync();

        // Act
        var result1 = await service.GetHeatmapAsync(analysis1.Id);
        var result2 = await service.GetHeatmapAsync(analysis2.Id);

        // Assert
        result1.Should().NotBeNull();
        result2.Should().NotBeNull();

        // Points should be different (different seeds)
        result1!.Points[0].X.Should().NotBe(result2!.Points[0].X);
    }

    #endregion

    #region GetReportAsync Tests

    [Fact]
    public async Task GetReportAsync_WithCompletedAnalysis_ShouldReturnReportPath()
    {
        // Arrange
        var (service, dbContext) = await CreateServiceAsync();
        var analysis = new AnalysisEntity
        {
            VideoId = 1,
            StartedAt = DateTime.UtcNow,
            Status = nameof(AnalysisStatus.Completed)
        };
        dbContext.Analyses.Add(analysis);
        await dbContext.SaveChangesAsync();

        // Act
        var result = await service.GetReportAsync(analysis.Id);

        // Assert
        result.Should().NotBeNull();
        result.Should().Contain($"/api/analyses/{analysis.Id}/report-placeholder");
    }

    [Fact]
    public async Task GetReportAsync_WithNonCompletedAnalysis_ShouldReturnNull()
    {
        // Arrange
        var (service, dbContext) = await CreateServiceAsync();
        var analysis = new AnalysisEntity
        {
            VideoId = 1,
            StartedAt = DateTime.UtcNow,
            Status = nameof(AnalysisStatus.Running)
        };
        dbContext.Analyses.Add(analysis);
        await dbContext.SaveChangesAsync();

        // Act
        var result = await service.GetReportAsync(analysis.Id);

        // Assert
        result.Should().BeNull();
    }

    #endregion
}
