using AnalizadorPadel.Api.Data;
using AnalizadorPadel.Api.Models.DTOs;
using AnalizadorPadel.Api.Models.Entities;
using AnalizadorPadel.Api.Services;
using AnalizadorPadel.Api.Tests.Infrastructure;
using Microsoft.AspNetCore.Hosting;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;
using Moq;

namespace AnalizadorPadel.Api.Tests.Unit.Services;

public class VideoServiceTests : TestBase
{
    private readonly Mock<IWebHostEnvironment> _envMock;
    private readonly Mock<ILogger<VideoService>> _loggerMock;
    private readonly DbContextOptions<PadelDbContext> _dbOptions;

    public VideoServiceTests()
    {
        _envMock = new Mock<IWebHostEnvironment>();
        _envMock.Setup(e => e.ContentRootPath).Returns(Path.GetTempPath());
        _loggerMock = new Mock<ILogger<VideoService>>();

        // Use unique database name for each test to ensure isolation
        _dbOptions = new DbContextOptionsBuilder<PadelDbContext>()
            .UseInMemoryDatabase($"TestDb_{Guid.NewGuid()}")
            .Options;
    }

    private async Task<(VideoService service, PadelDbContext dbContext)> CreateServiceAsync()
    {
        var dbContext = new PadelDbContext(_dbOptions);
        await dbContext.Database.EnsureCreatedAsync();

        var factory = new Mock<IDbContextFactory<PadelDbContext>>();
        factory.Setup(f => f.CreateDbContextAsync(It.IsAny<CancellationToken>()))
               .ReturnsAsync(dbContext);

        var service = new VideoService(factory.Object, _envMock.Object, _loggerMock.Object);
        return (service, dbContext);
    }

    #region CreateVideoAsync Tests

    [Fact]
    public async Task CreateVideoAsync_WithValidFile_ShouldCreateVideoAndReturnDto()
    {
        // Arrange
        var (service, dbContext) = await CreateServiceAsync();
        var fileName = "test_video.mp4";
        var fileSize = 1024L;
        var file = CreateMockFormFile(fileName, fileSize);
        var videoName = "Test Video";
        var description = "Test Description";

        // Act
        var result = await service.CreateVideoAsync(file, videoName, description);

        // Assert
        result.Should().NotBeNull();
        result.Name.Should().Be(videoName);
        result.Description.Should().Be(description);
        result.Status.Should().Be(VideoStatus.Uploaded);
        result.Id.Should().BeGreaterThan(0);

        // Verify database state
        var entity = await dbContext.Videos.FindAsync(result.Id);
        entity.Should().NotBeNull();
        entity!.Name.Should().Be(videoName);
        entity.FileSizeBytes.Should().Be(fileSize);
        entity.Status.Should().Be(nameof(VideoStatus.Uploaded));
    }

    [Fact]
    public async Task CreateVideoAsync_ShouldGenerateUniqueFileNames()
    {
        // Arrange
        var (service, dbContext) = await CreateServiceAsync();
        var file1 = CreateMockFormFile("video1.mp4", 1024);
        var file2 = CreateMockFormFile("video1.mp4", 1024); // Same name

        // Act
        var result1 = await service.CreateVideoAsync(file1, "Video 1");
        var result2 = await service.CreateVideoAsync(file2, "Video 2");

        // Assert
        result1.FilePath.Should().NotBe(result2.FilePath);
        result1.Id.Should().NotBe(result2.Id);
    }

    [Theory]
    [InlineData("test.mp4", ".mp4")]
    [InlineData("test.avi", ".avi")]
    [InlineData("test.mov", ".mov")]
    public async Task CreateVideoAsync_ShouldPreserveFileExtension(string fileName, string expectedExtension)
    {
        // Arrange
        var (service, dbContext) = await CreateServiceAsync();
        var file = CreateMockFormFile(fileName, 1024);

        // Act
        var result = await service.CreateVideoAsync(file, "Test");

        // Assert
        result.FilePath.Should().Contain(expectedExtension);
    }

    #endregion

    #region GetAllAsync Tests

    [Fact]
    public async Task GetAllAsync_WithNoVideos_ShouldReturnEmptyList()
    {
        // Arrange
        var (service, dbContext) = await CreateServiceAsync();

        // Act
        var result = await service.GetAllAsync();

        // Assert
        result.Should().BeEmpty();
    }

    [Fact]
    public async Task GetAllAsync_ShouldReturnVideosOrderedByUploadedAtDescending()
    {
        // Arrange
        var (service, dbContext) = await CreateServiceAsync();

        // Add videos with different timestamps
        var video1 = new VideoEntity
        {
            Name = "Video 1",
            FilePath = "/path/1.mp4",
            FileSizeBytes = 1024,
            FileExtension = ".mp4",
            UploadedAt = DateTime.UtcNow.AddHours(-2),
            Status = nameof(VideoStatus.Uploaded)
        };
        var video2 = new VideoEntity
        {
            Name = "Video 2",
            FilePath = "/path/2.mp4",
            FileSizeBytes = 1024,
            FileExtension = ".mp4",
            UploadedAt = DateTime.UtcNow.AddHours(-1),
            Status = nameof(VideoStatus.Uploaded)
        };
        var video3 = new VideoEntity
        {
            Name = "Video 3",
            FilePath = "/path/3.mp4",
            FileSizeBytes = 1024,
            FileExtension = ".mp4",
            UploadedAt = DateTime.UtcNow,
            Status = nameof(VideoStatus.Uploaded)
        };

        dbContext.Videos.AddRange(video1, video2, video3);
        await dbContext.SaveChangesAsync();

        // Act
        var result = await service.GetAllAsync();

        // Assert
        result.Should().HaveCount(3);
        result[0].Name.Should().Be("Video 3"); // Most recent first
        result[1].Name.Should().Be("Video 2");
        result[2].Name.Should().Be("Video 1");
    }

    #endregion

    #region GetByIdAsync Tests

    [Fact]
    public async Task GetByIdAsync_WithExistingVideo_ShouldReturnVideo()
    {
        // Arrange
        var (service, dbContext) = await CreateServiceAsync();
        var video = new VideoEntity
        {
            Name = "Test Video",
            FilePath = "/path/test.mp4",
            FileSizeBytes = 1024,
            FileExtension = ".mp4",
            UploadedAt = DateTime.UtcNow,
            Status = nameof(VideoStatus.Completed),
            AnalysisId = 42
        };
        dbContext.Videos.Add(video);
        await dbContext.SaveChangesAsync();

        // Act
        var result = await service.GetByIdAsync(video.Id);

        // Assert
        result.Should().NotBeNull();
        result!.Id.Should().Be(video.Id);
        result.Name.Should().Be(video.Name);
        result.Status.Should().Be(VideoStatus.Completed);
        result.AnalysisId.Should().Be(42);
    }

    [Fact]
    public async Task GetByIdAsync_WithNonExistingVideo_ShouldReturnNull()
    {
        // Arrange
        var (service, dbContext) = await CreateServiceAsync();

        // Act
        var result = await service.GetByIdAsync(999);

        // Assert
        result.Should().BeNull();
    }

    #endregion

    #region DeleteAsync Tests

    [Fact]
    public async Task DeleteAsync_WithExistingVideo_ShouldDeleteVideoAndReturnTrue()
    {
        // Arrange
        var (service, dbContext) = await CreateServiceAsync();
        var tempFile = GetTempFilePath();
        await File.WriteAllTextAsync(tempFile, "test content");

        var video = new VideoEntity
        {
            Name = "To Delete",
            FilePath = tempFile,
            FileSizeBytes = 1024,
            FileExtension = ".mp4",
            UploadedAt = DateTime.UtcNow,
            Status = nameof(VideoStatus.Uploaded)
        };
        dbContext.Videos.Add(video);
        await dbContext.SaveChangesAsync();

        // Act
        var result = await service.DeleteAsync(video.Id);

        // Assert
        result.Should().BeTrue();
        (await dbContext.Videos.FindAsync(video.Id)).Should().BeNull();
        File.Exists(tempFile).Should().BeFalse();
    }

    [Fact]
    public async Task DeleteAsync_WithNonExistingVideo_ShouldReturnFalse()
    {
        // Arrange
        var (service, dbContext) = await CreateServiceAsync();

        // Act
        var result = await service.DeleteAsync(999);

        // Assert
        result.Should().BeFalse();
    }

    #endregion

    #region UpdateStatusAsync Tests

    [Theory]
    [InlineData(VideoStatus.Uploaded)]
    [InlineData(VideoStatus.Processing)]
    [InlineData(VideoStatus.Completed)]
    [InlineData(VideoStatus.Failed)]
    public async Task UpdateStatusAsync_ShouldUpdateStatusAndAnalysisId(VideoStatus newStatus)
    {
        // Arrange
        var (service, dbContext) = await CreateServiceAsync();
        var video = new VideoEntity
        {
            Name = "Test Video",
            FilePath = "/path/test.mp4",
            FileSizeBytes = 1024,
            FileExtension = ".mp4",
            UploadedAt = DateTime.UtcNow,
            Status = nameof(VideoStatus.Uploaded)
        };
        dbContext.Videos.Add(video);
        await dbContext.SaveChangesAsync();

        // Act
        var result = await service.UpdateStatusAsync(video.Id, newStatus, 123);

        // Assert
        result.Should().NotBeNull();
        result!.Status.Should().Be(newStatus);
        result.AnalysisId.Should().Be(123);

        var entity = await dbContext.Videos.FindAsync(video.Id);
        entity!.Status.Should().Be(newStatus.ToString());
        entity.AnalysisId.Should().Be(123);
    }

    [Fact]
    public async Task UpdateStatusAsync_WithNonExistingVideo_ShouldReturnNull()
    {
        // Arrange
        var (service, dbContext) = await CreateServiceAsync();

        // Act
        var result = await service.UpdateStatusAsync(999, VideoStatus.Processing);

        // Assert
        result.Should().BeNull();
    }

    #endregion

    #region GetDashboardStatsAsync Tests

    [Fact]
    public async Task GetDashboardStatsAsync_WithNoData_ShouldReturnZeroStats()
    {
        // Arrange
        var (service, dbContext) = await CreateServiceAsync();

        // Act
        var result = await service.GetDashboardStatsAsync();

        // Assert
        result.TotalVideos.Should().Be(0);
        result.TotalAnalyses.Should().Be(0);
        result.CompletedAnalyses.Should().Be(0);
        result.FailedAnalyses.Should().Be(0);
        result.SuccessRatePercent.Should().Be(0);
        result.AvgDetectionRate.Should().Be(0);
        result.RecentVideos.Should().BeEmpty();
        result.RecentAnalyses.Should().BeEmpty();
    }

    [Fact]
    public async Task GetDashboardStatsAsync_ShouldCalculateCorrectStats()
    {
        // Arrange
        var (service, dbContext) = await CreateServiceAsync();

        // Add test data
        var videos = Enumerable.Range(1, 10).Select(i => new VideoEntity
        {
            Name = $"Video {i}",
            FilePath = $"/path/{i}.mp4",
            FileSizeBytes = 1024,
            FileExtension = ".mp4",
            UploadedAt = DateTime.UtcNow.AddMinutes(-i),
            Status = nameof(VideoStatus.Uploaded)
        }).ToList();
        dbContext.Videos.AddRange(videos);

        var analyses = new List<AnalysisEntity>
        {
            new() {
                VideoId = 1,
                Status = nameof(AnalysisStatus.Completed),
                StartedAt = DateTime.UtcNow.AddHours(-1),
                DetectionRatePercent = 85.5
            },
            new() {
                VideoId = 2,
                Status = nameof(AnalysisStatus.Completed),
                StartedAt = DateTime.UtcNow.AddHours(-2),
                DetectionRatePercent = 92.0
            },
            new() {
                VideoId = 3,
                Status = nameof(AnalysisStatus.Failed),
                StartedAt = DateTime.UtcNow.AddHours(-3),
                ErrorMessage = "Processing timeout"
            }
        };
        dbContext.Analyses.AddRange(analyses);
        await dbContext.SaveChangesAsync();

        // Act
        var result = await service.GetDashboardStatsAsync();

        // Assert
        result.TotalVideos.Should().Be(10);
        result.TotalAnalyses.Should().Be(3);
        result.CompletedAnalyses.Should().Be(2);
        result.FailedAnalyses.Should().Be(1);
        result.SuccessRatePercent.Should().BeApproximately(66.67, 0.01);
        result.AvgDetectionRate.Should().BeApproximately(88.75, 0.01);
        result.RecentVideos.Should().HaveCount(5);
        result.RecentAnalyses.Should().HaveCount(3);
    }

    #endregion
}
