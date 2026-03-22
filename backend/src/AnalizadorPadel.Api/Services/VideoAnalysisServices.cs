using System.Text.Json;
using AnalizadorPadel.Api.Data;
using AnalizadorPadel.Api.Models.DTOs;
using AnalizadorPadel.Api.Models.Entities;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;

namespace AnalizadorPadel.Api.Services;

internal static class PathConfigurationHelper
{
    public static string GetConfiguredStoragePath(IWebHostEnvironment env, IConfiguration configuration, string configurationKey, string folderName)
    {
        var configured = configuration[configurationKey];
        if (!string.IsNullOrWhiteSpace(configured))
        {
            return configured;
        }

        return Path.Combine(GetRepositoryRootOrContentRoot(env), "var", folderName);
    }

    public static string GetRepositoryScopedPath(IWebHostEnvironment env, IConfiguration configuration, string configurationKey, string folderName)
    {
        var configured = configuration[configurationKey];
        if (!string.IsNullOrWhiteSpace(configured))
        {
            return configured;
        }

        return Path.Combine(GetRepositoryRootOrContentRoot(env), folderName);
    }

    public static string GetRepositoryScopedFile(IWebHostEnvironment env, IConfiguration configuration, string configurationKey, params string[] pathSegments)
    {
        var configured = configuration[configurationKey];
        if (!string.IsNullOrWhiteSpace(configured))
        {
            return configured;
        }

        return Path.Combine(new[] { GetRepositoryRootOrContentRoot(env) }.Concat(pathSegments).ToArray());
    }

    private static string GetRepositoryRootOrContentRoot(IWebHostEnvironment env)
    {
        var current = new DirectoryInfo(env.ContentRootPath);

        while (current != null)
        {
            if (File.Exists(Path.Combine(current.FullName, "AnalizadorPadel.sln")) ||
                Directory.Exists(Path.Combine(current.FullName, ".git")))
            {
                return current.FullName;
            }

            current = current.Parent;
        }

        return env.ContentRootPath;
    }
}

/// <summary>
/// Servicio para gestión de videos - Persistencia con EF Core + SQLite
/// </summary>
public class VideoService : IVideoService
{
    private readonly IDbContextFactory<PadelDbContext> _dbFactory;
    private readonly string _uploadsPath;
    private readonly ILogger<VideoService> _logger;

    public VideoService(
        IDbContextFactory<PadelDbContext> dbFactory,
        IWebHostEnvironment env,
        IConfiguration configuration,
        ILogger<VideoService> logger)
    {
        _dbFactory = dbFactory;
        _uploadsPath = PathConfigurationHelper.GetConfiguredStoragePath(env, configuration, "Storage:UploadsPath", "uploads");
        _logger = logger;
        Directory.CreateDirectory(_uploadsPath);
    }

    public async Task<VideoDto> CreateVideoAsync(IFormFile file, string name, string? description = null)
    {
        var extension = Path.GetExtension(file.FileName).ToLowerInvariant();
        var fileName = $"{Guid.NewGuid()}{extension}";
        var filePath = Path.Combine(_uploadsPath, fileName);

        await using (var stream = new FileStream(filePath, FileMode.Create))
        {
            await file.CopyToAsync(stream);
        }

        var entity = new VideoEntity
        {
            Name = name,
            Description = description,
            FilePath = filePath,
            FileSizeBytes = file.Length,
            FileExtension = extension,
            UploadedAt = DateTime.UtcNow,
            Status = nameof(VideoStatus.Uploaded)
        };

        await using var db = await _dbFactory.CreateDbContextAsync();
        db.Videos.Add(entity);
        await db.SaveChangesAsync();

        _logger.LogInformation("Video uploaded: {VideoId} ({Name}, {Size} bytes)", entity.Id, name, file.Length);

        return MapToDto(entity);
    }

    public async Task<List<VideoDto>> GetAllAsync()
    {
        await using var db = await _dbFactory.CreateDbContextAsync();
        var entities = await db.Videos
            .OrderByDescending(v => v.UploadedAt)
            .ToListAsync();
        return entities.Select(MapToDto).ToList();
    }

    public async Task<VideoDto?> GetByIdAsync(int id)
    {
        await using var db = await _dbFactory.CreateDbContextAsync();
        var entity = await db.Videos.FindAsync(id);
        return entity == null ? null : MapToDto(entity);
    }

    public async Task<bool> DeleteAsync(int id)
    {
        await using var db = await _dbFactory.CreateDbContextAsync();
        var entity = await db.Videos.FindAsync(id);
        if (entity == null) return false;

        // Delete physical file
        if (File.Exists(entity.FilePath))
        {
            File.Delete(entity.FilePath);
            _logger.LogInformation("Deleted video file: {FilePath}", entity.FilePath);
        }

        db.Videos.Remove(entity);
        await db.SaveChangesAsync();

        _logger.LogInformation("Video deleted: {VideoId}", id);
        return true;
    }

    public async Task<VideoDto?> UpdateStatusAsync(int id, VideoStatus status, int? analysisId = null)
    {
        await using var db = await _dbFactory.CreateDbContextAsync();
        var entity = await db.Videos.FindAsync(id);
        if (entity == null) return null;

        entity.Status = status.ToString();
        entity.AnalysisId = analysisId;
        await db.SaveChangesAsync();

        return MapToDto(entity);
    }

    public async Task<DashboardStats> GetDashboardStatsAsync()
    {
        await using var db = await _dbFactory.CreateDbContextAsync();
        
        // Use database-level counting to avoid loading all entities into memory
        var totalVideos = await db.Videos.CountAsync();
        var totalAnalyses = await db.Analyses.CountAsync();
        var completedAnalyses = await db.Analyses.CountAsync(a => a.Status == nameof(AnalysisStatus.Completed));
        var failedAnalyses = await db.Analyses.CountAsync(a => a.Status == nameof(AnalysisStatus.Failed));
        var successRate = totalAnalyses > 0 ? (double)completedAnalyses / totalAnalyses * 100 : 0;
        
        var avgDetectionRate = await db.Analyses
            .Where(a => a.DetectionRatePercent.HasValue)
            .Select(a => a.DetectionRatePercent!.Value)
            .ToListAsync();
        var avgDetectionRateValue = avgDetectionRate.Count > 0 ? avgDetectionRate.Average() : 0;
        
        var recentVideoEntities = await db.Videos
            .OrderByDescending(v => v.UploadedAt)
            .Take(5)
            .ToListAsync();
        var recentVideos = recentVideoEntities.Select(MapToDto).ToList();
        
        // Load recent analyses (limited to 5) - select status as string then parse in memory
        var recentAnalysisEntities = await db.Analyses
            .OrderByDescending(a => a.StartedAt)
            .Take(5)
            .ToListAsync();
        var recentAnalyses = recentAnalysisEntities.Select(a => new AnalysisDto(
            Id: a.Id,
            VideoId: a.VideoId,
            StartedAt: a.StartedAt,
            CompletedAt: a.CompletedAt,
            Status: Enum.TryParse<AnalysisStatus>(a.Status, out var parsed) ? parsed : AnalysisStatus.Pending,
            ErrorMessage: a.ErrorMessage,
            Result: null
        )).ToList();
        
        return new DashboardStats(
            TotalVideos: totalVideos,
            TotalAnalyses: totalAnalyses,
            CompletedAnalyses: completedAnalyses,
            FailedAnalyses: failedAnalyses,
            SuccessRatePercent: successRate,
            AvgDetectionRate: avgDetectionRateValue,
            RecentVideos: recentVideos,
            RecentAnalyses: recentAnalyses
        );
    }

    /// <summary>
    /// Gets the file path for a video (internal use by AnalysisService)
    /// </summary>
    public async Task<string?> GetFilePathAsync(int id)
    {
        await using var db = await _dbFactory.CreateDbContextAsync();
        var entity = await db.Videos.FindAsync(id);
        return entity?.FilePath;
    }

    private static VideoDto MapToDto(VideoEntity entity)
    {
        var status = Enum.TryParse<VideoStatus>(entity.Status, out var parsed)
            ? parsed
            : VideoStatus.Uploaded;

        return new VideoDto(
            Id: entity.Id,
            Name: entity.Name,
            Description: entity.Description,
            FilePath: entity.FilePath,
            UploadedAt: entity.UploadedAt,
            Status: status,
            AnalysisId: entity.AnalysisId
        );
    }
}

/// <summary>
/// Servicio para gestión de análisis - Persistencia con EF Core + SQLite
/// </summary>
public class AnalysisService
{
    private readonly IDbContextFactory<PadelDbContext> _dbFactory;
    private readonly string _outputPath;
    private readonly string _modelsPath;
    private readonly string _pythonScriptPath;
    private readonly IVideoService _videoService;
    private readonly ILogger<AnalysisService> _logger;
    private readonly JsonSerializerOptions _jsonOptions;

    public AnalysisService(
        IDbContextFactory<PadelDbContext> dbFactory,
        IWebHostEnvironment env,
        IConfiguration configuration,
        IVideoService videoService,
        ILogger<AnalysisService> logger)
    {
        _dbFactory = dbFactory;
        _outputPath = PathConfigurationHelper.GetConfiguredStoragePath(env, configuration, "Storage:OutputsPath", "outputs");
        _modelsPath = PathConfigurationHelper.GetRepositoryScopedPath(env, configuration, "Processing:ModelsPath", "ml-models");
        _pythonScriptPath = PathConfigurationHelper.GetRepositoryScopedFile(env, configuration, "Processing:PythonScriptPath", "python-scripts", "process_video.py");
        _videoService = videoService;
        _logger = logger;
        Directory.CreateDirectory(_outputPath);

        _jsonOptions = new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true
        };
    }

    public async Task<AnalysisDto> StartAnalysisAsync(int videoId, int? courtId)
    {
        var video = await _videoService.GetByIdAsync(videoId);
        if (video == null)
            throw new ArgumentException($"Video {videoId} not found");

        // Update video status
        await _videoService.UpdateStatusAsync(videoId, VideoStatus.Processing);

        // Create analysis record
        var entity = new AnalysisEntity
        {
            VideoId = videoId,
            StartedAt = DateTime.UtcNow,
            Status = nameof(AnalysisStatus.Running)
        };

        await using (var db = await _dbFactory.CreateDbContextAsync())
        {
            db.Analyses.Add(entity);
            await db.SaveChangesAsync();
        }

        // Update video with analysis ID
        await _videoService.UpdateStatusAsync(videoId, VideoStatus.Processing, entity.Id);

        _logger.LogInformation("Analysis started: {AnalysisId} for video {VideoId}", entity.Id, videoId);

        // Run analysis in background with proper error handling
        _ = Task.Run(async () =>
        {
            try
            {
                await RunAnalysisAsync(entity.Id, video.FilePath);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Unhandled error in background analysis {AnalysisId}", entity.Id);
                await UpdateAnalysisFailedAsync(entity.Id, $"Unhandled error: {ex.Message}");
            }
        });

        return MapToDto(entity);
    }

    private async Task RunAnalysisAsync(int analysisId, string videoPath)
    {
        try
        {
            // Prepare paths for Python script
            var resultPath = Path.Combine(_outputPath, $"analysis_{analysisId}_result.json");

            _logger.LogInformation("Running Python analysis: script={Script}, video={Video}, output={Output}",
                _pythonScriptPath, videoPath, resultPath);

            // Execute Python script
            var processInfo = new System.Diagnostics.ProcessStartInfo
            {
                FileName = "python3",
                Arguments = $"\"{_pythonScriptPath}\" \"{videoPath}\" \"{resultPath}\" \"{_modelsPath}\"",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };

            using var process = System.Diagnostics.Process.Start(processInfo);
            if (process == null)
            {
                await UpdateAnalysisFailedAsync(analysisId, "Failed to start Python process");
                return;
            }

            // Read output streams to prevent deadlocks
            var outputTask = process.StandardOutput.ReadToEndAsync();
            var errorTask = process.StandardError.ReadToEndAsync();

            // Wait with timeout (10 minutes as per PLANNING.md)
            var completed = await Task.Run(() => process.WaitForExit(TimeSpan.FromMinutes(10)));
            if (!completed)
            {
                process.Kill(entireProcessTree: true);
                await UpdateAnalysisFailedAsync(analysisId, "Processing timeout: exceeded 10 minutes");
                _logger.LogWarning("Analysis {AnalysisId} killed due to timeout", analysisId);
                return;
            }

            var output = await outputTask;
            var error = await errorTask;

            if (process.ExitCode != 0)
            {
                _logger.LogError("Python script failed for analysis {AnalysisId}: {Error}", analysisId, error);
                await UpdateAnalysisFailedAsync(analysisId, $"Python script failed: {error}");
                return;
            }

            // Read and parse results
            if (File.Exists(resultPath))
            {
                var resultJson = await File.ReadAllTextAsync(resultPath);
                var result = JsonSerializer.Deserialize<AnalysisResult>(resultJson, _jsonOptions);

                if (result != null)
                {
                    await using var db = await _dbFactory.CreateDbContextAsync();
                    var analysis = await db.Analyses.FindAsync(analysisId);
                    if (analysis != null)
                    {
                        analysis.Status = nameof(AnalysisStatus.Completed);
                        analysis.CompletedAt = DateTime.UtcNow;
                        analysis.TotalFrames = result.TotalFrames;
                        analysis.PlayersDetected = result.PlayersDetected;
                        analysis.AvgDetectionsPerFrame = result.AvgDetectionsPerFrame;
                        analysis.FramesWith4Players = result.FramesWith4Players;
                        analysis.DetectionRatePercent = result.DetectionRatePercent;
                        analysis.ProcessingTimeSeconds = result.ProcessingTimeSeconds;
                        analysis.ModelUsed = result.ModelUsed;
                        analysis.VideoPath = result.VideoPath;
                        analysis.Timestamp = result.Timestamp;
                        await db.SaveChangesAsync();
                    }

                    await _videoService.UpdateStatusAsync(
                        (await db.Analyses.FindAsync(analysisId))?.VideoId ?? 0,
                        VideoStatus.Completed, analysisId);

                    _logger.LogInformation("Analysis {AnalysisId} completed successfully. Frames: {Frames}, Detection rate: {Rate}%",
                        analysisId, result.TotalFrames, result.DetectionRatePercent);
                }
                else
                {
                    await UpdateAnalysisFailedAsync(analysisId, "Failed to parse result JSON");
                }
            }
            else
            {
                await UpdateAnalysisFailedAsync(analysisId, "Result file not found");
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error running analysis {AnalysisId}", analysisId);
            await UpdateAnalysisFailedAsync(analysisId, ex.Message);
        }
    }

    private async Task UpdateAnalysisFailedAsync(int analysisId, string error)
    {
        await using var db = await _dbFactory.CreateDbContextAsync();
        var analysis = await db.Analyses.FindAsync(analysisId);
        if (analysis == null) return;

        analysis.Status = nameof(AnalysisStatus.Failed);
        analysis.CompletedAt = DateTime.UtcNow;
        analysis.ErrorMessage = error;
        await db.SaveChangesAsync();

        // Update video status
        await _videoService.UpdateStatusAsync(analysis.VideoId, VideoStatus.Failed, analysisId);

        _logger.LogWarning("Analysis {AnalysisId} failed: {Error}", analysisId, error);
    }

    public async Task<AnalysisDto?> GetByIdAsync(int id)
    {
        await using var db = await _dbFactory.CreateDbContextAsync();
        var entity = await db.Analyses.FindAsync(id);
        return entity == null ? null : MapToDto(entity);
    }

    public async Task<AnalysisStats?> GetStatsAsync(int id)
    {
        await using var db = await _dbFactory.CreateDbContextAsync();
        var analysis = await db.Analyses.FindAsync(id);
        if (analysis == null || analysis.Status != nameof(AnalysisStatus.Completed)) return null;

        return new AnalysisStats(
            TotalFrames: analysis.TotalFrames ?? 0,
            FramesWith4Players: analysis.FramesWith4Players ?? 0,
            DetectionRatePercent: analysis.DetectionRatePercent ?? 0,
            AvgDetectionsPerFrame: analysis.AvgDetectionsPerFrame ?? 0,
            PlayersDetected: analysis.PlayersDetected ?? 0,
            ProcessingTimeSeconds: analysis.ProcessingTimeSeconds ?? 0,
            ModelUsed: analysis.ModelUsed ?? "unknown"
        );
    }

    public async Task<HeatmapData?> GetHeatmapAsync(int id)
    {
        await using var db = await _dbFactory.CreateDbContextAsync();
        var analysis = await db.Analyses.FindAsync(id);
        if (analysis == null || analysis.Status != nameof(AnalysisStatus.Completed)) return null;

        // MVP: Return placeholder heatmap data seeded by analysis ID
        // TODO: Implement actual heatmap generation from frame-by-frame position data
        var points = new List<HeatmapPoint>();
        var random = new Random(id);

        for (int i = 0; i < 100; i++)
        {
            points.Add(new HeatmapPoint(
                X: random.NextDouble() * 23.77,
                Y: random.NextDouble() * 10.97,
                Intensity: random.Next(1, 10)
            ));
        }

        return new HeatmapData(
            Points: points,
            CourtDimensions: "23.77m x 10.97m"
        );
    }

    public async Task<string?> GetReportAsync(int id)
    {
        await using var db = await _dbFactory.CreateDbContextAsync();
        var analysis = await db.Analyses.FindAsync(id);
        if (analysis == null || analysis.Status != nameof(AnalysisStatus.Completed)) return null;

        // MVP: Return placeholder report path
        // TODO: Implement actual PDF report generation
        return $"/api/analyses/{id}/report-placeholder.pdf";
    }

    private static AnalysisDto MapToDto(AnalysisEntity entity)
    {
        var status = Enum.TryParse<AnalysisStatus>(entity.Status, out var parsed)
            ? parsed
            : AnalysisStatus.Pending;

        AnalysisResult? result = null;
        if (entity.Status == nameof(AnalysisStatus.Completed) && entity.TotalFrames.HasValue)
        {
            result = new AnalysisResult(
                Status: entity.Status,
                VideoPath: entity.VideoPath ?? "",
                ProcessingTimeSeconds: entity.ProcessingTimeSeconds ?? 0,
                TotalFrames: entity.TotalFrames ?? 0,
                PlayersDetected: entity.PlayersDetected ?? 0,
                AvgDetectionsPerFrame: entity.AvgDetectionsPerFrame ?? 0,
                FramesWith4Players: entity.FramesWith4Players ?? 0,
                DetectionRatePercent: entity.DetectionRatePercent ?? 0,
                ModelUsed: entity.ModelUsed ?? "unknown",
                Timestamp: entity.Timestamp ?? ""
            );
        }

        return new AnalysisDto(
            Id: entity.Id,
            VideoId: entity.VideoId,
            StartedAt: entity.StartedAt,
            CompletedAt: entity.CompletedAt,
            Status: status,
            ErrorMessage: entity.ErrorMessage,
            Result: result
        );
    }

}
