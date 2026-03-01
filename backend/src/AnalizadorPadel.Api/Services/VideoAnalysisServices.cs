using System.Text.Json;
using AnalizadorPadel.Api.Models.DTOs;

namespace AnalizadorPadel.Api.Services;

/// <summary>
/// Servicio para gestión de videos - MVP con almacenamiento en memoria
/// </summary>
public class VideoService
{
    private readonly List<VideoDto> _videos = new();
    private int _nextId = 1;
    private readonly string _uploadsPath;
    private readonly string _outputPath;

    public VideoService(IWebHostEnvironment env)
    {
        _uploadsPath = Path.Combine(env.ContentRootPath, "uploads");
        _outputPath = Path.Combine(env.ContentRootPath, "outputs");
        Directory.CreateDirectory(_uploadsPath);
        Directory.CreateDirectory(_outputPath);
    }

    public async Task<VideoDto> CreateVideoAsync(IFormFile file, string name, string? description = null)
    {
        var fileName = $"{Guid.NewGuid()}{Path.GetExtension(file.FileName)}";
        var filePath = Path.Combine(_uploadsPath, fileName);

        using (var stream = new FileStream(filePath, FileMode.Create))
        {
            await file.CopyToAsync(stream);
        }

        var video = new VideoDto(
            Id: _nextId++,
            Name: name,
            Description: description,
            FilePath: filePath,
            UploadedAt: DateTime.UtcNow,
            Status: VideoStatus.Uploaded,
            AnalysisId: null
        );

        _videos.Add(video);
        return video;
    }

    public List<VideoDto> GetAll()
    {
        return _videos.ToList();
    }

    public VideoDto? GetById(int id)
    {
        return _videos.FirstOrDefault(v => v.Id == id);
    }

    public bool Delete(int id)
    {
        var video = _videos.FirstOrDefault(v => v.Id == id);
        if (video == null) return false;

        // Delete file
        if (File.Exists(video.FilePath))
        {
            File.Delete(video.FilePath);
        }

        _videos.Remove(video);
        return true;
    }

    public VideoDto? UpdateStatus(int id, VideoStatus status, int? analysisId = null)
    {
        var video = _videos.FirstOrDefault(v => v.Id == id);
        if (video == null) return null;

        var index = _videos.IndexOf(video);
        _videos[index] = video with { Status = status, AnalysisId = analysisId };
        return _videos[index];
    }
}

/// <summary>
/// Servicio para gestión de análisis - MVP con almacenamiento en memoria
/// </summary>
public class AnalysisService
{
    private readonly List<AnalysisDto> _analyses = new();
    private int _nextId = 1;
    private readonly string _outputPath;
    private readonly VideoService _videoService;
    private readonly JsonSerializerOptions _jsonOptions;

    public AnalysisService(IWebHostEnvironment env, VideoService videoService)
    {
        _outputPath = Path.Combine(env.ContentRootPath, "outputs");
        _videoService = videoService;
        Directory.CreateDirectory(_outputPath);

        _jsonOptions = new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true
        };
    }

    public async Task<AnalysisDto> StartAnalysisAsync(int videoId, int? courtId)
    {
        var video = _videoService.GetById(videoId);
        if (video == null)
            throw new ArgumentException($"Video {videoId} not found");

        // Update video status
        _videoService.UpdateStatus(videoId, VideoStatus.Processing);

        // Create analysis record
        var analysis = new AnalysisDto(
            Id: _nextId++,
            VideoId: videoId,
            StartedAt: DateTime.UtcNow,
            CompletedAt: null,
            Status: AnalysisStatus.Running,
            ErrorMessage: null,
            Result: null
        );

        _analyses.Add(analysis);

        // Update video with analysis ID
        _videoService.UpdateStatus(videoId, VideoStatus.Processing, analysis.Id);

        // Run analysis in background
        _ = Task.Run(async () => await RunAnalysisAsync(analysis.Id, video.FilePath));

        return analysis;
    }

    private async Task RunAnalysisAsync(int analysisId, string videoPath)
    {
        try
        {
            var analysis = _analyses.FirstOrDefault(a => a.Id == analysisId);
            if (analysis == null) return;

            // Prepare paths for Python script
            var modelsPath = Path.Combine(Directory.GetCurrentDirectory(), "..", "..", "..", "models");
            var pythonScriptPath = Path.Combine(Directory.GetCurrentDirectory(), "..", "..", "..", "python-scripts", "process_video.py");
            var resultPath = Path.Combine(_outputPath, $"analysis_{analysisId}_result.json");

            // Execute Python script
            var processInfo = new System.Diagnostics.ProcessStartInfo
            {
                FileName = "python3",
                Arguments = $"\"{pythonScriptPath}\" \"{videoPath}\" \"{resultPath}\" \"{modelsPath}\"",
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

            var output = await process.StandardOutput.ReadToEndAsync();
            var error = await process.StandardError.ReadToEndAsync();
            await process.WaitForExitAsync();

            if (process.ExitCode != 0)
            {
                await UpdateAnalysisFailedAsync(analysisId, $"Python script failed: {error}");
                return;
            }

            // Read and parse results
            if (File.Exists(resultPath))
            {
                var resultJson = await File.ReadAllTextAsync(resultPath);
                var result = JsonSerializer.Deserialize<AnalysisResult>(resultJson, _jsonOptions);

                var completedAnalysis = analysis with
                {
                    Status = AnalysisStatus.Completed,
                    CompletedAt = DateTime.UtcNow,
                    Result = result
                };

                var index = _analyses.IndexOf(analysis);
                _analyses[index] = completedAnalysis;

                // Update video status
                _videoService.UpdateStatus(analysis.VideoId, VideoStatus.Completed, analysisId);
            }
            else
            {
                await UpdateAnalysisFailedAsync(analysisId, "Result file not found");
            }
        }
        catch (Exception ex)
        {
            await UpdateAnalysisFailedAsync(analysisId, ex.Message);
        }
    }

    private async Task UpdateAnalysisFailedAsync(int analysisId, string error)
    {
        var analysis = _analyses.FirstOrDefault(a => a.Id == analysisId);
        if (analysis == null) return;

        var index = _analyses.IndexOf(analysis);
        _analyses[index] = analysis with
        {
            Status = AnalysisStatus.Failed,
            CompletedAt = DateTime.UtcNow,
            ErrorMessage = error
        };

        // Update video status
        _videoService.UpdateStatus(analysis.VideoId, VideoStatus.Failed, analysisId);
    }

    public AnalysisDto? GetById(int id)
    {
        return _analyses.FirstOrDefault(a => a.Id == id);
    }

    public AnalysisStats? GetStats(int id)
    {
        var analysis = _analyses.FirstOrDefault(a => a.Id == id);
        if (analysis?.Result == null) return null;

        return new AnalysisStats(
            TotalFrames: analysis.Result.TotalFrames,
            FramesWith4Players: analysis.Result.FramesWith4Players,
            DetectionRatePercent: analysis.Result.DetectionRatePercent,
            AvgDetectionsPerFrame: analysis.Result.AvgDetectionsPerFrame,
            PlayersDetected: analysis.Result.PlayersDetected,
            ProcessingTimeSeconds: analysis.Result.ProcessingTimeSeconds,
            ModelUsed: analysis.Result.ModelUsed
        );
    }

    public HeatmapData? GetHeatmap(int id)
    {
        // MVP: Return placeholder heatmap data
        // TODO: Implement actual heatmap generation from analysis data
        var analysis = _analyses.FirstOrDefault(a => a.Id == id);
        if (analysis?.Result == null) return null;

        var points = new List<HeatmapPoint>();
        var random = new Random(id); // Seeded for consistency
        
        // Generate sample heatmap points for the court
        for (int i = 0; i < 100; i++)
        {
            points.Add(new HeatmapPoint(
                X: random.NextDouble() * 23.77, // Padel court length
                Y: random.NextDouble() * 10.97, // Padel court width
                Intensity: random.Next(1, 10)
            ));
        }

        return new HeatmapData(
            Points: points,
            CourtDimensions: "23.77m x 10.97m"
        );
    }

    public string? GetReport(int id)
    {
        // MVP: Return placeholder report path
        // TODO: Implement actual PDF report generation
        var analysis = _analyses.FirstOrDefault(a => a.Id == id);
        if (analysis?.Result == null) return null;

        return $"/api/analyses/{id}/report-placeholder";
    }
}
