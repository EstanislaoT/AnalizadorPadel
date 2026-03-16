namespace AnalizadorPadel.Api.Models.DTOs;

/// <summary>
/// Request para subir un nuevo video
/// </summary>
public record CreateVideoRequest(
    string Name,
    string? Description = null
);

/// <summary>
/// Video entity
/// </summary>
public record VideoDto(
    int Id,
    string Name,
    string? Description,
    string FilePath,
    DateTime UploadedAt,
    VideoStatus Status,
    int? AnalysisId
);

public enum VideoStatus
{
    Uploaded,
    Processing,
    Completed,
    Failed
}

/// <summary>
/// Request para iniciar análisis
/// </summary>
public record StartAnalysisRequest(
    int? CourtId = null
);

/// <summary>
/// Analysis entity
/// </summary>
public record AnalysisDto(
    int Id,
    int VideoId,
    DateTime StartedAt,
    DateTime? CompletedAt,
    AnalysisStatus Status,
    string? ErrorMessage,
    AnalysisResult? Result
);

public enum AnalysisStatus
{
    Pending,
    Running,
    Completed,
    Failed
}

/// <summary>
/// Resultado del análisis (datos de YOLO)
/// </summary>
public record AnalysisResult(
    string Status,
    string VideoPath,
    double ProcessingTimeSeconds,
    int TotalFrames,
    int PlayersDetected,
    double AvgDetectionsPerFrame,
    int FramesWith4Players,
    double DetectionRatePercent,
    string ModelUsed,
    string Timestamp
);

/// <summary>
/// Estadísticas del análisis
/// </summary>
public record AnalysisStats(
    int TotalFrames,
    int FramesWith4Players,
    double DetectionRatePercent,
    double AvgDetectionsPerFrame,
    int PlayersDetected,
    double ProcessingTimeSeconds,
    string ModelUsed
);

/// <summary>
/// Datos para heatmap
/// </summary>
public record HeatmapData(
    List<HeatmapPoint> Points,
    string CourtDimensions
);

public record HeatmapPoint(
    double X,
    double Y,
    int Intensity
);

/// <summary>
/// Estadísticas del dashboard
/// </summary>
public record DashboardStats(
    int TotalVideos,
    int TotalAnalyses,
    int CompletedAnalyses,
    int FailedAnalyses,
    double SuccessRatePercent,
    double AvgDetectionRate,
    List<VideoDto> RecentVideos,
    List<AnalysisDto> RecentAnalyses
);

/// <summary>
/// Response genérico para operaciones
/// </summary>
/// <typeparam name="T">Tipo de datos</typeparam>
public record ApiResponse<T>
{
    public bool Success { get; init; }
    public string Message { get; init; }
    public T? Data { get; init; }

    public ApiResponse()
    {
        Success = false;
        Message = string.Empty;
        Data = default;
    }

    public ApiResponse(bool success, string message, T? data = default)
    {
        Success = success;
        Message = message;
        Data = data;
    }

    public static ApiResponse<T> Ok(string message, T? data) => new(true, message, data);
    public static ApiResponse<T> Fail(string message) => new(false, message, default);
}
