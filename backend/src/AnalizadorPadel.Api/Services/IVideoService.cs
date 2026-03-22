using AnalizadorPadel.Api.Models.DTOs;

namespace AnalizadorPadel.Api.Services;

/// <summary>
/// Interfaz para el servicio de gestión de videos
/// </summary>
public interface IVideoService
{
    Task<VideoDto> CreateVideoAsync(IFormFile file, string name, string? description = null);
    Task<List<VideoDto>> GetAllAsync();
    Task<VideoDto?> GetByIdAsync(int id);
    Task<bool> DeleteAsync(int id);
    Task<VideoDto?> UpdateStatusAsync(int id, VideoStatus status, int? analysisId = null);
    Task<DashboardStats> GetDashboardStatsAsync();
    Task<string?> GetFilePathAsync(int id);
}
