using System.Text.Json;
using AnalizadorPadel.Api.Models.DTOs;
using AnalizadorPadel.Api.Services;
using Scalar.AspNetCore;

var builder = WebApplication.CreateBuilder(args);

// Configure OpenAPI
builder.Services.AddOpenApi();

// Register services
builder.Services.AddSingleton<VideoService>(sp => 
    new VideoService(sp.GetRequiredService<IWebHostEnvironment>()));
builder.Services.AddSingleton<AnalysisService>(sp =>
    new AnalysisService(
        sp.GetRequiredService<IWebHostEnvironment>(),
        sp.GetRequiredService<VideoService>()));

var app = builder.Build();

// Add OpenAPI endpoint
app.MapOpenApi();
app.MapScalarApiReference();

// Configure the HTTP request pipeline
if (app.Environment.IsDevelopment())
{
    app.UseDeveloperExceptionPage();
}
else
{
    app.UseExceptionHandler(errorApp =>
    {
        errorApp.Run(async context =>
        {
            context.Response.StatusCode = 500;
            context.Response.ContentType = "application/json";
            var exception = context.Features.Get<Microsoft.AspNetCore.Diagnostics.IExceptionHandlerFeature>()?.Error;
            await context.Response.WriteAsJsonAsync(new { error = exception?.Message });
        });
    });
}

// ============================================
// VIDEO ENDPOINTS
// ============================================

// POST /api/videos - Subir nuevo video
app.MapPost("/api/videos", async (IFormFile file, VideoService videoService) =>
{
    if (file == null || file.Length == 0)
    {
        return Results.BadRequest(new ApiResponse<object>(false, "No se proporcionó ningún video"));
    }

    const long maxFileSize = 500 * 1024 * 1024; // 500MB
    if (file.Length > maxFileSize)
    {
        return Results.BadRequest(new ApiResponse<object>(false, "El video excede el tamaño máximo de 500MB"));
    }

    try
    {
        var name = Path.GetFileNameWithoutExtension(file.FileName);
        var video = await videoService.CreateVideoAsync(file, name);
        return Results.Created($"/api/videos/{video.Id}", new ApiResponse<VideoDto>(true, "Video subido exitosamente", video));
    }
    catch (Exception ex)
    {
        return Results.StatusCode(500);
    }
})
.WithName("CreateVideo")
.WithSummary("Sube un nuevo video")
.WithDescription("Permite subir un archivo de video para su posterior análisis");

// GET /api/videos - Listar videos
app.MapGet("/api/videos", (VideoService videoService) =>
{
    var videos = videoService.GetAll();
    return Results.Ok(new ApiResponse<List<VideoDto>>(true, $"{videos.Count} videos encontrados", videos));
})
.WithName("GetVideos")
.WithSummary("Lista todos los videos")
.WithDescription("Retorna una lista de todos los videos subidos");

// GET /api/videos/{id} - Obtener video por ID
app.MapGet("/api/videos/{id:int}", (int id, VideoService videoService) =>
{
    var video = videoService.GetById(id);
    if (video == null)
    {
        return Results.NotFound(new ApiResponse<object>(false, $"Video {id} no encontrado"));
    }
    return Results.Ok(new ApiResponse<VideoDto>(true, "Video encontrado", video));
})
.WithName("GetVideo")
.WithSummary("Obtiene un video por ID")
.WithDescription("Retorna los detalles de un video específico");

// DELETE /api/videos/{id} - Eliminar video
app.MapDelete("/api/videos/{id:int}", (int id, VideoService videoService) =>
{
    var deleted = videoService.Delete(id);
    if (!deleted)
    {
        return Results.NotFound(new ApiResponse<object>(false, $"Video {id} no encontrado"));
    }
    return Results.Ok(new ApiResponse<object>(true, "Video eliminado exitosamente"));
})
.WithName("DeleteVideo")
.WithSummary("Elimina un video")
.WithDescription("Elimina un video y sus archivos asociados");

// ============================================
// ANALYSIS ENDPOINTS
// ============================================

// POST /api/videos/{id}/analyse - Iniciar análisis
app.MapPost("/api/videos/{id:int}/analyse", async (int id, AnalysisService analysisService, VideoService videoService) =>
{
    var video = videoService.GetById(id);
    if (video == null)
    {
        return Results.NotFound(new ApiResponse<object>(false, $"Video {id} no encontrado"));
    }

    try
    {
        var analysis = await analysisService.StartAnalysisAsync(id, null);
        return Results.Accepted($"/api/analyses/{analysis.Id}", new ApiResponse<AnalysisDto>(true, "Análisis iniciado", analysis));
    }
    catch (Exception ex)
    {
        return Results.BadRequest(new ApiResponse<object>(false, ex.Message));
    }
})
.WithName("StartAnalysis")
.WithSummary("Inicia el análisis de un video")
.WithDescription("Comienza el procesamiento del video para detectar jugadores");

// GET /api/analyses/{id} - Obtener análisis
app.MapGet("/api/analyses/{id:int}", (int id, AnalysisService analysisService) =>
{
    var analysis = analysisService.GetById(id);
    if (analysis == null)
    {
        return Results.NotFound(new ApiResponse<object>(false, $"Análisis {id} no encontrado"));
    }
    return Results.Ok(new ApiResponse<AnalysisDto>(true, "Análisis encontrado", analysis));
})
.WithName("GetAnalysis")
.WithSummary("Obtiene un análisis por ID")
.WithDescription("Retorna los detalles de un análisis específico");

// GET /api/analyses/{id}/stats - Estadísticas del análisis
app.MapGet("/api/analyses/{id:int}/stats", (int id, AnalysisService analysisService) =>
{
    var stats = analysisService.GetStats(id);
    if (stats == null)
    {
        return Results.NotFound(new ApiResponse<object>(false, $"Análisis {id} no encontrado o sin resultados"));
    }
    return Results.Ok(new ApiResponse<AnalysisStats>(true, "Estadísticas obtenidas", stats));
})
.WithName("GetAnalysisStats")
.WithSummary("Obtiene estadísticas del análisis")
.WithDescription("Retorna las estadísticas de detección del análisis");

// GET /api/analyses/{id}/heatmap - Datos del heatmap
app.MapGet("/api/analyses/{id:int}/heatmap", (int id, AnalysisService analysisService) =>
{
    var heatmap = analysisService.GetHeatmap(id);
    if (heatmap == null)
    {
        return Results.NotFound(new ApiResponse<object>(false, $"Análisis {id} no encontrado o sin resultados"));
    }
    return Results.Ok(new ApiResponse<HeatmapData>(true, "Datos de heatmap obtenidos", heatmap));
})
.WithName("GetAnalysisHeatmap")
.WithSummary("Obtiene datos del heatmap")
.WithDescription("Retorna los datos para visualizar el heatmap de posiciones");

// GET /api/analyses/{id}/report - Descargar PDF
app.MapGet("/api/analyses/{id:int}/report", (int id, AnalysisService analysisService) =>
{
    var reportPath = analysisService.GetReport(id);
    if (reportPath == null)
    {
        return Results.NotFound(new ApiResponse<object>(false, $"Análisis {id} no encontrado o sin resultados"));
    }
    return Results.Ok(new ApiResponse<string>(true, "Report path", reportPath));
})
.WithName("GetAnalysisReport")
.WithSummary("Obtiene el reporte del análisis")
.WithDescription("Retorna la ruta del reporte PDF del análisis");

// ============================================
// HEALTH ENDPOINT
// ============================================

app.MapGet("/api/health", () => new ApiResponse<object>(true, "API healthy", new { status = "healthy", timestamp = DateTime.UtcNow }))
    .WithName("HealthCheck")
    .WithSummary("Health check endpoint");

app.Run();
