using AnalizadorPadel.Api.Data;
using AnalizadorPadel.Api.Models.DTOs;
using AnalizadorPadel.Api.Services;
using Microsoft.EntityFrameworkCore;
using Scalar.AspNetCore;
using Serilog;

// Configure Serilog
Log.Logger = new LoggerConfiguration()
    .MinimumLevel.Information()
    .MinimumLevel.Override("Microsoft.AspNetCore", Serilog.Events.LogEventLevel.Warning)
    .MinimumLevel.Override("Microsoft.EntityFrameworkCore", Serilog.Events.LogEventLevel.Warning)
    .WriteTo.Console(outputTemplate: "[{Timestamp:HH:mm:ss} {Level:u3}] {Message:lj}{NewLine}{Exception}")
    .WriteTo.File("logs/padel-.log",
        rollingInterval: RollingInterval.Day,
        retainedFileCountLimit: 7,
        outputTemplate: "{Timestamp:yyyy-MM-dd HH:mm:ss.fff zzz} [{Level:u3}] {Message:lj}{NewLine}{Exception}")
    .CreateLogger();

try
{
    var builder = WebApplication.CreateBuilder(args);

    // Use Serilog
    builder.Host.UseSerilog();

    // Configure OpenAPI
    builder.Services.AddOpenApi();

    // Configure SQLite with EF Core
    var connectionString = builder.Configuration.GetConnectionString("DefaultConnection")
        ?? "Data Source=padel.db";
    builder.Services.AddDbContext<PadelDbContext>(options =>
        options.UseSqlite(connectionString));
    builder.Services.AddDbContextFactory<PadelDbContext>(options =>
        options.UseSqlite(connectionString), ServiceLifetime.Scoped);

    // Configure CORS
    builder.Services.AddCors(options =>
    {
        options.AddPolicy("Development", policy =>
        {
            policy.WithOrigins("http://localhost:5173", "http://localhost:3000")
                  .AllowAnyHeader()
                  .AllowAnyMethod();
        });

        options.AddPolicy("Production", policy =>
        {
            policy.WithOrigins("http://localhost")
                  .AllowAnyHeader()
                  .AllowAnyMethod();
        });
    });

    // Register services
    builder.Services.AddScoped<VideoService>();
    builder.Services.AddScoped<AnalysisService>();

    var app = builder.Build();

    // Apply pending migrations and ensure database is created
    using (var scope = app.Services.CreateScope())
    {
        var db = scope.ServiceProvider.GetRequiredService<PadelDbContext>();
        db.Database.EnsureCreated();
    }

    // Add OpenAPI endpoint
    app.MapOpenApi();
    app.MapScalarApiReference();

    // Configure CORS
    if (app.Environment.IsDevelopment())
    {
        app.UseCors("Development");
        app.UseDeveloperExceptionPage();
    }
    else
    {
        app.UseCors("Production");
        app.UseExceptionHandler(errorApp =>
        {
            errorApp.Run(async context =>
            {
                context.Response.StatusCode = 500;
                context.Response.ContentType = "application/json";
                var exception = context.Features.Get<Microsoft.AspNetCore.Diagnostics.IExceptionHandlerFeature>()?.Error;
                await context.Response.WriteAsJsonAsync(new { error = "An internal error occurred" });
            });
        });
    }

    // Request logging
    app.UseSerilogRequestLogging();

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

        var allowedExtensions = new[] { ".mp4", ".avi", ".mov" };
        var extension = Path.GetExtension(file.FileName).ToLowerInvariant();
        if (!allowedExtensions.Contains(extension))
        {
            return Results.BadRequest(new ApiResponse<object>(false, $"Formato no soportado: {extension}. Formatos válidos: MP4, AVI, MOV"));
        }

        try
        {
            var name = Path.GetFileNameWithoutExtension(file.FileName);
            var video = await videoService.CreateVideoAsync(file, name);
            return Results.Created($"/api/videos/{video.Id}", new ApiResponse<VideoDto>(true, "Video subido exitosamente", video));
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Error uploading video");
            return Results.StatusCode(500);
        }
    })
    .WithName("CreateVideo")
    .WithSummary("Sube un nuevo video")
    .WithDescription("Permite subir un archivo de video para su posterior análisis. Formatos: MP4, AVI, MOV. Máximo: 500MB")
    .DisableAntiforgery();

    // GET /api/videos - Listar videos
    app.MapGet("/api/videos", async (VideoService videoService) =>
    {
        var videos = await videoService.GetAllAsync();
        return Results.Ok(new ApiResponse<List<VideoDto>>(true, $"{videos.Count} videos encontrados", videos));
    })
    .WithName("GetVideos")
    .WithSummary("Lista todos los videos")
    .WithDescription("Retorna una lista de todos los videos subidos, ordenados por fecha descendente");

    // GET /api/videos/{id} - Obtener video por ID
    app.MapGet("/api/videos/{id:int}", async (int id, VideoService videoService) =>
    {
        var video = await videoService.GetByIdAsync(id);
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
    app.MapDelete("/api/videos/{id:int}", async (int id, VideoService videoService) =>
    {
        var deleted = await videoService.DeleteAsync(id);
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
        var video = await videoService.GetByIdAsync(id);
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
    app.MapGet("/api/analyses/{id:int}", async (int id, AnalysisService analysisService) =>
    {
        var analysis = await analysisService.GetByIdAsync(id);
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
    app.MapGet("/api/analyses/{id:int}/stats", async (int id, AnalysisService analysisService) =>
    {
        var stats = await analysisService.GetStatsAsync(id);
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
    app.MapGet("/api/analyses/{id:int}/heatmap", async (int id, AnalysisService analysisService) =>
    {
        var heatmap = await analysisService.GetHeatmapAsync(id);
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
    app.MapGet("/api/analyses/{id:int}/report", async (int id, AnalysisService analysisService) =>
    {
        var reportPath = await analysisService.GetReportAsync(id);
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

    // ============================================
    // DASHBOARD ENDPOINTS
    // ============================================

    // GET /api/dashboard/stats - Estadísticas del dashboard
    app.MapGet("/api/dashboard/stats", async (VideoService videoService) =>
    {
        var stats = await videoService.GetDashboardStatsAsync();
        return Results.Ok(new ApiResponse<DashboardStats>(true, "Estadísticas del dashboard", stats));
    })
    .WithName("GetDashboardStats")
    .WithSummary("Obtiene estadísticas del dashboard")
    .WithDescription("Retorna las estadísticas generales del dashboard: total de videos, análisis completados, tasa de éxito, etc.");

    Log.Information("AnalizadorPadel API started");
    app.Run();
}
catch (Exception ex)
{
    Log.Fatal(ex, "Application terminated unexpectedly");
}
finally
{
    Log.CloseAndFlush();
}
