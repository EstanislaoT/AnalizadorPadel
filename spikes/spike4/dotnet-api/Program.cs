using System.Diagnostics;
using System.Text.Json;
using Scalar.AspNetCore;

var builder = WebApplication.CreateBuilder(args);

// Configure OpenAPI
builder.Services.AddOpenApi();

// Configure CORS
/*builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowAll", policy =>
    {
        policy.AllowAnyOrigin()
              .AllowAnyMethod()
              .AllowAnyHeader();
    });
});*/

// Disable anti-forgery for this API
// builder.Services.AddAntiforgery(options => options.SuppressXFrameOptionsHeader = false);

// Cache JsonSerializerOptions for reuse
var jsonOptions = new JsonSerializerOptions
{
    PropertyNameCaseInsensitive = true
};

var app = builder.Build();

// Add OpenAPI endpoint
app.MapOpenApi();
app.MapScalarApiReference();

// Configure the HTTP request pipeline.
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

//app.UseHttpsRedirection();
//app.UseCors("AllowAll");
//app.UseAuthorization();
//app.UseAntiforgery();

// Helper method to execute Python script
async Task<(bool Success, string Error)> ExecutePythonScript(string scriptPath, string videoPath, string outputPath, string modelsPath)
{
    try
    {
        var processInfo = new ProcessStartInfo
        {
            FileName = "python3",
            Arguments = $"\"{scriptPath}\" \"{videoPath}\" \"{outputPath}\" \"{modelsPath}\"",
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true
        };

        using var process = Process.Start(processInfo);
        if (process == null)
        {
            return (false, "No se pudo iniciar el proceso Python");
        }

        var output = await process.StandardOutput.ReadToEndAsync();
        var error = await process.StandardError.ReadToEndAsync();
        await process.WaitForExitAsync();

        if (process.ExitCode != 0)
        {
            return (false, $"Python script failed with exit code {process.ExitCode}: {error}");
        }

        Console.WriteLine($"🐍 Python output: {output}");
        return (true, string.Empty);
    }
    catch (Exception ex)
    {
        return (false, $"Error executing Python script: {ex.Message}");
    }
}

// Endpoint para procesar video
app.MapPost("/api/video/process", async (IFormFile video, IWebHostEnvironment env, HttpContext context) =>
{
    // Disable antiforgery validation for this endpoint
    //context.Response.Headers.Add("X-Content-Type-Options", "nosniff");
    context.Response.Headers.Append("X-Content-Type-Options", "nosniff");
    
    try
    {
        // Validar input
        if (video == null || video.Length == 0)
        {
            return Results.BadRequest(new ProcessVideoResponse(
                false, 
                "No se proporcionó ningún video"
            ));
        }

        // Validar tamaño máximo (500MB)
        const long maxFileSize = 500 * 1024 * 1024;
        if (video.Length > maxFileSize)
        {
            return Results.BadRequest(new ProcessVideoResponse(
                false,
                "El video excede el tamaño máximo de 500MB"
            ));
        }

        // Crear directorios temporales
        var uploadsDir = Path.Combine(env.ContentRootPath, "uploads");
        var outputDir = Path.Combine(env.ContentRootPath, "outputs");
        
        Directory.CreateDirectory(uploadsDir);
        Directory.CreateDirectory(outputDir);

        // Guardar video temporal
        var videoFileName = $"{Guid.NewGuid()}{Path.GetExtension(video.FileName)}";
        var videoPath = Path.Combine(uploadsDir, videoFileName);
        
        using (var stream = new FileStream(videoPath, FileMode.Create))
        {
            await video.CopyToAsync(stream);
        }

        // Preparar paths para Python - deconstructed variable declaration
        var (modelsPath, outputPath, pythonScriptPath) = (
            Path.Combine(env.ContentRootPath, "..", "..", "..", "models"),
            Path.Combine(outputDir, $"{Path.GetFileNameWithoutExtension(videoFileName)}_results.json"),
            Path.Combine(env.ContentRootPath, "..", "..", "..", "python-scripts", "process_video.py")
        );

        Console.WriteLine($"🎾 Spike 4 API - Procesando video: {videoFileName}");
        Console.WriteLine($"📁 Video path: {videoPath}");
        Console.WriteLine($"🐍 Python script: {pythonScriptPath}");
        Console.WriteLine($"🤖 Models path: {modelsPath}");
        Console.WriteLine($"📊 Output path: {outputPath}");

        // Ejecutar script Python via subprocess
        var result = await ExecutePythonScript(pythonScriptPath, videoPath, outputPath, modelsPath);

        if (result.Success)
        {
            // Leer resultados JSON - using cached jsonOptions
            var resultJson = await File.ReadAllTextAsync(outputPath);
            var processingResult = JsonSerializer.Deserialize<PythonVideoResult>(resultJson, jsonOptions);

            return Results.Ok(new ProcessVideoResponse(
                true,
                "Video procesado exitosamente",
                processingResult
            ));
        }
        else
        {
            return Results.StatusCode(500);
        }
    }
    catch (Exception ex)
    {
        Console.WriteLine($"❌ Error en endpoint: {ex.Message}");
        return Results.StatusCode(500);
    }
})
.WithName("ProcessVideo")
.WithSummary("Procesa un video de pádel detectando jugadores con YOLO");

// Endpoint de health check
app.MapGet("/api/health", () => new { status = "healthy", timestamp = DateTime.UtcNow })
.WithName("HealthCheck")
.WithSummary("Health check endpoint");

app.Run();

// Response classes - declared after all top-level statements
record ProcessVideoResponse(bool Success, string Message, PythonVideoResult? Result = null);

record PythonVideoResult(
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
