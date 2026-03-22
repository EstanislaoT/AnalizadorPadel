using AnalizadorPadel.Api.Data;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Data.Sqlite;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace AnalizadorPadel.Api.Tests.Infrastructure;

/// <summary>
/// Custom WebApplicationFactory for integration testing.
/// Uses an isolated SQLite file and resets persisted state before each test.
/// </summary>
public class CustomWebApplicationFactory : WebApplicationFactory<Program>
{
    private readonly string _databasePath = Path.Combine(Path.GetTempPath(), $"padel-tests-{Guid.NewGuid():N}.db");

    protected override void ConfigureWebHost(IWebHostBuilder builder)
    {
        builder.UseEnvironment("Testing");

        builder.ConfigureServices(services =>
        {
            RemoveExistingDbRegistrations(services);

            var connectionString = new SqliteConnectionStringBuilder
            {
                DataSource = _databasePath
            }.ToString();

            services.AddDbContext<PadelDbContext>(options => options.UseSqlite(connectionString));
            services.AddDbContextFactory<PadelDbContext>(options => options.UseSqlite(connectionString), ServiceLifetime.Scoped);

            services.AddLogging(logging =>
            {
                logging.SetMinimumLevel(LogLevel.Warning);
                logging.ClearProviders();
            });
        });
    }

    public async Task ResetStateAsync()
    {
        await using var scope = Services.CreateAsyncScope();
        var dbContext = scope.ServiceProvider.GetRequiredService<PadelDbContext>();

        await dbContext.Database.EnsureDeletedAsync();
        await dbContext.Database.EnsureCreatedAsync();

        var env = scope.ServiceProvider.GetRequiredService<IWebHostEnvironment>();
        DeleteFiles(Path.Combine(env.ContentRootPath, "uploads"));
        DeleteFiles(Path.Combine(env.ContentRootPath, "outputs"));
    }

    protected override void Dispose(bool disposing)
    {
        base.Dispose(disposing);

        if (disposing && File.Exists(_databasePath))
        {
            File.Delete(_databasePath);
        }
    }

    private static void RemoveExistingDbRegistrations(IServiceCollection services)
    {
        var descriptors = services.Where(descriptor =>
                descriptor.ServiceType == typeof(DbContextOptions<PadelDbContext>) ||
                descriptor.ServiceType == typeof(PadelDbContext) ||
                descriptor.ServiceType == typeof(IDbContextFactory<PadelDbContext>))
            .ToList();

        foreach (var descriptor in descriptors)
        {
            services.Remove(descriptor);
        }
    }

    private static void DeleteFiles(string directoryPath)
    {
        if (!Directory.Exists(directoryPath))
        {
            return;
        }

        foreach (var file in Directory.EnumerateFiles(directoryPath))
        {
            File.Delete(file);
        }
    }
}
