using AnalizadorPadel.Api.Data;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace AnalizadorPadel.Api.Tests.Infrastructure;

/// <summary>
/// Custom WebApplicationFactory for integration testing
/// Configures in-memory SQLite database and mocks external dependencies
/// </summary>
public class CustomWebApplicationFactory : WebApplicationFactory<Program>
{
    protected override void ConfigureWebHost(IWebHostBuilder builder)
    {
        builder.UseEnvironment("Testing");

        builder.ConfigureAppConfiguration((context, config) =>
        {
            // Override connection string for testing to use in-memory SQLite
            config.AddInMemoryCollection(new[]
            {
                new KeyValuePair<string, string?>("ConnectionStrings:DefaultConnection", ":memory:")
            });
        });

        builder.ConfigureServices(services =>
        {
            // Reduce logging verbosity for tests
            services.AddLogging(logging =>
            {
                logging.SetMinimumLevel(LogLevel.Warning);
                logging.ClearProviders();
            });
        });
    }

    /// <summary>
    /// Creates a new scope and returns the DbContext for database operations in tests
    /// </summary>
    public async Task<PadelDbContext> CreateDbContextAsync()
    {
        var scope = Services.CreateAsyncScope();
        var dbContext = scope.ServiceProvider.GetRequiredService<PadelDbContext>();
        await dbContext.Database.EnsureCreatedAsync();
        return dbContext;
    }
}
