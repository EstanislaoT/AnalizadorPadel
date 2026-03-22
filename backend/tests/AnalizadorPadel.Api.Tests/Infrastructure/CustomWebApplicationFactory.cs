using AnalizadorPadel.Api.Data;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.EntityFrameworkCore;
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

        builder.ConfigureServices(services =>
        {
            // Remove the existing DbContext registration
            var descriptor = services.SingleOrDefault(
                d => d.ServiceType == typeof(DbContextOptions<PadelDbContext>));
            if (descriptor != null)
            {
                services.Remove(descriptor);
            }

            var factoryDescriptor = services.SingleOrDefault(
                d => d.ServiceType == typeof(IDbContextFactory<PadelDbContext>));
            if (factoryDescriptor != null)
            {
                services.Remove(factoryDescriptor);
            }

            // Add in-memory database for testing
            services.AddDbContext<PadelDbContext>(options =>
                options.UseInMemoryDatabase($"TestDb_{Guid.NewGuid()}"));

            services.AddDbContextFactory<PadelDbContext>(options =>
                options.UseInMemoryDatabase($"TestDb_{Guid.NewGuid()}"), ServiceLifetime.Scoped);

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
