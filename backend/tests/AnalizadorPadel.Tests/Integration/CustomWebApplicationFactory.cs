using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using AnalizadorPadel.Api.Data;

namespace AnalizadorPadel.Tests.Integration;

public class CustomWebApplicationFactory : WebApplicationFactory<Program>
{
    protected override void ConfigureWebHost(IWebHostBuilder builder)
    {
        builder.UseEnvironment("Testing");
        
        builder.ConfigureServices(services =>
        {
            // Remove existing DbContext registrations
            var descriptors = services.Where(
                d => d.ServiceType == typeof(DbContextOptions<PadelDbContext>) ||
                     d.ServiceType == typeof(IDbContextFactory<PadelDbContext>))
                .ToList();
            
            foreach (var descriptor in descriptors)
            {
                services.Remove(descriptor);
            }

            // Add in-memory database for testing with unique name
            services.AddDbContext<PadelDbContext>(options =>
            {
                options.UseInMemoryDatabase($"TestDb_{Guid.NewGuid():N}");
            });

            services.AddDbContextFactory<PadelDbContext>(options =>
            {
                options.UseInMemoryDatabase($"TestDb_{Guid.NewGuid():N}");
            }, ServiceLifetime.Scoped);
        });
    }
}
