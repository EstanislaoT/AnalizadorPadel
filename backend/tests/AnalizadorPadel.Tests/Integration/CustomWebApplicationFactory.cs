using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using AnalizadorPadel.Api.Data;

namespace AnalizadorPadel.Tests.Integration;

public class CustomWebApplicationFactory : WebApplicationFactory<Program>
{
    protected override IHost CreateHost(IHostBuilder builder)
    {
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

            // Add in-memory database for testing
            services.AddDbContext<PadelDbContext>(options =>
            {
                options.UseInMemoryDatabase("TestDb");
            });

            services.AddDbContextFactory<PadelDbContext>(options =>
            {
                options.UseInMemoryDatabase("TestDb");
            }, ServiceLifetime.Scoped);
        });

        return base.CreateHost(builder);
    }
}
