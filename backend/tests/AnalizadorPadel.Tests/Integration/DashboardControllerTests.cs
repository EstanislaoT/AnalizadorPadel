using System.Net;
using System.Net.Http.Json;
using FluentAssertions;
using Xunit;

namespace AnalizadorPadel.Tests.Integration;

public class DashboardControllerTests : IntegrationTestBase
{
    public DashboardControllerTests(CustomWebApplicationFactory factory) : base(factory) { }

    [Fact]
    public async Task GetDashboardStats_ReturnsAggregatedData()
    {
        var response = await Client.GetAsync("/api/dashboard/stats");
        
        response.StatusCode.Should().Be(HttpStatusCode.OK);
        
        var content = await response.Content.ReadAsStringAsync();
        content.Should().NotBeNullOrEmpty();
        // Verify it contains expected properties
        content.Should().ContainAny("totalVideos", "totalAnalyses", "recentVideos", "recentAnalyses");
    }

    [Fact]
    public async Task GetHealth_Returns200AndHealthy()
    {
        var response = await Client.GetAsync("/api/health");
        
        response.StatusCode.Should().Be(HttpStatusCode.OK);
    }
}
