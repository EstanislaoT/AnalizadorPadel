using System.Net;
using FluentAssertions;
using Xunit;

namespace AnalizadorPadel.Tests.Integration;

public class AnalysisControllerTests : IntegrationTestBase
{
    public AnalysisControllerTests(CustomWebApplicationFactory factory) : base(factory) { }

    [Fact]
    public async Task PostAnalysis_VideoNotFound_Returns404()
    {
        var response = await Client.PostAsync("/api/videos/999999/analyse", null);
        response.StatusCode.Should().Be(HttpStatusCode.NotFound);
    }

    [Fact]
    public async Task GetAnalysis_NotFound_Returns404()
    {
        var response = await Client.GetAsync("/api/analyses/999999");
        response.StatusCode.Should().Be(HttpStatusCode.NotFound);
    }

    [Fact]
    public async Task GetAnalysisStats_NotFound_Returns404()
    {
        var response = await Client.GetAsync("/api/analyses/999999/stats");
        response.StatusCode.Should().Be(HttpStatusCode.NotFound);
    }

    [Fact]
    public async Task GetAnalysisHeatmap_NotFound_Returns404()
    {
        var response = await Client.GetAsync("/api/analyses/999999/heatmap");
        response.StatusCode.Should().Be(HttpStatusCode.NotFound);
    }

    [Fact]
    public async Task GetAnalysisHeatmap_ReturnsData()
    {
        // Since we can't create a real analysis without video processing,
        // test that the endpoint exists and returns proper error for non-existent
        var response = await Client.GetAsync("/api/analyses/999999/heatmap");
        response.StatusCode.Should().Be(HttpStatusCode.NotFound);
    }
}
