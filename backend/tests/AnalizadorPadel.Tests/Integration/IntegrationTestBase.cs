using System.Net.Http.Json;
using Xunit;

namespace AnalizadorPadel.Tests.Integration;

public abstract class IntegrationTestBase : IClassFixture<CustomWebApplicationFactory>, IDisposable
{
    protected readonly CustomWebApplicationFactory Factory;
    protected readonly HttpClient Client;

    protected IntegrationTestBase(CustomWebApplicationFactory factory)
    {
        Factory = factory;
        Client = factory.CreateClient();
    }

    public void Dispose()
    {
        Client.Dispose();
    }
}
