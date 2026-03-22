using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc.Testing;

namespace AnalizadorPadel.Api.Tests.Infrastructure;

/// <summary>
/// Base class for all tests providing common utilities
/// </summary>
public abstract class TestBase : IDisposable
{
    protected TestBase()
    {
        // Setup that runs before each test
    }

    public virtual void Dispose()
    {
        // Cleanup that runs after each test
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Creates a temporary file path for test uploads
    /// </summary>
    protected string GetTempFilePath(string extension = ".mp4")
    {
        return Path.Combine(Path.GetTempPath(), $"test_{Guid.NewGuid()}{extension}");
    }

    /// <summary>
    /// Creates a mock FormFile for testing file uploads
    /// </summary>
    protected IFormFile CreateMockFormFile(string fileName, long fileSize, string contentType = "video/mp4")
    {
        var stream = new MemoryStream(new byte[fileSize]);
        return new FormFile(stream, 0, fileSize, "file", fileName)
        {
            Headers = new HeaderDictionary(),
            ContentType = contentType
        };
    }
}

/// <summary>
/// Base class for integration tests using CustomWebApplicationFactory
/// </summary>
public abstract class IntegrationTestBase : TestBase, IClassFixture<CustomWebApplicationFactory>
{
    protected readonly CustomWebApplicationFactory Factory;
    protected readonly HttpClient Client;

    protected IntegrationTestBase(CustomWebApplicationFactory factory)
    {
        Factory = factory;
        Client = factory.CreateClient(new WebApplicationFactoryClientOptions
        {
            AllowAutoRedirect = false
        });
    }

    public override void Dispose()
    {
        Client.Dispose();
        base.Dispose();
    }
}
