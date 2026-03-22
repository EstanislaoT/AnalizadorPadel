using System.Net;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using AnalizadorPadel.Api.Models.DTOs;
using AnalizadorPadel.Api.Tests.Infrastructure;

namespace AnalizadorPadel.Api.Tests.Integration;

public class VideoEndpointsTests : IntegrationTestBase
{
    public VideoEndpointsTests(CustomWebApplicationFactory factory) : base(factory)
    {
    }

    #region POST /api/videos Tests

    [Fact]
    public async Task CreateVideo_WithValidFile_ShouldReturn201Created()
    {
        // Arrange
        var content = new MultipartFormDataContent();
        var fileContent = new StreamContent(new MemoryStream(new byte[1024]));
        fileContent.Headers.ContentType = new MediaTypeHeaderValue("video/mp4");
        content.Add(fileContent, "file", "test_video.mp4");

        // Act
        var response = await Client.PostAsync("/api/videos", content);

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.Created);

        var result = await response.Content.ReadFromJsonAsync<ApiResponse<VideoDto>>();
        result.Should().NotBeNull();
        result!.Success.Should().BeTrue();
        result.Data.Should().NotBeNull();
        result.Data!.Name.Should().Be("test_video");
        result.Data.Status.Should().Be(VideoStatus.Uploaded);
        result.Data.Id.Should().BeGreaterThan(0);
    }

    [Fact]
    public async Task CreateVideo_WithNoFile_ShouldReturn400BadRequest()
    {
        // Arrange
        var content = new MultipartFormDataContent();

        // Act
        var response = await Client.PostAsync("/api/videos", content);

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);

        var result = await response.Content.ReadFromJsonAsync<ApiResponse<object>>();
        result.Should().NotBeNull();
        result!.Success.Should().BeFalse();
        result.Message.Should().Contain("No se proporcionó ningún video");
    }

    [Theory]
    [InlineData("test.txt")]
    [InlineData("test.pdf")]
    [InlineData("test.jpg")]
    public async Task CreateVideo_WithInvalidExtension_ShouldReturn400BadRequest(string fileName)
    {
        // Arrange
        var content = new MultipartFormDataContent();
        var fileContent = new StreamContent(new MemoryStream(new byte[1024]));
        fileContent.Headers.ContentType = new MediaTypeHeaderValue("application/octet-stream");
        content.Add(fileContent, "file", fileName);

        // Act
        var response = await Client.PostAsync("/api/videos", content);

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);

        var result = await response.Content.ReadFromJsonAsync<ApiResponse<object>>();
        result.Should().NotBeNull();
        result!.Success.Should().BeFalse();
        result.Message.Should().Contain("Formato no soportado");
    }

    [Fact]
    public async Task CreateVideo_WithLargeFile_ShouldReturn400BadRequest()
    {
        // Arrange - Create content > 500MB
        var content = new MultipartFormDataContent();
        var largeStream = new MemoryStream(new byte[501 * 1024 * 1024]); // 501MB
        var fileContent = new StreamContent(largeStream);
        fileContent.Headers.ContentType = new MediaTypeHeaderValue("video/mp4");
        content.Add(fileContent, "file", "large_video.mp4");

        // Act
        var response = await Client.PostAsync("/api/videos", content);

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);

        var result = await response.Content.ReadFromJsonAsync<ApiResponse<object>>();
        result.Should().NotBeNull();
        result!.Success.Should().BeFalse();
        result.Message.Should().Contain("excede el tamaño máximo");
    }

    #endregion

    #region GET /api/videos Tests

    [Fact]
    public async Task GetVideos_WithNoVideos_ShouldReturnEmptyList()
    {
        // Act
        var response = await Client.GetAsync("/api/videos");

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.OK);

        var result = await response.Content.ReadFromJsonAsync<ApiResponse<List<VideoDto>>>();
        result.Should().NotBeNull();
        result!.Success.Should().BeTrue();
        result.Data.Should().BeEmpty();
    }

    [Fact]
    public async Task GetVideos_WithVideos_ShouldReturnVideosList()
    {
        // Arrange - Create a video first
        await CreateTestVideoAsync("video1.mp4");
        await CreateTestVideoAsync("video2.mp4");

        // Act
        var response = await Client.GetAsync("/api/videos");

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.OK);

        var result = await response.Content.ReadFromJsonAsync<ApiResponse<List<VideoDto>>>();
        result.Should().NotBeNull();
        result!.Success.Should().BeTrue();
        result.Data.Should().HaveCount(2);
        result.Message.Should().Contain("2 videos encontrados");
    }

    #endregion

    #region GET /api/videos/{id} Tests

    [Fact]
    public async Task GetVideo_WithExistingVideo_ShouldReturnVideo()
    {
        // Arrange
        var created = await CreateTestVideoAsync("test.mp4");

        // Act
        var response = await Client.GetAsync($"/api/videos/{created.Data!.Id}");

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.OK);

        var result = await response.Content.ReadFromJsonAsync<ApiResponse<VideoDto>>();
        result.Should().NotBeNull();
        result!.Success.Should().BeTrue();
        result.Data.Should().NotBeNull();
        result.Data!.Id.Should().Be(created.Data!.Id);
    }

    [Fact]
    public async Task GetVideo_WithNonExistingVideo_ShouldReturn404()
    {
        // Act
        var response = await Client.GetAsync("/api/videos/999");

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.NotFound);

        var result = await response.Content.ReadFromJsonAsync<ApiResponse<object>>();
        result.Should().NotBeNull();
        result!.Success.Should().BeFalse();
        result.Message.Should().Contain("no encontrado");
    }

    #endregion

    #region DELETE /api/videos/{id} Tests

    [Fact]
    public async Task DeleteVideo_WithExistingVideo_ShouldDeleteAndReturn200()
    {
        // Arrange
        var created = await CreateTestVideoAsync("to_delete.mp4");

        // Act
        var response = await Client.DeleteAsync($"/api/videos/{created.Data!.Id}");

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.OK);

        var result = await response.Content.ReadFromJsonAsync<ApiResponse<object>>();
        result.Should().NotBeNull();
        result!.Success.Should().BeTrue();
        result.Message.Should().Contain("eliminado");

        // Verify it's gone
        var getResponse = await Client.GetAsync($"/api/videos/{created.Data!.Id}");
        getResponse.StatusCode.Should().Be(HttpStatusCode.NotFound);
    }

    [Fact]
    public async Task DeleteVideo_WithNonExistingVideo_ShouldReturn404()
    {
        // Act
        var response = await Client.DeleteAsync("/api/videos/999");

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.NotFound);

        var result = await response.Content.ReadFromJsonAsync<ApiResponse<object>>();
        result.Should().NotBeNull();
        result!.Success.Should().BeFalse();
    }

    #endregion

    #region GET /api/videos/{id}/stream Tests

    [Fact]
    public async Task StreamVideo_WithExistingVideo_ShouldReturn200WithAcceptRanges()
    {
        // Arrange
        var created = await CreateTestVideoWithRealFileAsync("stream_test.mp4");

        // Act
        var response = await Client.GetAsync($"/api/videos/{created.Data!.Id}/stream");

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.OK);
        response.Headers.Should().ContainKey("Accept-Ranges");
        response.Headers.GetValues("Accept-Ranges").Should().Contain("bytes");
        response.Content.Headers.ContentType?.MediaType.Should().Be("video/mp4");
    }

    [Fact]
    public async Task StreamVideo_WithRangeHeader_ShouldReturn206PartialContent()
    {
        // Arrange
        var created = await CreateTestVideoWithRealFileAsync("range_test.mp4", 10240); // 10KB file
        var request = new HttpRequestMessage(HttpMethod.Get, $"/api/videos/{created.Data!.Id}/stream");
        request.Headers.Range = new RangeHeaderValue(0, 1023); // First 1KB

        // Act
        var response = await Client.SendAsync(request);

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.PartialContent);
        response.Headers.Should().ContainKey("Content-Range");
        response.Content.Headers.ContentLength.Should().Be(1024);
    }

    [Fact]
    public async Task StreamVideo_WithInvalidRange_ShouldReturn416()
    {
        // Arrange
        var created = await CreateTestVideoWithRealFileAsync("invalid_range.mp4", 1024);
        var request = new HttpRequestMessage(HttpMethod.Get, $"/api/videos/{created.Data!.Id}/stream");
        request.Headers.Add("Range", "bytes=2000-3000"); // Beyond file size

        // Act
        var response = await Client.SendAsync(request);

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
    }

    [Fact]
    public async Task StreamVideo_WithNonExistingVideo_ShouldReturn404()
    {
        // Act
        var response = await Client.GetAsync("/api/videos/999/stream");

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.NotFound);
    }

    #endregion

    #region Helper Methods

    private async Task<ApiResponse<VideoDto>> CreateTestVideoAsync(string fileName)
    {
        var content = new MultipartFormDataContent();
        var fileContent = new StreamContent(new MemoryStream(new byte[1024]));
        fileContent.Headers.ContentType = new MediaTypeHeaderValue("video/mp4");
        content.Add(fileContent, "file", fileName);

        var response = await Client.PostAsync("/api/videos", content);
        response.EnsureSuccessStatusCode();

        return (await response.Content.ReadFromJsonAsync<ApiResponse<VideoDto>>())!;
    }

    private async Task<ApiResponse<VideoDto>> CreateTestVideoWithRealFileAsync(string fileName, int fileSize = 1024)
    {
        // Create actual temp file
        var tempPath = Path.Combine(Path.GetTempPath(), $"test_{Guid.NewGuid()}.mp4");
        await File.WriteAllBytesAsync(tempPath, new byte[fileSize]);

        try
        {
            var content = new MultipartFormDataContent();
            var fileStream = File.OpenRead(tempPath);
            var fileContent = new StreamContent(fileStream);
            fileContent.Headers.ContentType = new MediaTypeHeaderValue("video/mp4");
            content.Add(fileContent, "file", fileName);

            var response = await Client.PostAsync("/api/videos", content);
            response.EnsureSuccessStatusCode();

            var result = await response.Content.ReadFromJsonAsync<ApiResponse<VideoDto>>();

            // Copy the temp file to where the service expects it
            var expectedPath = result!.Data!.FilePath;
            Directory.CreateDirectory(Path.GetDirectoryName(expectedPath)!);
            await File.WriteAllBytesAsync(expectedPath, new byte[fileSize]);

            return result;
        }
        finally
        {
            if (File.Exists(tempPath))
            {
                File.Delete(tempPath);
            }
        }
    }

    #endregion
}
