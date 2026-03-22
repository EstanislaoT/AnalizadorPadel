using System.Net;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using AnalizadorPadel.Api.Models.DTOs;
using AnalizadorPadel.Api.Tests.Infrastructure;
using TechTalk.SpecFlow;

namespace AnalizadorPadel.Api.Tests.BDD.StepDefinitions;

[Binding]
public class VideoSteps : IntegrationTestBase
{
    private HttpResponseMessage? _response;
    private ApiResponse<VideoDto>? _videoResponse;
    private ApiResponse<List<VideoDto>>? _videosListResponse;

    public VideoSteps(CustomWebApplicationFactory factory) : base(factory)
    {
    }

    [Given("la API está funcionando correctamente")]
    public void GivenLaAPIEstaFuncionandoCorrectamente()
    {
        // Verify API is running via health check
        var response = Client.GetAsync("/api/health").Result;
        response.StatusCode.Should().Be(HttpStatusCode.OK);
    }

    [Given("el sistema de almacenamiento está disponible")]
    public void GivenElSistemaDeAlmacenamientoEstaDisponible()
    {
        // Storage is always available in test environment
    }

    [Given("el usuario tiene un video en formato MP4")]
    public void GivenElUsuarioTieneUnVideoEnFormatoMP4()
    {
        // Preparation step - no action needed
    }

    [Given("el usuario tiene un video en formato (.*)")]
    public void GivenElUsuarioTieneUnVideoEnFormato(string formato)
    {
        // Preparation step - no action needed
    }

    [Given("el usuario tiene un video de (\\d+) MB")]
    public void GivenElUsuarioTieneUnVideoDeMB(int tamaño)
    {
        // Preparation step - video size will be handled during upload
    }

    [Given("el usuario ha subido previamente (\\d+) videos")]
    public async Task GivenElUsuarioHaSubidoPreviamenteVideos(int count)
    {
        for (int i = 0; i < count; i++)
        {
            await CreateTestVideoAsync($"video{i + 1}.mp4");
        }
    }

    [Given("existe un video con ID (\\d+) en el sistema")]
    public async Task GivenExisteUnVideoConIDEnElSistema(int id)
    {
        // Create a video that will get the specified ID
        var video = await CreateTestVideoAsync("test_video.mp4");
        video.Data!.Id.Should().BeGreaterThan(0);
    }

    [Given("no existe un video con ID (\\d+) en el sistema")]
    public void GivenNoExisteUnVideoConIDEnElSistema(int id)
    {
        // Ensure video doesn't exist - no action needed as we start with empty DB
    }

    [When("el usuario sube el video \"(.*)\"")]
    public async Task WhenElUsuarioSubeElVideo(string fileName)
    {
        _response = await CreateTestVideoAsync(fileName);
        if (_response.IsSuccessStatusCode)
        {
            _videoResponse = await _response.Content.ReadFromJsonAsync<ApiResponse<VideoDto>>();
        }
    }

    [When("el usuario intenta subir el video grande")]
    public async Task WhenElUsuarioIntentaSubirElVideoGrande()
    {
        var content = new MultipartFormDataContent();
        var largeStream = new MemoryStream(new byte[501 * 1024 * 1024]); // 501MB
        var fileContent = new StreamContent(largeStream);
        fileContent.Headers.ContentType = new MediaTypeHeaderValue("video/mp4");
        content.Add(fileContent, "file", "large_video.mp4");

        _response = await Client.PostAsync("/api/videos", content);
    }

    [When("el usuario envía la petición sin archivo adjunto")]
    public async Task WhenElUsuarioEnviaLaPeticionSinArchivoAdjunto()
    {
        var content = new MultipartFormDataContent();
        _response = await Client.PostAsync("/api/videos", content);
    }

    [When("el usuario solicita la lista de videos")]
    public async Task WhenElUsuarioSolicitaLaListaDeVideos()
    {
        _response = await Client.GetAsync("/api/videos");
        if (_response.IsSuccessStatusCode)
        {
            _videosListResponse = await _response.Content.ReadFromJsonAsync<ApiResponse<List<VideoDto>>>();
        }
    }

    [When("el usuario elimina el video con ID (\\d+)")]
    public async Task WhenElUsuarioEliminaElVideoConID(int id)
    {
        _response = await Client.DeleteAsync($"/api/videos/{id}");
    }

    [When("el usuario intenta eliminar el video con ID (\\d+)")]
    public async Task WhenElUsuarioIntentaEliminarElVideoConID(int id)
    {
        _response = await Client.DeleteAsync($"/api/videos/{id}");
    }

    [Then("el sistema responde con código (\\d+) (.*)")]
    public void ThenElSistemaRespondeConCodigo(int expectedCode, string description)
    {
        _response.Should().NotBeNull();
        _response!.StatusCode.Should().Be((HttpStatusCode)expectedCode);
    }

    [Then("el video se almacena en el sistema")]
    public void ThenElVideoSeAlmacenaEnElSistema()
    {
        _videoResponse.Should().NotBeNull();
        _videoResponse!.Data.Should().NotBeNull();
        _videoResponse.Data!.Id.Should().BeGreaterThan(0);
    }

    [Then("el video se almacena correctamente")]
    public void ThenElVideoSeAlmacenaCorrectamente()
    {
        _videoResponse.Should().NotBeNull();
        _videoResponse!.Success.Should().BeTrue();
    }

    [Then("el sistema devuelve los detalles del video incluyendo un ID único")]
    public void ThenElSistemaDevuelveLosDetallesDelVideoIncluyendoUnIDUnico()
    {
        _videoResponse.Should().NotBeNull();
        _videoResponse!.Data.Should().NotBeNull();
        _videoResponse.Data!.Id.Should().BeGreaterThan(0);
    }

    [Then("el estado del video es \"(.*)\"")]
    public void ThenElEstadoDelVideoEs(string expectedStatus)
    {
        _videoResponse.Should().NotBeNull();
        _videoResponse!.Data.Should().NotBeNull();
        _videoResponse.Data!.Status.ToString().Should().Be(expectedStatus);
    }

    [Then("el mensaje de error indica \"(.*)\"")]
    public async Task ThenElMensajeDeErrorIndica(string expectedMessage)
    {
        _response.Should().NotBeNull();
        var content = await _response!.Content.ReadFromJsonAsync<ApiResponse<object>>();
        content.Should().NotBeNull();
        content!.Message.Should().Contain(expectedMessage);
    }

    [Then("el video no se almacena en el sistema")]
    public void ThenElVideoNoSeAlmacenaEnElSistema()
    {
        // The response should indicate failure
        _response.Should().NotBeNull();
        _response!.StatusCode.Should().Be(HttpStatusCode.BadRequest);
    }

    [Then("la lista contiene (\\d+) videos")]
    public void ThenLaListaContieneVideos(int expectedCount)
    {
        _videosListResponse.Should().NotBeNull();
        _videosListResponse!.Data.Should().NotBeNull();
        _videosListResponse.Data!.Count.Should().Be(expectedCount);
    }

    [Then("el video más reciente aparece primero en la lista")]
    public void ThenElVideoMasRecienteAparecePrimeroEnLaLista()
    {
        _videosListResponse.Should().NotBeNull();
        _videosListResponse!.Data.Should().NotBeNull();
        var videos = _videosListResponse.Data!;

        if (videos.Count >= 2)
        {
            videos[0].UploadedAt.Should().BeOnOrAfter(videos[1].UploadedAt);
        }
    }

    [Then("el mensaje confirma \"(.*)\"")]
    public async Task ThenElMensajeConfirma(string expectedMessage)
    {
        var content = await _response!.Content.ReadFromJsonAsync<ApiResponse<object>>();
        content.Should().NotBeNull();
        content!.Message.Should().Contain(expectedMessage);
    }

    [Then("el video ya no aparece en la lista de videos")]
    public async Task ThenElVideoYaNoApareceEnLaListaDeVideos()
    {
        var response = await Client.GetAsync("/api/videos");
        var videos = await response.Content.ReadFromJsonAsync<ApiResponse<List<VideoDto>>>();
        videos!.Data!.Should().BeEmpty();
    }

    private async Task<HttpResponseMessage> CreateTestVideoAsync(string fileName)
    {
        var content = new MultipartFormDataContent();
        var fileContent = new StreamContent(new MemoryStream(new byte[1024]));
        fileContent.Headers.ContentType = new MediaTypeHeaderValue("video/mp4");
        content.Add(fileContent, "file", fileName);

        return await Client.PostAsync("/api/videos", content);
    }
}
