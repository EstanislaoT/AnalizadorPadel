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
    private const string ResponseKey = "HttpResponse";
    private const string VideoResponseKey = "VideoResponse";
    private const string VideosListResponseKey = "VideosListResponse";

    private readonly ScenarioContext _scenarioContext;

    private HttpResponseMessage? Response
    {
        get => _scenarioContext.TryGetValue<HttpResponseMessage>(ResponseKey, out var response) ? response : null;
        set => _scenarioContext[ResponseKey] = value!;
    }

    private ApiResponse<VideoDto>? VideoResponse
    {
        get => _scenarioContext.TryGetValue<ApiResponse<VideoDto>>(VideoResponseKey, out var response) ? response : null;
        set => _scenarioContext[VideoResponseKey] = value!;
    }

    private ApiResponse<List<VideoDto>>? VideosListResponse
    {
        get => _scenarioContext.TryGetValue<ApiResponse<List<VideoDto>>>(VideosListResponseKey, out var response) ? response : null;
        set => _scenarioContext[VideosListResponseKey] = value!;
    }

    public VideoSteps(CustomWebApplicationFactory factory, ScenarioContext scenarioContext) : base(factory)
    {
        _scenarioContext = scenarioContext;
    }

    [Given("la API está funcionando correctamente")]
    public void GivenLaAPIEstaFuncionandoCorrectamente()
    {
        var response = Client.GetAsync("/api/health").Result;
        response.StatusCode.Should().Be(HttpStatusCode.OK);
    }

    [Given("el sistema de almacenamiento está disponible")]
    public void GivenElSistemaDeAlmacenamientoEstaDisponible()
    {
    }

    [Given("el usuario tiene un video en formato (.*)")]
    public void GivenElUsuarioTieneUnVideoEnFormato(string formato)
    {
    }

    [Given("el usuario tiene un video de (\\d+) MB")]
    public void GivenElUsuarioTieneUnVideoDeMB(int tamaño)
    {
    }

    [Given("el usuario intenta subir un video")]
    public void GivenElUsuarioIntentaSubirUnVideo()
    {
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
        var response = await CreateTestVideoAsync("test_video.mp4");
        var video = await response.Content.ReadFromJsonAsync<ApiResponse<VideoDto>>();
        video.Should().NotBeNull();
        video!.Data.Should().NotBeNull();
        video.Data!.Id.Should().BeGreaterThan(0);
    }

    [Given("no existe un video con ID (\\d+) en el sistema")]
    public void GivenNoExisteUnVideoConIDEnElSistema(int id)
    {
    }

    [When("el usuario sube el video \"(.*)\"")]
    public async Task WhenElUsuarioSubeElVideo(string fileName)
    {
        Response = await CreateTestVideoAsync(fileName);
        if (Response.IsSuccessStatusCode)
        {
            VideoResponse = await Response.Content.ReadFromJsonAsync<ApiResponse<VideoDto>>();
        }
    }

    [When("el usuario sube un nuevo video \"(.*)\"")]
    public Task WhenElUsuarioSubeUnNuevoVideo(string fileName)
    {
        return WhenElUsuarioSubeElVideo(fileName);
    }

    [When("el usuario intenta subir el video grande")]
    public async Task WhenElUsuarioIntentaSubirElVideoGrande()
    {
        var content = new MultipartFormDataContent();
        var largeStream = new MemoryStream(new byte[501 * 1024 * 1024]);
        var fileContent = new StreamContent(largeStream);
        fileContent.Headers.ContentType = new MediaTypeHeaderValue("video/mp4");
        content.Add(fileContent, "file", "large_video.mp4");

        Response = await Client.PostAsync("/api/videos", content);
    }

    [When("el usuario envía la petición sin archivo adjunto")]
    public async Task WhenElUsuarioEnviaLaPeticionSinArchivoAdjunto()
    {
        var content = new MultipartFormDataContent();
        Response = await Client.PostAsync("/api/videos", content);
    }

    [When("el usuario solicita la lista de videos")]
    public async Task WhenElUsuarioSolicitaLaListaDeVideos()
    {
        Response = await Client.GetAsync("/api/videos");
        if (Response.IsSuccessStatusCode)
        {
            VideosListResponse = await Response.Content.ReadFromJsonAsync<ApiResponse<List<VideoDto>>>();
        }
    }

    [When("el usuario elimina el video con ID (\\d+)")]
    public async Task WhenElUsuarioEliminaElVideoConID(int id)
    {
        Response = await Client.DeleteAsync($"/api/videos/{id}");
    }

    [When("el usuario intenta eliminar el video con ID (\\d+)")]
    public async Task WhenElUsuarioIntentaEliminarElVideoConID(int id)
    {
        Response = await Client.DeleteAsync($"/api/videos/{id}");
    }

    [Then("el sistema responde con código (\\d+) (.*)")]
    public void ThenElSistemaRespondeConCodigo(int expectedCode, string description)
    {
        Response.Should().NotBeNull();
        Response!.StatusCode.Should().Be((HttpStatusCode)expectedCode);
    }

    [Then("el video se almacena en el sistema")]
    public void ThenElVideoSeAlmacenaEnElSistema()
    {
        VideoResponse.Should().NotBeNull();
        VideoResponse!.Data.Should().NotBeNull();
        VideoResponse.Data!.Id.Should().BeGreaterThan(0);
    }

    [Then("el video se almacena correctamente")]
    public void ThenElVideoSeAlmacenaCorrectamente()
    {
        VideoResponse.Should().NotBeNull();
        VideoResponse!.Success.Should().BeTrue();
    }

    [Then("el sistema devuelve los detalles del video incluyendo un ID único")]
    public void ThenElSistemaDevuelveLosDetallesDelVideoIncluyendoUnIDUnico()
    {
        VideoResponse.Should().NotBeNull();
        VideoResponse!.Data.Should().NotBeNull();
        VideoResponse.Data!.Id.Should().BeGreaterThan(0);
    }

    [Then("el estado del video es \"(.*)\"")]
    public void ThenElEstadoDelVideoEs(string expectedStatus)
    {
        VideoResponse.Should().NotBeNull();
        VideoResponse!.Data.Should().NotBeNull();
        VideoResponse.Data!.Status.ToString().Should().Be(expectedStatus);
    }

    [Then("el mensaje de error indica \"(.*)\"")]
    public async Task ThenElMensajeDeErrorIndica(string expectedMessage)
    {
        Response.Should().NotBeNull();
        var content = await Response!.Content.ReadFromJsonAsync<ApiResponse<object>>();
        content.Should().NotBeNull();
        content!.Message.Should().Contain(expectedMessage);
    }

    [Then("el video no se almacena en el sistema")]
    public void ThenElVideoNoSeAlmacenaEnElSistema()
    {
        Response.Should().NotBeNull();
        Response!.StatusCode.Should().Be(HttpStatusCode.BadRequest);
    }

    [Then("la lista contiene (\\d+) videos")]
    public void ThenLaListaContieneVideos(int expectedCount)
    {
        VideosListResponse.Should().NotBeNull();
        VideosListResponse!.Data.Should().NotBeNull();
        VideosListResponse.Data!.Count.Should().Be(expectedCount);
    }

    [Then("el video más reciente aparece primero en la lista")]
    public void ThenElVideoMasRecienteAparecePrimeroEnLaLista()
    {
        VideosListResponse.Should().NotBeNull();
        VideosListResponse!.Data.Should().NotBeNull();
        var videos = VideosListResponse.Data!;

        if (videos.Count >= 2)
        {
            videos[0].UploadedAt.Should().BeOnOrAfter(videos[1].UploadedAt);
        }
    }

    [Then("el mensaje confirma \"(.*)\"")]
    public async Task ThenElMensajeConfirma(string expectedMessage)
    {
        Response.Should().NotBeNull();
        var content = await Response!.Content.ReadFromJsonAsync<ApiResponse<object>>();
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
