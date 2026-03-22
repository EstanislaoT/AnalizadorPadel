using System.Net;
using System.Net.Http.Headers;
using FluentAssertions;
using TechTalk.SpecFlow;

namespace AnalizadorPadel.Tests.BDD.Steps;

[Binding]
public class VideoUploadSteps
{
    private readonly ScenarioContext _scenarioContext;
    private HttpResponseMessage? _response;
    private string? _fileName;

    public VideoUploadSteps(ScenarioContext scenarioContext)
    {
        _scenarioContext = scenarioContext;
    }

    private HttpClient Client => (HttpClient)_scenarioContext["Client"];

    [Given("the API is running")]
    public void GivenTheAPIIsRunning()
    {
        Client.Should().NotBeNull("Client should be initialized");
    }

    [Given("I have a valid MP4 video file named {string}")]
    public void GivenIHaveAValidMP4VideoFile(string fileName)
    {
        _fileName = fileName;
    }

    [Given("I have a text file named {string}")]
    public void GivenIHaveATextFile(string fileName)
    {
        _fileName = fileName;
    }

    [When("I submit the video to the upload endpoint")]
    [When("I submit the file to the upload endpoint")]
    public async Task WhenISubmitTheVideoToTheUploadEndpoint()
    {
        var content = new MultipartFormDataContent();
        var fileContent = new StringContent("fake video content");

        if (_fileName?.EndsWith(".mp4") == true)
            fileContent.Headers.ContentType = new MediaTypeHeaderValue("video/mp4");
        else
            fileContent.Headers.ContentType = new MediaTypeHeaderValue("text/plain");

        content.Add(fileContent, "file", _fileName!);

        _response = await Client.PostAsync("/api/videos", content);
        _scenarioContext["Response"] = _response;
    }

    [Then("I should receive a {int} Created response")]
    [Then("I should receive a {int} OK response")]
    public void ThenIShouldReceiveAResponse(int statusCode)
    {
        _response!.StatusCode.Should().Be((HttpStatusCode)statusCode);
    }

    [Then("the response should contain a video ID")]
    public async Task ThenTheResponseShouldContainAVideoID()
    {
        var content = await _response!.Content.ReadAsStringAsync();
        content.Should().Contain("id");
    }

    [Then("the video metadata should include file size and duration")]
    public async Task ThenTheVideoMetadataShouldIncludeFileSizeAndDuration()
    {
        var content = await _response!.Content.ReadAsStringAsync();
        content.Should().MatchRegex("(fileSize|file_size|duration)");
    }

    [Then("I should receive a {int} Bad Request response")]
    public void ThenIShouldReceiveABadRequestResponse(int statusCode)
    {
        _response!.StatusCode.Should().Be((HttpStatusCode)statusCode);
    }

    [Then("the error message should indicate {string}")]
    public async Task ThenTheErrorMessageShouldIndicate(string message)
    {
        var content = await _response!.Content.ReadAsStringAsync();
        content.ToLower().Should().Contain(message.ToLower());
    }

    [Given("I have a video file larger than 500MB")]
    public void GivenIHaveAVideoFileLargerThan500MB()
    {
        _fileName = "large-video.mp4";
    }

    [Given("I have uploaded a video")]
    public async Task GivenIHaveUploadedAVideo()
    {
        var content = new MultipartFormDataContent();
        var fileContent = new StringContent("test video");
        fileContent.Headers.ContentType = new MediaTypeHeaderValue("video/mp4");
        content.Add(fileContent, "file", "test.mp4");

        _response = await Client.PostAsync("/api/videos", content);
        _response.StatusCode.Should().Be(HttpStatusCode.Created);
    }

    [When("I request the video list")]
    public async Task WhenIRequestTheVideoList()
    {
        _response = await Client.GetAsync("/api/videos");
        _scenarioContext["Response"] = _response;
    }

    [Then("the list should contain at least {int} video")]
    public async Task ThenTheListShouldContainAtLeastVideo(int count)
    {
        var content = await _response!.Content.ReadAsStringAsync();
        content.Should().NotBeNullOrEmpty();
    }
}
