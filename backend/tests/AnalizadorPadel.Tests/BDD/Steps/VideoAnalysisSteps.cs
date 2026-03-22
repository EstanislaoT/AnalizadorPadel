using System.Net;
using FluentAssertions;
using TechTalk.SpecFlow;

namespace AnalizadorPadel.Tests.BDD.Steps;

[Binding]
public class VideoAnalysisSteps
{
    private readonly ScenarioContext _scenarioContext;
    private HttpResponseMessage? _response;
    private string? _videoId;
    private string? _analysisId;

    public VideoAnalysisSteps(ScenarioContext scenarioContext)
    {
        _scenarioContext = scenarioContext;
    }

    private HttpClient Client => (HttpClient)_scenarioContext["Client"];

    [Given("I have uploaded a video file")]
    public async Task GivenIHaveUploadedAVideoFile()
    {
        var content = new MultipartFormDataContent();
        var fileContent = new StringContent("test video");
        fileContent.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("video/mp4");
        content.Add(fileContent, "file", "test.mp4");

        var response = await Client.PostAsync("/api/videos", content);
        response.StatusCode.Should().Be(HttpStatusCode.Created);

        _videoId = "1";
    }

    [Given("I have an uploaded video with ID")]
    public void GivenIHaveAnUploadedVideoWithID()
    {
        _videoId = "1";
    }

    [When("I request analysis for the video")]
    public async Task WhenIRequestAnalysisForTheVideo()
    {
        _response = await Client.PostAsync($"/api/videos/{_videoId}/analyse", null);
        _scenarioContext["Response"] = _response;
    }

    [Then("the analysis status should be {string}")]
    public async Task ThenTheAnalysisStatusShouldBe(string status)
    {
        var content = await _response!.Content.ReadAsStringAsync();
        content.ToLower().Should().Contain(status.ToLower());
    }

    [Given("I have a completed analysis")]
    public void GivenIHaveACompletedAnalysis()
    {
        _analysisId = "1";
    }

    [When("I request the analysis results")]
    public async Task WhenIRequestTheAnalysisResults()
    {
        _response = await Client.GetAsync($"/api/analyses/{_analysisId}");
        _scenarioContext["Response"] = _response;
    }

    [Then("the response should contain match statistics")]
    public async Task ThenTheResponseShouldContainMatchStatistics()
    {
        var content = await _response!.Content.ReadAsStringAsync();
        content.Should().NotBeNullOrEmpty();
    }

    [Then("the response should contain player positions")]
    public async Task ThenTheResponseShouldContainPlayerPositions()
    {
        var content = await _response!.Content.ReadAsStringAsync();
        content.Should().NotBeNullOrEmpty();
    }

    [When("I request the analysis statistics")]
    public async Task WhenIRequestTheAnalysisStatistics()
    {
        _response = await Client.GetAsync($"/api/analyses/{_analysisId}/stats");
        _scenarioContext["Response"] = _response;
    }

    [Then("the statistics should include total frames")]
    public async Task ThenTheStatisticsShouldIncludeTotalFrames()
    {
        var content = await _response!.Content.ReadAsStringAsync();
        var lowerContent = content.ToLower();
        lowerContent.Should().MatchRegex("(total_frames|totalframes|frames)");
    }

    [Then("the statistics should include average detections per frame")]
    public async Task ThenTheStatisticsShouldIncludeAverageDetectionsPerFrame()
    {
        var content = await _response!.Content.ReadAsStringAsync();
        var lowerContent = content.ToLower();
        lowerContent.Should().MatchRegex("(average|detections|avg)");
    }

    [When("I request the heatmap data")]
    public async Task WhenIRequestTheHeatmapData()
    {
        _response = await Client.GetAsync($"/api/analyses/{_analysisId}/heatmap");
        _scenarioContext["Response"] = _response;
    }

    [Then("the response should contain an array of positions")]
    public async Task ThenTheResponseShouldContainAnArrayOfPositions()
    {
        var content = await _response!.Content.ReadAsStringAsync();
        content.Should().NotBeNullOrEmpty();
    }

    [When("I request analysis with ID {string}")]
    public async Task WhenIRequestAnalysisWithID(string id)
    {
        _response = await Client.GetAsync($"/api/analyses/{id}");
        _scenarioContext["Response"] = _response;
    }

    [Then("I should receive a {int} Not Found response")]
    public void ThenIShouldReceiveANotFoundResponse(int statusCode)
    {
        _response!.StatusCode.Should().Be((HttpStatusCode)statusCode);
    }
}
