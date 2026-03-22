using TechTalk.SpecFlow;
using AnalizadorPadel.Tests.Integration;

namespace AnalizadorPadel.Tests.BDD.Hooks;

[Binding]
public class ScenarioHooks
{
    private readonly ScenarioContext _scenarioContext;
    private CustomWebApplicationFactory? _factory;

    public ScenarioHooks(ScenarioContext scenarioContext)
    {
        _scenarioContext = scenarioContext;
    }

    [BeforeScenario]
    public void BeforeScenario()
    {
        _factory = new CustomWebApplicationFactory();
        var client = _factory.CreateClient();
        _scenarioContext["Client"] = client;
        _scenarioContext["Factory"] = _factory;
    }

    [AfterScenario]
    public void AfterScenario()
    {
        _factory?.Dispose();
    }
}
