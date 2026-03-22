using Xunit;

namespace AnalizadorPadel.Tests;

[CollectionDefinition("BDD Tests")]
public class BddTestCollection : ICollectionFixture<Integration.CustomWebApplicationFactory>
{
    // Esta clase no tiene código, solo sirve para definir la colección
}
