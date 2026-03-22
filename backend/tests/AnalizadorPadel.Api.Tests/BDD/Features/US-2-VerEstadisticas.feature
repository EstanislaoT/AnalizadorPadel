Feature: US-2 - Ver Estadísticas
  Como usuario
  Quiero ver estadísticas del análisis de mis videos
  Para entender el rendimiento de los jugadores

  Background:
    Given la API está funcionando correctamente
    And existe un análisis completado en el sistema

  @smoke @US-2
  Scenario: Ver estadísticas de un análisis completado
    Given existe un análisis con ID 1 en estado "Completed"
    When el usuario solicita las estadísticas del análisis 1
    Then el sistema responde con código 200 OK
    And las estadísticas incluyen frames totales
    And las estadísticas incluyen tasa de detección
    And las estadísticas incluyen tiempo de procesamiento
    And las estadísticas incluyen modelo utilizado

  @regression @US-2
  Scenario: Intentar ver estadísticas de análisis en proceso
    Given existe un análisis con ID 2 en estado "Running"
    When el usuario solicita las estadísticas del análisis 2
    Then el sistema responde con código 404 Not Found
    And el mensaje indica "no encontrado o sin resultados"

  @regression @US-2
  Scenario: Intentar ver estadísticas de análisis inexistente
    Given no existe un análisis con ID 999
    When el usuario solicita las estadísticas del análisis 999
    Then el sistema responde con código 404 Not Found

  @smoke @US-2
  Scenario: Ver datos de heatmap para análisis completado
    Given existe un análisis con ID 1 en estado "Completed"
    When el usuario solicita los datos del heatmap del análisis 1
    Then el sistema responde con código 200 OK
    And el heatmap contiene 100 puntos
    And cada punto tiene coordenadas X, Y e intensidad
    And las dimensiones de la cancha son "23.77m x 10.97m"

  @regression @US-2
  Scenario: Intentar ver heatmap de análisis fallido
    Given existe un análisis con ID 3 en estado "Failed"
    When el usuario solicita los datos del heatmap del análisis 3
    Then el sistema responde con código 404 Not Found

  @smoke @US-2
  Scenario: Ver estadísticas del dashboard
    Given existen 5 videos en el sistema
    And existen 3 análisis completados
    And existen 1 análisis fallido
    When el usuario solicita las estadísticas del dashboard
    Then el sistema responde con código 200 OK
    And el dashboard muestra 5 videos totales
    And el dashboard muestra 4 análisis totales
    And el dashboard muestra 3 análisis completados
    And el dashboard muestra 1 análisis fallido
    And la tasa de éxito es 75%

  @regression @US-2
  Scenario: Dashboard muestra videos recientes
    Given el usuario ha subido 10 videos en los últimos días
    When el usuario solicita las estadísticas del dashboard
    Then el dashboard muestra los 5 videos más recientes
    And los videos están ordenados por fecha descendente

  @regression @US-2
  Scenario: Dashboard muestra análisis recientes
    Given se han completado 8 análisis
    When el usuario solicita las estadísticas del dashboard
    Then el dashboard muestra los 5 análisis más recientes
    And los análisis están ordenados por fecha de inicio descendente

  @smoke @US-2
  Scenario: Descargar reporte de análisis completado
    Given existe un análisis con ID 1 en estado "Completed"
    When el usuario solicita el reporte del análisis 1
    Then el sistema responde con código 200 OK
    And la respuesta incluye la ruta al archivo PDF

  @regression @US-2
  Scenario: Intentar descargar reporte de análisis en proceso
    Given existe un análisis con ID 2 en estado "Running"
    When el usuario solicita el reporte del análisis 2
    Then el sistema responde con código 404 Not Found

  @regression @US-2
  Scenario Outline: Verificar cálculo de tasa de éxito
    Given existen <total> análisis en total
    And <completados> análisis están completados
    When el usuario solicita las estadísticas del dashboard
    Then la tasa de éxito es <tasa>%

    Examples:
      | total | completados | tasa |
      | 10    | 10          | 100  |
      | 10    | 7           | 70   |
      | 10    | 5           | 50   |
      | 10    | 0           | 0    |
      | 0     | 0           | 0    |
