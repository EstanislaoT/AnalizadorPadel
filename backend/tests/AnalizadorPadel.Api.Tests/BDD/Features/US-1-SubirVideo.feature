Feature: US-1 - Subir Video
  Como usuario
  Quiero subir videos de mis partidos de pádel
  Para analizar el rendimiento de los jugadores

  Background:
    Given la API está funcionando correctamente
    And el sistema de almacenamiento está disponible

  @smoke @US-1
  Scenario: Subir video con formato válido MP4
    Given el usuario tiene un video en formato MP4
    When el usuario sube el video "partido_padel.mp4"
    Then el sistema responde con código 201 Created
    And el video se almacena en el sistema
    And el sistema devuelve los detalles del video incluyendo un ID único
    And el estado del video es "Uploaded"

  @regression @US-1
  Scenario Outline: Subir videos con diferentes formatos válidos
    Given el usuario tiene un video en formato <formato>
    When el usuario sube el video "partido.<extension>"
    Then el sistema responde con código 201 Created
    And el video se almacena correctamente

    Examples:
      | formato | extension |
      | MP4     | mp4       |
      | AVI     | avi       |
      | MOV     | mov       |

  @regression @US-1
  Scenario: Intentar subir video con formato no soportado
    Given el usuario tiene un video en formato "texto"
    When el usuario sube el video "documento.txt"
    Then el sistema responde con código 400 Bad Request
    And el mensaje de error indica "Formato no soportado"
    And el video no se almacena en el sistema

  @regression @US-1
  Scenario: Intentar subir video excediendo tamaño máximo
    Given el usuario tiene un video de 501 MB
    When el usuario intenta subir el video grande
    Then el sistema responde con código 400 Bad Request
    And el mensaje de error indica "excede el tamaño máximo de 500MB"
    And el video no se almacena en el sistema

  @smoke @US-1
  Scenario: Subir video sin proporcionar archivo
    Given el usuario intenta subir un video
    When el usuario envía la petición sin archivo adjunto
    Then el sistema responde con código 400 Bad Request
    And el mensaje de error indica "No se proporcionó ningún video"

  @regression @US-1
  Scenario: Listar videos después de subir uno nuevo
    Given el usuario ha subido previamente 2 videos
    When el usuario sube un nuevo video "nuevo_partido.mp4"
    And el usuario solicita la lista de videos
    Then el sistema responde con código 200 OK
    And la lista contiene 3 videos
    And el video más reciente aparece primero en la lista

  @regression @US-1
  Scenario: Eliminar un video existente
    Given existe un video con ID 1 en el sistema
    When el usuario elimina el video con ID 1
    Then el sistema responde con código 200 OK
    And el mensaje confirma "Video eliminado exitosamente"
    And el video ya no aparece en la lista de videos

  @regression @US-1
  Scenario: Intentar eliminar un video inexistente
    Given no existe un video con ID 999 en el sistema
    When el usuario intenta eliminar el video con ID 999
    Then el sistema responde con código 404 Not Found
    And el mensaje de error indica "Video 999 no encontrado"
