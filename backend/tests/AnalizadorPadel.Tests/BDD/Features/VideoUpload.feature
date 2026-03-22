Feature: Video Upload
  As a padel player
  I want to upload match videos
  So that I can analyze them later

  Background:
    Given the API is running

  Scenario: Successfully upload a valid video
    Given I have a valid MP4 video file named "test-match.mp4"
    When I submit the video to the upload endpoint
    Then I should receive a 201 Created response
    And the response should contain a video ID
    And the video metadata should include file size and duration

  Scenario: Upload fails with invalid file type
    Given I have a text file named "invalid.txt"
    When I submit the file to the upload endpoint
    Then I should receive a 400 Bad Request response
    And the error message should indicate "Invalid file type"

  Scenario: Upload fails when file is too large
    Given I have a video file larger than 500MB
    When I submit the video to the upload endpoint
    Then I should receive a 400 Bad Request response
    And the error message should indicate "File too large"

  Scenario: Get video list after upload
    Given I have uploaded a video
    When I request the video list
    Then I should receive a 200 OK response
    And the list should contain at least 1 video
