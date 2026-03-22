Feature: Video Analysis
  As a padel player
  I want to analyze uploaded videos
  So that I can get statistics about the match

  Background:
    Given the API is running
    And I have uploaded a video file

  Scenario: Start analysis for existing video
    Given I have an uploaded video with ID
    When I request analysis for the video
    Then I should receive a 202 Accepted response
    And the analysis status should be "Processing"

  Scenario: Get analysis results after completion
    Given I have a completed analysis
    When I request the analysis results
    Then I should receive a 200 OK response
    And the response should contain match statistics
    And the response should contain player positions

  Scenario: Get analysis statistics
    Given I have a completed analysis
    When I request the analysis statistics
    Then I should receive a 200 OK response
    And the statistics should include total frames
    And the statistics should include average detections per frame

  Scenario: Get heatmap data
    Given I have a completed analysis
    When I request the heatmap data
    Then I should receive a 200 OK response
    And the response should contain an array of positions

  Scenario: Analysis not found
    When I request analysis with ID "999999"
    Then I should receive a 404 Not Found response
