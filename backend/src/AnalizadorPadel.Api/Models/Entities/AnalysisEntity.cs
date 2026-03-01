using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace AnalizadorPadel.Api.Models.Entities;

/// <summary>
/// Entidad de análisis almacenada en base de datos
/// </summary>
[Table("Analyses")]
public class AnalysisEntity
{
    [Key]
    [DatabaseGenerated(DatabaseGeneratedOption.Identity)]
    public int Id { get; set; }

    [Required]
    public int VideoId { get; set; }

    [Required]
    public DateTime StartedAt { get; set; } = DateTime.UtcNow;

    public DateTime? CompletedAt { get; set; }

    [Required]
    [MaxLength(20)]
    public string Status { get; set; } = "Pending";

    [MaxLength(2000)]
    public string? ErrorMessage { get; set; }

    // Result fields (flattened from AnalysisResult for SQLite storage)
    public int? TotalFrames { get; set; }
    public int? PlayersDetected { get; set; }
    public double? AvgDetectionsPerFrame { get; set; }
    public int? FramesWith4Players { get; set; }
    public double? DetectionRatePercent { get; set; }
    public double? ProcessingTimeSeconds { get; set; }

    [MaxLength(100)]
    public string? ModelUsed { get; set; }

    [MaxLength(500)]
    public string? VideoPath { get; set; }

    [MaxLength(50)]
    public string? Timestamp { get; set; }

    // Navigation property
    [ForeignKey("VideoId")]
    public VideoEntity Video { get; set; } = null!;
}
