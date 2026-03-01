using AnalizadorPadel.Api.Models.Entities;
using Microsoft.EntityFrameworkCore;

namespace AnalizadorPadel.Api.Data;

/// <summary>
/// DbContext para la base de datos SQLite del Analizador de Pádel
/// </summary>
public class PadelDbContext : DbContext
{
    public PadelDbContext(DbContextOptions<PadelDbContext> options) : base(options)
    {
    }

    public DbSet<VideoEntity> Videos => Set<VideoEntity>();
    public DbSet<AnalysisEntity> Analyses => Set<AnalysisEntity>();

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        base.OnModelCreating(modelBuilder);

        // Video configuration
        modelBuilder.Entity<VideoEntity>(entity =>
        {
            entity.HasIndex(v => v.Status);
            entity.HasIndex(v => v.UploadedAt);

            entity.HasOne(v => v.Analysis)
                  .WithOne(a => a.Video)
                  .HasForeignKey<AnalysisEntity>(a => a.VideoId)
                  .OnDelete(DeleteBehavior.Cascade);
        });

        // Analysis configuration
        modelBuilder.Entity<AnalysisEntity>(entity =>
        {
            entity.HasIndex(a => a.VideoId);
            entity.HasIndex(a => a.Status);
        });
    }
}
