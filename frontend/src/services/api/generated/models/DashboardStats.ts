/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { VideoDto } from './VideoDto';
import type { AnalysisDto } from './AnalysisDto';
export type DashboardStats = {
    totalVideos?: number;
    totalAnalyses?: number;
    completedAnalyses?: number;
    failedAnalyses?: number;
    successRatePercent?: number;
    avgDetectionRate?: number;
    recentVideos?: VideoDto[];
    recentAnalyses?: AnalysisDto[];
};
