/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export type AnalysisDto = {
    id?: number;
    videoId?: number;
    startedAt?: string;
    completedAt?: string | null;
    status?: "Pending" | "Running" | "Completed" | "Failed";
    errorMessage?: string | null;
    result?: unknown;
};
