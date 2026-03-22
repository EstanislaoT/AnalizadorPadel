/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export type VideoDto = {
    id?: number;
    name?: string;
    description?: string | null;
    filePath?: string;
    uploadedAt?: string;
    status?: "Uploaded" | "Processing" | "Completed" | "Failed";
    analysisId?: number | null;
};
