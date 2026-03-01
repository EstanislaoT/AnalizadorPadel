/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { ApiResponseOfObject } from '../models/ApiResponseOfObject';
import type { IFormFile } from '../models/IFormFile';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class AnalizadorPadelApiService {
    /**
     * Sube un nuevo video
     * Permite subir un archivo de video para su posterior análisis
     * @param formData
     * @returns any OK
     * @throws ApiError
     */
    public static createVideo(
        formData: {
            file: IFormFile;
        },
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/videos',
            formData: formData,
            mediaType: 'multipart/form-data',
        });
    }
    /**
     * Lista todos los videos
     * Retorna una lista de todos los videos subidos
     * @returns any OK
     * @throws ApiError
     */
    public static getVideos(): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/videos',
        });
    }
    /**
     * Obtiene un video por ID
     * Retorna los detalles de un video específico
     * @param id
     * @returns any OK
     * @throws ApiError
     */
    public static getVideo(
        id: number,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/videos/{id}',
            path: {
                'id': id,
            },
        });
    }
    /**
     * Elimina un video
     * Elimina un video y sus archivos asociados
     * @param id
     * @returns any OK
     * @throws ApiError
     */
    public static deleteVideo(
        id: number,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'DELETE',
            url: '/api/videos/{id}',
            path: {
                'id': id,
            },
        });
    }
    /**
     * Inicia el análisis de un video
     * Comienza el procesamiento del video para detectar jugadores
     * @param id
     * @returns any OK
     * @throws ApiError
     */
    public static startAnalysis(
        id: number,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/videos/{id}/analyse',
            path: {
                'id': id,
            },
        });
    }
    /**
     * Obtiene un análisis por ID
     * Retorna los detalles de un análisis específico
     * @param id
     * @returns any OK
     * @throws ApiError
     */
    public static getAnalysis(
        id: number,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/analyses/{id}',
            path: {
                'id': id,
            },
        });
    }
    /**
     * Obtiene estadísticas del análisis
     * Retorna las estadísticas de detección del análisis
     * @param id
     * @returns any OK
     * @throws ApiError
     */
    public static getAnalysisStats(
        id: number,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/analyses/{id}/stats',
            path: {
                'id': id,
            },
        });
    }
    /**
     * Obtiene datos del heatmap
     * Retorna los datos para visualizar el heatmap de posiciones
     * @param id
     * @returns any OK
     * @throws ApiError
     */
    public static getAnalysisHeatmap(
        id: number,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/analyses/{id}/heatmap',
            path: {
                'id': id,
            },
        });
    }
    /**
     * Obtiene el reporte del análisis
     * Retorna la ruta del reporte PDF del análisis
     * @param id
     * @returns any OK
     * @throws ApiError
     */
    public static getAnalysisReport(
        id: number,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/analyses/{id}/report',
            path: {
                'id': id,
            },
        });
    }
    /**
     * Health check endpoint
     * @returns ApiResponseOfObject OK
     * @throws ApiError
     */
    public static healthCheck(): CancelablePromise<ApiResponseOfObject> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/health',
        });
    }
}
