export interface VideoFormat {
    resolution: string;
    url: string;
    size?: string;
    quality?: string;
    format_id?: string;
    ext?: string;
}

export interface VideoInfo {
    title: string;
    thumbnail?: string;
    duration?: string;
    formats: VideoFormat[];
    platform: string;
    url: string;
} 