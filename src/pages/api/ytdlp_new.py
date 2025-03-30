from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, HttpUrl
import yt_dlp
import json
import time
import os
import logging
import re
import asyncio
from typing import List, Optional
from urllib.parse import urlparse
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Video Downloader API",
    description="API for downloading videos from various platforms",
    version="1.0.0"
)

# Configure CORS with more permissive settings for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins during development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Add CORS headers to all responses
@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Expose-Headers"] = "*"
    return response

class DownloadRequest(BaseModel):
    url: HttpUrl
    format: Optional[str] = None
    quality: Optional[int] = None
    platform: Optional[str] = None
    output_format: Optional[str] = "mp4"

class VideoFormat(BaseModel):
    format_id: str
    height: Optional[int]
    quality: str
    ext: str
    filesize: Optional[int]
    format: str
    url: Optional[str]
    vcodec: Optional[str]
    acodec: Optional[str]
    fps: Optional[float]
    tbr: Optional[float]
    asr: Optional[int]
    filesize_approx: Optional[int]
    format_note: Optional[str]
    dynamic_range: Optional[str]
    abr: Optional[float]
    tbr_approx: Optional[float]

class VideoInfo(BaseModel):
    title: str
    formats: List[VideoFormat]
    thumbnail: Optional[str]
    duration: Optional[float]
    description: Optional[str]
    uploader: Optional[str]
    upload_date: Optional[str]
    view_count: Optional[int]
    webpage_url: str
    channel: Optional[str]
    channel_url: Optional[str]
    tags: Optional[List[str]]
    categories: Optional[List[str]]
    age_limit: Optional[int]
    is_live: Optional[bool]
    availability: Optional[str]
    qualities_table: Optional[List[dict]]

async def timeout_handler(seconds: int):
    """Windows-compatible timeout handler"""
    await asyncio.sleep(seconds)
    raise TimeoutError("Operation timed out")

def validate_url(url: str) -> bool:
    """Validate if the URL is from a supported platform"""
    supported_domains = [
        'youtube.com', 'youtu.be',
        'facebook.com', 'fb.watch',
        'instagram.com',
        'twitter.com', 'x.com',
        'tiktok.com',
        'vimeo.com',
        'dailymotion.com',
        'reddit.com',
        'pinterest.com',
        'linkedin.com'
    ]
    
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        return any(sd in domain for sd in supported_domains)
    except:
        return False

def filter_formats(formats):
    """Filter and sort available formats"""
    filtered_formats = []
    seen_qualities = set()  # Track seen qualities to avoid duplicates
    
    # First, get all available heights
    available_heights = sorted(set(f.get('height', 0) for f in formats if f.get('height')))
    
    # Get all formats for each height
    for height in available_heights:
        # Skip if we've already seen this quality
        if height in seen_qualities:
            continue
            
        # Get all formats for this height
        height_formats = [f for f in formats if f.get('height') == height]
        
        # Skip if no formats found
        if not height_formats:
            continue
            
        # Get the best format for this height
        best_format = None
        for fmt in height_formats:
            # Skip audio-only formats
            if fmt.get('vcodec') == 'none':
                continue
                
            # If we have a format with both video and audio, prefer it
            if fmt.get('acodec') != 'none' and fmt.get('vcodec') != 'none':
                best_format = fmt
                break
                
            # If we don't have a combined format yet, use this one
            if not best_format:
                best_format = fmt
        
        if best_format:
            seen_qualities.add(height)
            filtered_formats.append({
                'format_id': best_format.get('format_id'),
                'height': best_format.get('height'),
                'quality': f"{best_format.get('height')}p",
                'ext': best_format.get('ext'),
                'filesize': best_format.get('filesize'),
                'format': best_format.get('format'),
                'url': best_format.get('url'),
                'vcodec': best_format.get('vcodec'),
                'acodec': best_format.get('acodec'),
                'fps': best_format.get('fps'),
                'tbr': best_format.get('tbr'),
                'asr': best_format.get('asr'),
                'filesize_approx': best_format.get('filesize_approx'),
                'format_note': best_format.get('format_note'),
                'dynamic_range': best_format.get('dynamic_range'),
                'abr': best_format.get('abr'),
                'tbr_approx': best_format.get('tbr_approx')
            })
    
    # Sort by height (quality)
    filtered_formats.sort(key=lambda x: x['height'])
    return filtered_formats

def get_ydl_opts(download=False):
    """Get common yt-dlp options"""
    return {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
        'format': 'best' if not download else None,
        'outtmpl': '%(title)s.%(ext)s' if not download else None,
        'nocheckcertificate': True,
        'ignoreerrors': True,
        'no_warnings': True,
        'extractor_args': {
            'youtube': {
                'player_skip': [],
                'player_client': ['android', 'web'],
                'player_extra': ['android', 'web'],
            }
        },
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-us,en;q=0.5',
            'Sec-Fetch-Mode': 'navigate',
        },
        'socket_timeout': 30,
        'retries': 5,
        'fragment_retries': 5,
        'file_access_retries': 5,
        'extractor_retries': 5,
        'retry_sleep': 2,
        'sleep_interval': 2,
        'max_sleep_interval': 10,
        'sleep_interval_requests': 2,
        'throttledratelimit': 100000,
        'concurrent_fragments': 3,
        'buffersize': 32768,
        'http_chunk_size': 10485760,
        'format_sort': ['res', 'fps', 'codec', 'size', 'br', 'asr', 'ext'],
        'format_sort_force': True,
        'no_check_formats': False,
        'no_check_certificates': True,
        'prefer_insecure': True,
        'legacyserverconnect': True,
        'geo_verification_proxy': '',
        'source_address': '0.0.0.0',
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }],
        'merge_output_format': 'mp4',
        'postprocessor_args': [
            'ffmpeg:-c:v libx264 -preset medium -crf 23 -c:a aac -b:a 128k'
        ],
        'progress_hooks': [lambda d: logger.info(f"Download progress: {d.get('status', 'unknown')} - {d.get('_percent_str', '0%')}")],
    }

@app.get("/")
async def root():
    return {
        "message": "Video Downloader API is running",
        "version": "1.0.0",
        "supported_platforms": [
            "YouTube", "Facebook", "Instagram", "Twitter", "TikTok",
            "Vimeo", "Dailymotion", "Reddit", "Pinterest", "LinkedIn"
        ]
    }

@app.get("/api/info")
async def get_video_info(url: str = Query(..., description="URL of the video to download")):
    """Get video information and available formats"""
    logger.info(f"Received request for video info: {url}")
    
    try:
        if not validate_url(url):
            raise HTTPException(
                status_code=400,
                detail="Unsupported URL. Please provide a URL from a supported platform."
            )
        
        # Convert YouTube Shorts URL to regular URL if needed
        if 'youtube.com/shorts/' in url:
            video_id = url.split('/shorts/')[1].split('?')[0]
            url = f'https://www.youtube.com/watch?v={video_id}'
            logger.info(f"Converted Shorts URL to regular URL: {url}")
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Create a task for the timeout
                timeout_task = asyncio.create_task(timeout_handler(30))
                
                # Create a task for the video info extraction
                async def extract_info():
                    ydl_opts = get_ydl_opts(download=False)
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        try:
                            return ydl.extract_info(url, download=False)
                        except Exception as e:
                            logger.error(f"First extraction attempt failed: {str(e)}")
                            # Try with different options if first attempt fails
                            ydl_opts.update({
                                'extract_flat': False,
                                'force_generic_extractor': False,
                                'no_check_formats': False,
                                'extractor_args': {
                                    'youtube': {
                                        'player_skip': [],
                                        'player_client': ['android', 'web'],
                                        'player_extra': ['android', 'web'],
                                    }
                                }
                            })
                            with yt_dlp.YoutubeDL(ydl_opts) as ydl2:
                                return ydl2.extract_info(url, download=False)
                
                extract_task = asyncio.create_task(extract_info())
                
                # Wait for either the timeout or the extraction to complete
                done, pending = await asyncio.wait(
                    [timeout_task, extract_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel any pending tasks
                for task in pending:
                    task.cancel()
                
                # Check if we got a timeout
                if timeout_task in done:
                    raise TimeoutError("Video information extraction timed out")
                
                # Get the result from the extraction task
                info = extract_task.result()
                
                if not info:
                    raise HTTPException(status_code=400, detail="Could not extract video information")
                
                # Check if we have the required information
                if not info.get('formats'):
                    raise HTTPException(status_code=400, detail="No video formats found")
                
                filtered_formats = filter_formats(info.get('formats', []))
                
                if not filtered_formats:
                    raise HTTPException(status_code=400, detail="No suitable video formats found")
                
                # Create a table of available qualities
                qualities_table = []
                for fmt in filtered_formats:
                    qualities_table.append({
                        'quality': fmt['quality'],
                        'format': fmt['format'],
                        'filesize': fmt.get('filesize', fmt.get('filesize_approx', 0)),
                        'fps': fmt.get('fps', 0),
                        'codec': f"{fmt.get('vcodec', 'unknown')} / {fmt.get('acodec', 'unknown')}",
                        'format_id': fmt['format_id']
                    })
                
                # Handle duration - convert to float if it's a float or int
                duration = info.get('duration')
                if isinstance(duration, (int, float)):
                    duration = float(duration)
                
                response_data = VideoInfo(
                    title=info.get('title'),
                    formats=filtered_formats,
                    thumbnail=info.get('thumbnail'),
                    duration=duration,
                    description=info.get('description'),
                    uploader=info.get('uploader'),
                    upload_date=info.get('upload_date'),
                    view_count=info.get('view_count'),
                    webpage_url=info.get('webpage_url', url),
                    channel=info.get('channel'),
                    channel_url=info.get('channel_url'),
                    tags=info.get('tags'),
                    categories=info.get('categories'),
                    age_limit=info.get('age_limit'),
                    is_live=info.get('is_live'),
                    availability=info.get('availability'),
                    qualities_table=qualities_table
                )
                
                logger.info(f"Successfully extracted video info for: {url}")
                return JSONResponse(content=response_data.dict())
                    
            except TimeoutError:
                logger.error(f"Timeout while extracting video info for: {url}")
                raise HTTPException(
                    status_code=408,
                    detail="Video information extraction timed out. Please try again."
                )
            except Exception as e:
                retry_count += 1
                error_msg = str(e)
                logger.error(f"Error extracting video info (attempt {retry_count}/{max_retries}): {error_msg}")
                
                if "403" in error_msg and retry_count < max_retries:
                    await asyncio.sleep(2 * retry_count)  # Exponential backoff
                    continue
                if "403" in error_msg:
                    raise HTTPException(
                        status_code=403,
                        detail="Access to this video is restricted. Please try again later or check if the video is available."
                    )
                if retry_count == max_retries:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to extract video information after {max_retries} attempts. Error: {error_msg}"
                    )
                await asyncio.sleep(2 * retry_count)  # Exponential backoff
                
    except Exception as e:
        logger.error(f"Unexpected error in get_video_info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )

@app.post("/api/download")
async def download_video(request: DownloadRequest):
    """Download video in the specified format and quality"""
    logger.info(f"Received download request for: {request.url}")
    
    if not validate_url(str(request.url)):
        raise HTTPException(
            status_code=400,
            detail="Unsupported URL. Please provide a URL from a supported platform."
        )
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Create a task for the timeout
            timeout_task = asyncio.create_task(timeout_handler(60))
            
            # Create a task for the video download
            async def download_info():
                ydl_opts = get_ydl_opts(download=False)
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(str(request.url), download=False)
                    if not info:
                        raise HTTPException(status_code=400, detail="Could not extract video information")
                    
                    # Generate filename with proper encoding handling
                    title = info.get('title', 'video')
                    # Remove all non-ASCII characters
                    title = ''.join(char for char in title if ord(char) < 128)
                    # Remove invalid characters for filenames
                    title = re.sub(r'[<>:"/\\|?*]', '_', title)
                    # Ensure the title is not too long
                    title = title[:50] if len(title) > 50 else title
                    
                    filename = f"{title}_{request.quality}p.{request.output_format}"
                    
                    # Now download with the selected format
                    ydl_opts = get_ydl_opts(download=True)
                    ydl_opts.update({
                        'format': f'best[height<={request.quality}]/best',
                        'outtmpl': filename,
                        'merge_output_format': request.output_format,
                        'postprocessors': [{
                            'key': 'FFmpegVideoConvertor',
                            'preferedformat': request.output_format,
                        }],
                        'extract_flat': False,
                        'force_generic_extractor': False,
                        'no_check_formats': False,
                        'extractor_args': {
                            'youtube': {
                                'player_skip': [],
                                'player_client': ['android', 'web'],
                                'player_extra': ['android', 'web'],
                            }
                        }
                    })
                    
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(str(request.url), download=True)
                        if not info:
                            raise HTTPException(status_code=400, detail="Download failed")
                        
                        # Check if file exists and has size
                        if not os.path.exists(filename):
                            raise HTTPException(status_code=400, detail="Download failed - file is missing")
                        
                        file_size = os.path.getsize(filename)
                        if file_size == 0:
                            raise HTTPException(status_code=400, detail="Download failed - file is empty")
                        
                        # Verify the file is a valid video
                        if file_size < 1024:  # Less than 1KB
                            raise HTTPException(status_code=400, detail="Download failed - file is too small to be a valid video")
                        
                        # Return file info and path for streaming
                        return {
                            "title": info.get('title'),
                            "format": "MP4",
                            "quality": f"{request.quality}p",
                            "status": "success",
                            "file_size": file_size,
                            "filename": filename,
                            "url": str(request.url),
                            "output_format": request.output_format,
                            "duration": info.get('duration'),
                            "thumbnail": info.get('thumbnail'),
                            "uploader": info.get('uploader')
                        }
            
            download_task = asyncio.create_task(download_info())
            
            # Wait for either the timeout or the download to complete
            done, pending = await asyncio.wait(
                [timeout_task, download_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel any pending tasks
            for task in pending:
                task.cancel()
            
            # Check if we got a timeout
            if timeout_task in done:
                raise TimeoutError("Download timed out")
            
            # Get the result from the download task
            result = download_task.result()
            
            # Create a streaming response for the video file
            def file_iterator():
                try:
                    with open(result["filename"], "rb") as f:
                        while chunk := f.read(8192):  # Read 8KB chunks
                            yield chunk
                finally:
                    # Clean up the file after streaming
                    try:
                        os.remove(result["filename"])
                    except:
                        pass
            
            # Return streaming response with the video file
            return StreamingResponse(
                file_iterator(),
                media_type=f"video/{result['output_format']}",
                headers={
                    "Content-Disposition": f'attachment; filename="{result["filename"]}"',
                    "Content-Length": str(result["file_size"])
                }
            )
                    
        except TimeoutError:
            logger.error(f"Timeout while downloading video: {request.url}")
            raise HTTPException(
                status_code=408,
                detail="Download timed out. Please try again."
            )
        except Exception as e:
            retry_count += 1
            error_msg = str(e)
            logger.error(f"Error downloading video (attempt {retry_count}/{max_retries}): {error_msg}")
            
            if "403" in error_msg and retry_count < max_retries:
                await asyncio.sleep(2 * retry_count)  # Exponential backoff
                continue
            if "403" in error_msg:
                raise HTTPException(
                    status_code=403,
                    detail="Access to this video is restricted. Please try again later or check if the video is available."
                )
            if "Requested format is not available" in error_msg:
                raise HTTPException(
                    status_code=400,
                    detail=f"Selected quality {request.quality}p is not available. Please try a different quality."
                )
            raise HTTPException(status_code=400, detail=error_msg)

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 