from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from pydantic import BaseModel, validator, HttpUrl
from typing import Optional, List, Dict, Any
import yt_dlp
import os
import tempfile
import shutil
import logging
import traceback
from urllib.parse import quote, urlparse
import aiohttp
import asyncio
import json
from datetime import datetime
import hashlib
import requests
import io
import ffmpeg

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API Configuration
RAPIDAPI_KEY = "408bb9589amsh1872fe8447998b3p14706bjsnc0f242a84e50"  # Updated RapidAPI key
API_CONFIGS = {
    "youtube": {
        "url": "https://yt-video-download.p.rapidapi.com/downloads/mp4",
        "host": "yt-video-download.p.rapidapi.com"
    },
    "tiktok": {
        "url": "https://tiktok-video-no-watermark2.p.rapidapi.com/",
        "host": "tiktok-video-no-watermark2.p.rapidapi.com"
    },
    "instagram": {
        "url": "https://instagram-video-downloader-download-instagram-videos-stories.p.rapidapi.com/index",
        "host": "instagram-video-downloader-download-instagram-videos-stories.p.rapidapi.com"
    },
    "facebook": {
        "url": "https://facebook-reel-and-video-downloader.p.rapidapi.com/app/main.php",
        "host": "facebook-reel-and-video-downloader.p.rapidapi.com"
    }
}

SUPPORTED_PLATFORMS = {
    "youtube.com": "youtube",
    "youtu.be": "youtube",
    "facebook.com": "facebook",
    "fb.watch": "facebook",
    "instagram.com": "instagram",
    "twitter.com": "twitter",
    "x.com": "twitter",
    "tiktok.com": "tiktok",
    "linkedin.com": "linkedin"
}

# In-memory storage for download progress
download_progress = {}

# Add at the top of the file with other global variables
video_info_cache = {}

class VideoRequest(BaseModel):
    url: str
    format_id: Optional[str] = None
    quality: Optional[str] = None
    download_id: Optional[str] = None

    @validator('url')
    def validate_url(cls, v):
        if not v:
            raise ValueError('URL is required')
        if not any(platform in v.lower() for platform in ['youtube.com', 'youtu.be', 'facebook.com', 'instagram.com', 'tiktok.com', 'twitter.com', 'pinterest.com']):
            raise ValueError('Invalid URL. Please enter a valid video URL from YouTube, Facebook, Instagram, TikTok, Twitter, or Pinterest.')
        return v

class VideoFormat(BaseModel):
    resolution: str
    url: str
    size: Optional[str] = None
    quality: Optional[str] = None
    format_id: Optional[str] = None
    ext: Optional[str] = None

class VideoResponse(BaseModel):
    title: str
    thumbnail: Optional[str] = None
    duration: Optional[str] = None
    formats: List[VideoFormat]
    platform: str
    download_id: Optional[str] = None

class DownloadProgress(BaseModel):
    status: str
    progress: float
    speed: Optional[str] = None
    eta: Optional[str] = None
    error: Optional[str] = None

class VideoDownloadRequest(BaseModel):
    url: str
    format: str
    fileName: str = None
    compress: bool = False
    quality: int = None

def generate_download_id(url: str) -> str:
    """Generate a unique download ID based on URL and timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    return f"{timestamp}_{url_hash}"

def progress_hook(d):
    """Track download progress."""
    if d['status'] == 'downloading':
        download_id = d.get('info_dict', {}).get('download_id')
        if download_id:
            download_progress[download_id] = {
                'status': 'downloading',
                'progress': d.get('_percent_str', '0%').replace('%', ''),
                'speed': d.get('_speed_str', 'N/A'),
                'eta': d.get('_eta_str', 'N/A')
            }
    elif d['status'] == 'finished':
        download_id = d.get('info_dict', {}).get('download_id')
        if download_id:
            download_progress[download_id] = {
                'status': 'finished',
                'progress': '100%'
            }

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://localhost:8081", "http://192.168.8.104:8080", "http://192.168.8.104:8081", "http://127.0.0.1:8080", "http://127.0.0.1:8081", "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_platform(url: str) -> str:
    """Determine the platform from the URL."""
    if 'youtube.com' in url or 'youtu.be' in url:
        return 'youtube'
    elif 'facebook.com' in url or 'fb.watch' in url:
        return 'facebook'
    elif 'instagram.com' in url:
        return 'instagram'
    elif 'tiktok.com' in url:
        return 'tiktok'
    else:
        raise ValueError("Unsupported platform")

async def fetch_from_rapidapi(platform: str, url: str) -> Dict:
    """Fetch video information from RapidAPI."""
    if platform not in API_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Platform {platform} not supported")
    
    api_config = API_CONFIGS[platform]
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": api_config["host"]
    }
    
    params = {"url": url}
    if platform == "youtube":
        params["format"] = "mp4"
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(api_config["url"], headers=headers, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"RapidAPI error for {platform}: {error_text}")
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Failed to fetch video from {platform}"
                    )
                
                data = await response.json()
                logger.info(f"RapidAPI response for {platform}: {data}")
                return data
                
        except aiohttp.ClientError as e:
            logger.error(f"RapidAPI request failed for {platform}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to connect to {platform} service"
            )

def parse_rapidapi_response(platform: str, data: Dict) -> VideoResponse:
    """Parse RapidAPI response into our VideoResponse format."""
    try:
        if platform == "youtube":
            formats = []
            # Handle the new YouTube API response format
            if isinstance(data, dict):
                # Check for different quality options
                for quality, url in data.get("downloads", {}).items():
                    if url and isinstance(url, str):
                        formats.append(VideoFormat(
                            resolution=quality,
                            url=url,
                            quality=quality,
                            format_id=quality,
                            ext="mp4"
                        ))
                
                return VideoResponse(
                    title=data.get("title", ""),
                    thumbnail=data.get("thumbnail", ""),
                    duration=str(data.get("duration", "")),
                    formats=formats,
                    platform=platform
                )
            else:
                raise HTTPException(status_code=400, detail="Invalid response format from YouTube API")
        
        elif platform == "facebook":
            formats = []
            # Handle the new Facebook API response format
            if isinstance(data, dict):
                # Check for HD quality
                if hd_url := data.get("hd"):
                    formats.append(VideoFormat(
                        resolution="HD",
                        url=hd_url,
                        quality="HD",
                        format_id="hd",
                        ext="mp4"
                    ))
                # Check for SD quality
                if sd_url := data.get("sd"):
                    formats.append(VideoFormat(
                        resolution="SD",
                        url=sd_url,
                        quality="SD",
                        format_id="sd",
                        ext="mp4"
                    ))
                # Check for thumbnail
                thumbnail = data.get("thumb", "")
                
                return VideoResponse(
                    title=data.get("title", ""),
                    thumbnail=thumbnail,
                    formats=formats,
                    platform=platform
                )
            else:
                raise HTTPException(status_code=400, detail="Invalid response format from Facebook API")
        
        elif platform == "tiktok":
            formats = []
            for quality, url in data.get("videos", {}).items():
                formats.append(VideoFormat(
                    resolution=quality,
                    url=url,
                    quality=quality,
                    format_id=quality,
                    ext="mp4"
                ))
            
            return VideoResponse(
                title=data.get("desc", ""),
                thumbnail=data.get("cover", ""),
                formats=formats,
                platform=platform
            )
        
        elif platform == "instagram":
            formats = []
            for quality, url in data.get("videos", {}).items():
                formats.append(VideoFormat(
                    resolution=quality,
                    url=url,
                    quality=quality,
                    format_id=quality,
                    ext="mp4"
                ))
            
            return VideoResponse(
                title=data.get("title", ""),
                thumbnail=data.get("thumbnail", ""),
                formats=formats,
                platform=platform
            )
        
        raise HTTPException(status_code=400, detail=f"Unsupported platform: {platform}")
    except Exception as e:
        logger.error(f"Error parsing RapidAPI response: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to parse video information")

@app.get("/api/progress/{download_id}")
async def get_download_progress(download_id: str):
    """Get the progress of a download."""
    if download_id not in download_progress:
        raise HTTPException(status_code=404, detail="Download not found")
    return download_progress[download_id]

@app.post("/api/convert")
async def convert_video(request: VideoRequest):
    """Handle video conversion for all platforms."""
    try:
        platform = get_platform(request.url)
        logger.info(f"Processing convert request for platform: {platform}")

        if platform == "youtube":
            return await convert_youtube_video(request)
        elif platform in API_CONFIGS:
            # For other platforms, get the direct download URL
            data = await fetch_from_rapidapi(platform, request.url)
            response = parse_rapidapi_response(platform, data)
            
            # Get the best quality URL
            if not response.formats:
                raise HTTPException(status_code=400, detail="No download URL available")
            
            # Sort formats by quality and get the best one
            if request.quality:
                # Try to match requested quality
                for fmt in response.formats:
                    if fmt.quality and str(request.quality) in fmt.quality:
                        return JSONResponse(content={
                            "download_url": fmt.url,
                            "title": response.title,
                            "quality": fmt.quality,
                            "ext": fmt.ext
                        })
            
            # Default to highest quality
            return JSONResponse(content={
                "download_url": response.formats[0].url,
                "title": response.title,
                "quality": response.formats[0].quality,
                "ext": response.formats[0].ext
            })
            
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported platform: {platform}")
            
    except HTTPException as he:
        logger.error(f"HTTP error in convert_video: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"Error in convert_video: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/formats")
async def get_formats(url: str):
    """Get available formats for a video URL."""
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if not info:
                raise HTTPException(status_code=400, detail="Could not extract video information")
            
            # Define standard formats
            standard_formats = [
                {'format_id': '1080p', 'height': 1080, 'ext': 'mp4'},
                {'format_id': '720p', 'height': 720, 'ext': 'mp4'},
                {'format_id': '360p', 'height': 360, 'ext': 'mp4'},
                {'format_id': '240p', 'height': 240, 'ext': 'mp4'},
                {'format_id': '144p', 'height': 144, 'ext': 'mp4'}
            ]
            
            return {"formats": standard_formats}
            
    except Exception as e:
        logger.error(f"Error getting formats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get video formats: {str(e)}")

def compress_video(input_stream, quality):
    input_file = None
    output_file = None
    try:
        # Create temporary files with unique names
        input_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        output_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        
        # Close the files to release handles
        input_file.close()
        output_file.close()
        
        # Write input data to temporary file
        with open(input_file.name, 'wb') as f:
            f.write(input_stream.read())
        
        # Calculate bitrate based on quality
        bitrate = {
            1080: '5000k',
            720: '2500k',
            480: '1000k',
            360: '750k',
            240: '500k',
            144: '250k'
        }.get(quality, '1000k')
        
        # Run FFmpeg command
        ffmpeg.input(input_file.name).output(
            output_file.name,
            vcodec='libx264',
            acodec='aac',
            video_bitrate=bitrate,
            audio_bitrate='128k',
            preset='medium',
            movflags='faststart'
        ).overwrite_output().run(quiet=True)
        
        # Read compressed output
        with open(output_file.name, 'rb') as f:
            compressed_data = f.read()
        
        if len(compressed_data) == 0:
            raise ValueError("Compression resulted in empty file")
        
        return io.BytesIO(compressed_data)
    
    except Exception as e:
        logger.error(f"Compression error: {str(e)}")
        # Return original video if compression fails
        input_stream.seek(0)
        return input_stream
    
    finally:
        # Clean up temp files
        try:
            if input_file and os.path.exists(input_file.name):
                os.unlink(input_file.name)
            if output_file and os.path.exists(output_file.name):
                os.unlink(output_file.name)
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {str(e)}")

async def download_video(request: VideoDownloadRequest):
    """Download video in specified quality."""
    platform = get_platform(request.url)
    
    if platform == 'youtube':
        return await download_youtube_video(request)
    elif platform == 'facebook':
        return await download_facebook_video(request)
    elif platform == 'instagram':
        return await download_instagram_video(request)
    elif platform == 'tiktok':
        return await download_tiktok_video(request)
    else:
        raise HTTPException(status_code=400, detail="Unsupported platform")

@app.post("/api/download")
async def handle_download(request: VideoDownloadRequest):
    """Handle video download request."""
    try:
        return await download_video(request)
    except Exception as e:
        logger.error(f"Download error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

async def convert_youtube_video(request: VideoRequest):
    """Handle YouTube video conversion using yt-dlp."""
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'no_playlist': True,
            'playlist_items': '1',
            'no_download': True,
            'format_sort': ['res', 'fps', 'codec', 'size', 'br', 'asr', 'ext'],
            'merge_output_format': 'mp4',
            'retries': 10,
            'fragment_retries': 10,
            'file_access_retries': 10,
            'extractor_retries': 10,
            'ignoreerrors': True,
            'no_check_certificate': True,
            'prefer_insecure': True,
            'legacyserverconnect': True,
            'source_address': '0.0.0.0',
            'extractor_args': {
                'youtube': {
                    'player_client': ['android', 'web'],
                    'player_skip': ['webpage', 'config', 'js']
                }
            },
            'socket_timeout': 30,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(request.url, download=False)
            if not info:
                raise HTTPException(status_code=400, detail="Could not extract video information")

            formats = []
            if 'formats' in info:
                # Create a dictionary to store unique resolutions
                unique_formats = {}
                
                for f in info['formats']:
                    if f.get('vcodec') != 'none' and f.get('acodec') != 'none':
                        height = f.get('height', 0)
                        if height and height not in unique_formats:
                            unique_formats[height] = f

                # Convert dictionary to list and sort by height
                for height, f in sorted(unique_formats.items(), reverse=True):
                    formats.append(VideoFormat(
                        resolution=f"{height}p",
                        url=f.get('url', ''),
                        size=f.get('filesize_str'),
                        quality=f"{height}p",
                        format_id=f"{height}p"  # Use resolution as format_id
                    ))

            return VideoResponse(
                title=info.get('title', ''),
                thumbnail=info.get('thumbnail', ''),
                duration=str(info.get('duration', '')),
                formats=formats,
                platform="youtube"
            )

    except Exception as e:
        logger.error(f"Error in convert_youtube_video: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))

async def download_youtube_video(request: VideoDownloadRequest):
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp()
        
        ydl_opts = {
            'format': f'bestvideo[height<={request.quality}]+bestaudio/best[height<={request.quality}]',
            'merge_output_format': 'mp4',
            'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Download the video
            info = ydl.extract_info(request.url, download=True)
            if not info:
                raise HTTPException(status_code=400, detail="Could not extract video information")
            
            # Get the downloaded file path
            filename = ydl.prepare_filename(info)
            if not os.path.exists(filename):
                raise HTTPException(status_code=400, detail="Downloaded file not found")
            
            # Check file size
            file_size = os.path.getsize(filename)
            if file_size == 0:
                raise HTTPException(status_code=400, detail="Downloaded file is empty")
            
            # Create safe filename with proper encoding
            safe_title = ''.join(c for c in info.get('title', '') if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_title = safe_title.replace(' ', '_')
            output_filename = request.fileName or f"{safe_title}_{request.quality}p.mp4"
            
            # Read file into memory
            with open(filename, 'rb') as f:
                video_data = f.read()
            
            if len(video_data) == 0:
                raise HTTPException(status_code=400, detail="Video data is empty")
            
            buffer = io.BytesIO(video_data)
            
            # Compress if requested
            if request.compress:
                try:
                    compressed_buffer = compress_video(buffer, request.quality)
                    if compressed_buffer.getbuffer().nbytes > 0:
                        buffer = compressed_buffer
                    else:
                        logger.warning("Compression resulted in empty file, using original")
                        buffer.seek(0)
                except Exception as e:
                    logger.error(f"Compression error: {str(e)}")
                    buffer.seek(0)
            
            # Final size check
            final_size = buffer.getbuffer().nbytes
            if final_size == 0:
                raise HTTPException(status_code=400, detail="Final video buffer is empty")
            
            # Return the video with properly encoded headers
            return StreamingResponse(
                buffer,
                media_type="video/mp4",
                headers={
                    "Content-Disposition": f'attachment; filename="{output_filename.encode("ascii", "ignore").decode("ascii")}"',
                    "Content-Type": "video/mp4",
                    "Content-Length": str(final_size)
                }
            )
            
    except Exception as e:
        logger.error(f"Error downloading video: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Failed to download video: {str(e)}")
    finally:
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.error(f"Error cleaning up temp directory: {str(e)}")

async def download_facebook_video(request: VideoDownloadRequest):
    """Download Facebook video using yt-dlp."""
    temp_dir = tempfile.mkdtemp()
    try:
        # First get video info without downloading
        info_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'socket_timeout': 30,
            'retries': 10,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        }
        
        with yt_dlp.YoutubeDL(info_opts) as ydl:
            try:
                # First get video info
                info = ydl.extract_info(request.url, download=False)
                if not info:
                    raise HTTPException(status_code=400, detail="Could not extract video information")
                
                # Store thumbnail and title
                thumbnail_url = info.get('thumbnail', '')
                original_title = info.get('title', '')
                
                # Create safe filename from original title
                safe_title = ''.join(c for c in original_title if c.isascii() and (c.isalnum() or c in (' ', '-', '_'))).strip()
                safe_title = safe_title.replace(' ', '_')
                if not safe_title:
                    safe_title = 'facebook_video'
                
                # Configure download options
                download_opts = {
                    'quiet': True,
                    'no_warnings': True,
                    'format': 'best',  # Always use best available format for Facebook
                    'outtmpl': {
                        'default': os.path.join(temp_dir, f"{safe_title}.%(ext)s")
                    },
                    'socket_timeout': 30,
                    'retries': 10,
                    'http_headers': {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                }
                
                # Download the video
                with yt_dlp.YoutubeDL(download_opts) as ydl_download:
                    info = ydl_download.extract_info(request.url, download=True)
                    filename = ydl_download.prepare_filename(info)
                    
                    if not os.path.exists(filename):
                        raise HTTPException(status_code=404, detail="Video file not found after download")
                    
                    with open(filename, 'rb') as f:
                        video_buffer = io.BytesIO(f.read())
                    
                    if request.compress:
                        video_buffer = compress_video(video_buffer, request.quality)
                    
                    # Use the safe title for the output filename
                    output_filename = f"{safe_title}.mp4"
                    
                    # Return the video with headers
                    headers = {
                        'Content-Disposition': f'attachment; filename="{quote(output_filename)}"',
                        'Content-Type': 'video/mp4',
                        'X-Video-Title': quote(original_title),
                        'X-Video-Thumbnail': quote(thumbnail_url)
                    }
                    
                    return StreamingResponse(
                        video_buffer,
                        media_type='video/mp4',
                        headers=headers
                    )
                    
            except Exception as e:
                logger.error(f"Facebook download error: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Failed to download Facebook video: {str(e)}")
    finally:
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.error(f"Error cleaning up temporary directory: {str(e)}")

@app.get("/api/info")
async def get_video_info(url: str):
    """Get video information including title, thumbnail, and available formats."""
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'socket_timeout': 30,
            'retries': 10,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if not info:
                raise HTTPException(status_code=400, detail="Could not extract video information")
            
            # Get platform-specific thumbnail
            platform = get_platform(url)
            thumbnail = info.get("thumbnail", "")
            
            # Define standard formats
            standard_formats = [
                {'format_id': '1080p', 'height': 1080, 'ext': 'mp4'},
                {'format_id': '720p', 'height': 720, 'ext': 'mp4'},
                {'format_id': '480p', 'height': 480, 'ext': 'mp4'},
                {'format_id': '360p', 'height': 360, 'ext': 'mp4'},
                {'format_id': '240p', 'height': 240, 'ext': 'mp4'},
                {'format_id': '144p', 'height': 144, 'ext': 'mp4'}
            ]
            
            response_data = {
                "title": info.get("title", ""),
                "duration": info.get("duration", ""),
                "thumbnail": thumbnail,
                "formats": standard_formats,
                "platform": platform
            }
            
            return response_data
            
    except Exception as e:
        logger.error(f"Error getting video info: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))

async def update_video_info(url: str, info: dict):
    """Update video info in memory cache."""
    try:
        # Generate a unique key based on the URL
        video_id = url.split('v=')[-1] if 'v=' in url else url.split('/')[-1]
        cache_key = f"video_info_{video_id}"
        
        # Store in memory cache
        video_info_cache[cache_key] = {
            "title": info.get("title", ""),
            "thumbnail": info.get("thumbnail", ""),
            "duration": info.get("duration", ""),
            "formats": info.get("formats", []),
            "platform": info.get("platform", "unknown")
        }
        return video_info_cache[cache_key]
    except Exception as e:
        logger.error(f"Error updating video info: {str(e)}")
        return info

async def download_instagram_video(request: VideoDownloadRequest):
    """Download Instagram video using yt-dlp."""
    temp_dir = tempfile.mkdtemp()
    try:
        ydl_opts = {
            'format': request.format,  # Use the format from the request
            'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'progress_hooks': [progress_hook],
            'verbose': True
        }
        
        logger.info(f"Starting Instagram video download for URL: {request.url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(request.url, download=True)
                if not info:
                    raise HTTPException(status_code=400, detail="Could not extract video information")
                
                filename = ydl.prepare_filename(info)
                if not os.path.exists(filename):
                    raise HTTPException(status_code=404, detail="Video file not found after download")
                
                logger.info(f"Instagram video downloaded successfully: {filename}")
                
                with open(filename, 'rb') as f:
                    video_buffer = io.BytesIO(f.read())
                
                if request.compress:
                    logger.info("Compressing Instagram video...")
                    video_buffer = compress_video(video_buffer, request.quality)
                
                return StreamingResponse(
                    video_buffer,
                    media_type='video/mp4',
                    headers={
                        'Content-Disposition': f'attachment; filename="{quote(info["title"])}.mp4"'
                    }
                )
                
            except yt_dlp.utils.DownloadError as e:
                logger.error(f"Instagram download error: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Failed to download Instagram video: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error during Instagram download: {str(e)}")
                raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.error(f"Error cleaning up temp directory: {str(e)}")

async def download_tiktok_video(request: VideoDownloadRequest):
    """Download TikTok video using yt-dlp."""
    temp_dir = tempfile.mkdtemp()
    try:
        ydl_opts = {
            'format': f'bestvideo[height<={request.quality}]+bestaudio/best[height<={request.quality}]',
            'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'progress_hooks': [progress_hook],
            'verbose': True
        }
        
        logger.info(f"Starting TikTok video download for URL: {request.url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(request.url, download=True)
                if not info:
                    raise HTTPException(status_code=400, detail="Could not extract video information")
                
                filename = ydl.prepare_filename(info)
                if not os.path.exists(filename):
                    raise HTTPException(status_code=404, detail="Video file not found after download")
                
                logger.info(f"TikTok video downloaded successfully: {filename}")
                
                with open(filename, 'rb') as f:
                    video_buffer = io.BytesIO(f.read())
                
                if request.compress:
                    logger.info("Compressing TikTok video...")
                    video_buffer = compress_video(video_buffer, request.quality)
                
                return StreamingResponse(
                    video_buffer,
                    media_type='video/mp4',
                    headers={
                        'Content-Disposition': f'attachment; filename="{quote(info["title"])}.mp4"'
                    }
                )
                
            except yt_dlp.utils.DownloadError as e:
                logger.error(f"TikTok download error: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Failed to download TikTok video: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error during TikTok download: {str(e)}")
                raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.error(f"Error cleaning up temporary directory: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000) 