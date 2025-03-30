import React, { useState } from 'react';
import { FiSearch, FiDownload, FiLoader } from "react-icons/fi";
import { MdVideocam } from "react-icons/md";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { getVideoInfo, downloadVideo } from '@/lib/api';
import Footer from './Footer';

interface VideoFormat {
  format_id: string;
  height: number;
  ext: string;
  filesize: number;
  format_note: string;
  vcodec: string;
  acodec: string;
}

interface VideoInfo {
  title: string;
  thumbnail: string;
  duration: string | number;
  formats: VideoFormat[];
  platform: string;
  url?: string;
}

const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return 'Unknown';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

const formatDuration = (duration: string | number): string => {
  const seconds = typeof duration === 'string' ? parseInt(duration) : duration;
  if (isNaN(seconds)) return 'Unknown';
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
};

const isValidUrl = (url: string) => {
  try {
    const urlObj = new URL(url);
    const validHosts = [
      'youtube.com', 'youtu.be',
      'facebook.com', 'fb.watch',
      'instagram.com',
      'tiktok.com'
    ];
    return validHosts.some(host => urlObj.hostname.includes(host));
  } catch {
    return false;
  }
};

const VideoPreview: React.FC<{ 
  videoInfo: VideoInfo;
  onVideoInfoUpdate: (updatedInfo: VideoInfo) => void;
}> = ({ videoInfo, onVideoInfoUpdate }) => {
  const [downloadingFormat, setDownloadingFormat] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [thumbnailError, setThumbnailError] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState<number>(0);
  const [isCompressing, setIsCompressing] = useState(false);

  const sanitizeFileName = (title: string): string => {
    return title
      .replace(/[<>:"/\\|?*]/g, '')
      .replace(/\s+/g, '_')
      .toLowerCase()
      .slice(0, 200);
  };

  const handleDownload = async (formatId: string) => {
    if (!videoInfo.url || downloadingFormat) return;
    
    setDownloadingFormat(formatId);
    setError(null);
    setDownloadProgress(0);
    setIsCompressing(false);
    
    try {
      // Create safe filename from video title
      const safeFileName = sanitizeFileName(videoInfo.title);
      const fileName = `${safeFileName}.mp4`;

      // Extract quality from formatId
      const quality = formatId.replace(/[^0-9]/g, '');
      
      // Special handling for TikTok and Instagram videos
      const isTikTok = videoInfo.platform === 'tiktok';
      const isInstagram = videoInfo.platform === 'instagram';
      const downloadParams = {
        url: videoInfo.url,
        format: isTikTok || isInstagram ? 'best' : `bestvideo[height<=${quality}]+bestaudio/best[height<=${quality}]`,
        filename: fileName,
        quality: isTikTok || isInstagram ? 1080 : parseInt(quality),
        platform: videoInfo.platform || 'youtube'
      };

      console.log('Starting download with params:', downloadParams); // Debug log

      const response = await fetch('http://127.0.0.1:8000/api/download', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(downloadParams)
      });

      if (!response.ok) {
        const errorData = await response.json();
        console.error('Download error response:', errorData); // Debug log
        
        // Extract error message from the response
        let errorMessage = 'Failed to download video';
        if (errorData) {
          if (typeof errorData === 'string') {
            errorMessage = errorData;
          } else if (errorData.detail) {
            errorMessage = typeof errorData.detail === 'string' 
              ? errorData.detail 
              : JSON.stringify(errorData.detail);
          } else if (errorData.message) {
            errorMessage = errorData.message;
          }
        }
        throw new Error(errorMessage);
      }

      // Get the blob from the response
      const blob = await response.blob();
      
      // Create download link
      const downloadUrl = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = downloadUrl;
      a.download = fileName;
      document.body.appendChild(a);
      a.click();
      URL.revokeObjectURL(downloadUrl);
      document.body.removeChild(a);

      setDownloadingFormat(null);
      setDownloadProgress(0);
      setIsCompressing(false);
      
    } catch (error) {
      console.error('Download error:', error);
      setDownloadingFormat(null);
      setDownloadProgress(0);
      setIsCompressing(false);
      setError(
        error instanceof Error ? error.message : 
        typeof error === 'string' ? error :
        'Failed to download video. Please try again.'
      );
    }
  };

  const getThumbnailUrl = (videoInfo: VideoInfo) => {
    if (!videoInfo || !videoInfo.thumbnail) return '';
    
    // For YouTube videos, ensure we're using the high-quality thumbnail
    if (videoInfo.platform === 'youtube' && videoInfo.url) {
      try {
        const videoId = videoInfo.url.includes('youtu.be') 
          ? videoInfo.url.split('/').pop() 
          : new URLSearchParams(new URL(videoInfo.url).search).get('v');
        if (videoId) {
          return `https://img.youtube.com/vi/${videoId}/maxresdefault.jpg`;
        }
      } catch (error) {
        console.error('Error getting YouTube thumbnail:', error);
      }
    }
    
    return videoInfo.thumbnail;
  };

  // Define standard quality formats with platform-specific adjustments
  const getVideoFormats = () => {
    if (videoInfo.platform === 'tiktok') {
      // For TikTok, show multiple quality options
      return [
        { format_id: '1080', height: 1080, ext: 'mp4', filesize: 0, format_note: 'Full HD', vcodec: 'h264', acodec: 'aac' },
        { format_id: '720', height: 720, ext: 'mp4', filesize: 0, format_note: 'HD', vcodec: 'h264', acodec: 'aac' },
        { format_id: '540', height: 540, ext: 'mp4', filesize: 0, format_note: 'SD', vcodec: 'h264', acodec: 'aac' },
        { format_id: '360', height: 360, ext: 'mp4', filesize: 0, format_note: 'Low', vcodec: 'h264', acodec: 'aac' }
      ];
    }

    // Use actual formats from the video info if available
    if (videoInfo.formats && videoInfo.formats.length > 0) {
      return videoInfo.formats.map(format => ({
        ...format,
        ext: format.ext || 'mp4',
        format_note: format.format_note || `${format.height}p`,
        filesize: format.filesize || 0
      }));
    }

    // Fallback formats if no formats are provided
    return [
      { format_id: '1080', height: 1080, ext: 'mp4', filesize: 0, format_note: 'Full HD', vcodec: 'h264', acodec: 'aac' },
      { format_id: '720', height: 720, ext: 'mp4', filesize: 0, format_note: 'HD', vcodec: 'h264', acodec: 'aac' },
      { format_id: '480', height: 480, ext: 'mp4', filesize: 0, format_note: 'SD', vcodec: 'h264', acodec: 'aac' },
      { format_id: '360', height: 360, ext: 'mp4', filesize: 0, format_note: 'Low', vcodec: 'h264', acodec: 'aac' }
    ];
  };

  // Use platform-specific formats
  const videoFormats = getVideoFormats();

  return (
    <div className="bg-gray-900 rounded-lg shadow-lg p-2 border border-gray-800 max-w-lg mx-auto">
      <div className="aspect-video relative rounded-lg overflow-hidden bg-gray-800 shadow-lg mb-2 max-h-[180px]">
        <div className="relative group cursor-pointer w-full h-full">
          {!thumbnailError ? (
            <img
              src={getThumbnailUrl(videoInfo)}
              alt={videoInfo.title}
              className="w-full h-full object-cover transform group-hover:scale-105 transition-transform duration-300"
              onError={() => setThumbnailError(true)}
              loading="eager"
            />
          ) : (
            <div className="w-full h-full bg-gray-800 flex items-center justify-center">
              <MdVideocam className="text-green-400 text-3xl" />
            </div>
          )}
          <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-1.5">
            <h3 className="text-xs font-bold text-white truncate">
              {videoInfo.title}
            </h3>
            <p className="text-[10px] font-medium text-gray-300">
              {formatDuration(videoInfo.duration)}
            </p>
          </div>
        </div>
      </div>

      {error && (
        <div className="p-1.5 mb-2 bg-red-900/50 text-red-400 rounded text-[10px] font-medium border border-red-800">
          {error}
        </div>
      )}

      {downloadProgress > 0 && (
        <div className="space-y-0.5 mb-2">
          <div className="h-0.5 bg-gray-800 rounded-full overflow-hidden">
            <div 
              className="h-full bg-green-500 transition-all duration-300"
              style={{ width: `${downloadProgress}%` }}
            />
          </div>
          <p className="text-[9px] text-gray-400">
            {isCompressing ? 'Compressing...' : `Downloading: ${Math.round(downloadProgress)}%`}
          </p>
        </div>
      )}

      <div className="overflow-hidden rounded-xl bg-[#242424] border border-[#333] shadow-lg">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="bg-[#1a1a1a]">
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Format</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Quality</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider hidden sm:table-cell">Size</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase tracking-wider">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#333]">
              {videoFormats.map((format, index) => (
                <tr 
                  key={index}
                  className="hover:bg-[#2a2a2a] transition-colors"
                >
                  <td className="px-4 py-3 whitespace-nowrap">
                    <div className="flex items-center">
                      <span className="h-6 w-6 text-[#99cc00] mr-2">
                        {format.ext === 'mp4' ? (
                          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
                            <path d="M4 8h16v8h-16z"/>
                            <path fillRule="evenodd" d="M2 6a2 2 0 012-2h16a2 2 0 012 2v12a2 2 0 01-2 2H4a2 2 0 01-2-2V6zm2 0h16v12H4V6z"/>
                          </svg>
                        ) : (
                          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
                            <path d="M12 18.66l-6.536-3.536 6.536-3.536 6.536 3.536-6.536 3.536zm0-9l-6.536-3.536 6.536-3.536 6.536 3.536-6.536 3.536z"/>
                          </svg>
                        )}
                      </span>
                      <span className="text-sm text-white">{format.ext.toUpperCase()}</span>
                    </div>
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap">
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-[#99cc00] bg-opacity-10 text-[#99cc00]">
                      {format.height}p
                    </span>
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap hidden sm:table-cell">
                    <span className="text-sm text-gray-300">{formatFileSize(format.filesize)}</span>
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-right">
                    {downloadingFormat === format.format_id ? (
                      <div className="flex items-center justify-end space-x-2">
                        <div className="w-24 bg-[#333] rounded-full h-1.5">
                          <div 
                            className="bg-[#99cc00] h-1.5 rounded-full transition-all duration-300"
                            style={{ width: `${downloadProgress}%` }}
                          ></div>
                        </div>
                        <span className="text-xs text-gray-400">{Math.round(downloadProgress)}%</span>
                      </div>
                    ) : (
                      <button
                        onClick={() => handleDownload(format.format_id)}
                        className="inline-flex items-center px-3 py-1.5 border border-[#99cc00] rounded-lg text-sm font-medium text-[#99cc00] hover:bg-[#99cc00] hover:bg-opacity-10 transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#99cc00] focus:ring-offset-[#242424]"
                      >
                        <svg className="h-4 w-4 mr-1.5" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L10 11.586V3a1 1 0 112 0v8.586l2.293-2.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                        </svg>
                        Download
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

const DownloadSection: React.FC = () => {
  const [url, setUrl] = useState('');
  const [videoInfo, setVideoInfo] = useState<VideoInfo | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleUrlSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!url || !isValidUrl(url)) {
      setError('Please enter a valid video URL');
      setVideoInfo(null);
      return;
    }

    try {
      setIsLoading(true);
      setError(null);

      const response = await fetch(`http://127.0.0.1:8000/api/info?url=${encodeURIComponent(url)}`);
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || errorData.message || 'Failed to fetch video information');
      }

      const info = await response.json();
      console.log('Video info received:', info); // Debug log
      
      setVideoInfo({
        title: info.title || '',
        duration: info.duration || '',
        thumbnail: info.thumbnail || '',
        formats: info.formats || [],
        platform: info.platform || getPlatformFromUrl(url),
        url: url
      });
    } catch (err) {
      console.error('Error fetching video info:', err);
      setError(
        err instanceof Error ? err.message : 
        typeof err === 'object' && err !== null && 'detail' in err ? 
        (err as any).detail : 'Failed to fetch video information. Please try again.'
      );
      setVideoInfo(null);
    } finally {
      setIsLoading(false);
    }
  };

  // Clear video info when URL is cleared
  const handleUrlChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newUrl = e.target.value;
    setUrl(newUrl);
    if (!newUrl) {
      setVideoInfo(null);
      setError(null);
    }
  };

  const getPlatformFromUrl = (url: string): string => {
    if (url.includes('youtube.com') || url.includes('youtu.be')) return 'youtube';
    if (url.includes('facebook.com') || url.includes('fb.watch')) return 'facebook';
    if (url.includes('instagram.com')) return 'instagram';
    if (url.includes('tiktok.com')) return 'tiktok';
    return 'unknown';
  };

  return (
    <div className="min-h-screen bg-[#1a1a1a] flex flex-col relative overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0 z-0">
        <div className="absolute inset-0 bg-gradient-to-br from-[#1a1a1a] via-[#242424] to-[#1a1a1a] animate-gradient"></div>
        <div className="absolute inset-0 opacity-20">
          <div className="absolute top-0 -left-4 w-72 h-72 bg-[#99cc00] rounded-full mix-blend-multiply filter blur-xl animate-blob"></div>
          <div className="absolute top-0 -right-4 w-72 h-72 bg-[#99cc00] rounded-full mix-blend-multiply filter blur-xl animate-blob animation-delay-2000"></div>
          <div className="absolute -bottom-8 left-20 w-72 h-72 bg-[#99cc00] rounded-full mix-blend-multiply filter blur-xl animate-blob animation-delay-4000"></div>
        </div>
      </div>

      {/* Navigation with Logo */}
      <nav className="bg-[#111]/80 backdrop-blur-sm border-b border-[#333] sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-14">
            <div className="flex items-center">
              <a href="/" className="flex items-center gap-2">
                <svg className="w-8 h-8 text-[#99cc00]" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M12 16L12 8M12 16L9 13M12 16L15 13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M3 15C3 17.8284 3 19.2426 3.87868 20.1213C4.75736 21 6.17157 21 9 21H15C17.8284 21 19.2426 21 20.1213 20.1213C21 19.2426 21 17.8284 21 15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
                <span className="text-lg font-bold text-white">
                  FastDownload<span className="text-[#99cc00]">LK</span>
                </span>
              </a>
            </div>
            <div className="flex items-center space-x-6">
              <a href="/features" className="text-gray-300 hover:text-white text-sm font-medium transition-colors">Features</a>
              <a href="/how-it-works" className="text-gray-300 hover:text-white text-sm font-medium transition-colors">How It Works</a>
              <a href="/legal" className="text-gray-300 hover:text-white text-sm font-medium transition-colors">Legal</a>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex-1 relative z-10">
        <div className="max-w-4xl mx-auto px-4 py-12 sm:px-6 lg:px-8">
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold text-white mb-2">
              Download Videos from YouTube, Facebook, Instagram & Twitter
            </h1>
            <p className="text-gray-400 text-lg">
              Fast, free, and easy video downloads in multiple formats
            </p>
          </div>

          {/* Download Form */}
          <div className="max-w-3xl mx-auto">
            <form onSubmit={handleUrlSubmit} className="space-y-3">
              <div className="flex gap-2">
                <div className="flex-1 relative">
                  <Input
                    type="text"
                    value={url}
                    onChange={handleUrlChange}
                    placeholder="Paste your video link here"
                    className="w-full h-12 text-base bg-[#111] border-2 border-[#333] rounded-xl text-white placeholder:text-gray-500 focus:border-[#99cc00] focus:ring-[#99cc00] pr-12"
                    disabled={isLoading}
                  />
                  {url && (
                    <button
                      type="button"
                      onClick={() => setUrl('')}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-white"
                    >
                      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  )}
                </div>
                <Button 
                  type="submit" 
                  disabled={isLoading || !url}
                  className="h-12 px-8 font-medium text-base bg-[#99cc00] hover:bg-[#8ab800] text-black disabled:opacity-50 disabled:cursor-not-allowed rounded-xl"
                >
                  {isLoading ? (
                    <div className="flex items-center gap-2">
                      <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                      </svg>
                      Processing
                    </div>
                  ) : (
                    'Download'
                  )}
                </Button>
              </div>

              <div className="text-xs text-gray-400 text-center">
                By using our service you accept our{' '}
                <a href="/legal" className="text-[#99cc00] hover:underline">Terms of Service</a>
                {' '}and{' '}
                <a href="/legal" className="text-[#99cc00] hover:underline">Privacy Policy</a>
              </div>

              {error && (
                <div className="p-3 bg-red-900/20 text-red-400 rounded-xl text-sm border border-red-900/50">
                  {error}
                </div>
              )}
            </form>

            <div className="mt-4 text-center">
              <button className="text-[#99cc00] hover:text-[#8ab800] text-sm flex items-center justify-center mx-auto gap-1">
                <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                How to download?
              </button>
            </div>

            {/* Platform Icons */}
            <div className="mt-8 grid grid-cols-2 sm:grid-cols-4 gap-4">
              {[
                { name: 'facebook.com', icon: 'f', color: '#1877f2' },
                { name: 'instagram.com', icon: <path d="M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zm0-2.163c-3.259 0-3.667.014-4.947.072-4.358.2-6.78 2.618-6.98 6.98-.059 1.281-.073 1.689-.073 4.948 0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98 1.281.058 1.689.072 4.948.072 3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98-1.281-.059-1.69-.073-4.949-.073zm0 5.838c-3.403 0-6.162 2.759-6.162 6.162s2.759 6.163 6.162 6.163 6.162-2.759 6.162-6.163c0-3.403-2.759-6.162-6.162-6.162zm0 10.162c-2.209 0-4-1.79-4-4 0-2.209 1.791-4 4-4s4 1.791 4 4c0 2.21-1.791 4-4 4zm6.406-11.845c-.796 0-1.441.645-1.441 1.44s.645 1.44 1.441 1.44c.795 0 1.439-.645 1.439-1.44s-.644-1.44-1.439-1.44z" />, color: '#E4405F' },
                { name: 'youtube.com', icon: <path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z" />, color: '#FF0000' },
                { name: 'tiktok.com', icon: <path d="M19.59 6.69a4.83 4.83 0 0 1-3.77-4.25V2h-3.45v13.67a2.89 2.89 0 0 1-5.2 1.74 2.89 2.89 0 0 1 2.31-4.64 2.93 2.93 0 0 1 .88.13V9.4a6.84 6.84 0 0 0-1-.05A6.33 6.33 0 0 0 5 20.1a6.34 6.34 0 0 0 10.86-4.43v-7a8.16 8.16 0 0 0 4.77 1.52v-3.4a4.85 4.85 0 0 1-1-.1z" />, color: '#000000' }
              ].map((platform) => (
                <div
                  key={platform.name}
                  className="flex items-center gap-2 p-3 rounded-xl bg-[#242424] hover:bg-[#2a2a2a] transition-colors cursor-pointer"
                >
                  <div className={`w-8 h-8 rounded-lg flex items-center justify-center text-white`} style={{ backgroundColor: platform.color }}>
                    {typeof platform.icon === 'string' ? (
                      <span className="font-bold">{platform.icon}</span>
                    ) : (
                      <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
                        {platform.icon}
                      </svg>
                    )}
                  </div>
                  <span className="text-gray-300 text-sm">{platform.name}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Video Preview Section */}
          {videoInfo && videoInfo.title && (
            <div className="mt-8">
              <VideoPreview 
                videoInfo={videoInfo} 
                onVideoInfoUpdate={setVideoInfo}
              />
            </div>
          )}
        </div>
      </main>
      <Footer />
    </div>
  );
};

export default DownloadSection; 