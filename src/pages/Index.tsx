import React, { useState } from 'react';
import { DownloadTabs } from '../components/DownloadTabs';
import { downloadVideo, getVideoInfo } from '../lib/api';
import Navbar from "@/components/layout/Navbar";
import Footer from "@/components/layout/Footer";
import Features from "@/components/Features";
import HowItWorks from "@/components/sections/HowItWorks";
import Legal from "@/components/sections/Legal";

export default function Home() {
  const [url, setUrl] = useState('');
  const [formats, setFormats] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [isDownloading, setIsDownloading] = useState(false);

  const handleUrlSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');
    
    try {
      const info = await getVideoInfo(url);
      setFormats(info.formats);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to get video information');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownload = async (formatId: string) => {
    setIsDownloading(true);
    setError('');
    setDownloadProgress(0);
    
    try {
      await downloadVideo(url, formatId);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to download video');
    } finally {
      setIsDownloading(false);
      setDownloadProgress(0);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />
      <main className="py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-3xl mx-auto">
          <div className="text-center mb-12">
            <h1 className="text-4xl font-bold text-gray-900 mb-4">
              Video Downloader
            </h1>
            <p className="text-lg text-gray-600">
              Download videos from YouTube and other platforms
            </p>
          </div>

          <form onSubmit={handleUrlSubmit} className="mb-8">
            <div className="flex gap-4">
              <input
                type="text"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="Enter video URL"
                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-transparent"
                required
                disabled={isDownloading}
              />
              <button
                type="submit"
                disabled={isLoading || isDownloading}
                className="px-6 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {isLoading ? 'Loading...' : 'Get Formats'}
              </button>
            </div>
          </form>

          {error && (
            <div className="mb-8 p-4 bg-red-50 text-red-700 rounded-lg">
              {error}
            </div>
          )}

          {downloadProgress > 0 && (
            <div className="mb-8">
              <div className="w-full bg-gray-200 rounded-full h-2.5">
                <div 
                  className="bg-green-500 h-2.5 rounded-full transition-all duration-300"
                  style={{ width: `${downloadProgress}%` }}
                ></div>
              </div>
              <p className="text-sm text-gray-600 mt-2 text-center">
                Downloading: {downloadProgress}%
              </p>
            </div>
          )}

          {formats.length > 0 && (
            <DownloadTabs
              formats={formats}
              onDownload={handleDownload}
              isLoading={isDownloading}
            />
          )}
        </div>
      </main>
      <Footer />
    </div>
  );
}
