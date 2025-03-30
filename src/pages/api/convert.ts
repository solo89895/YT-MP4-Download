import { NextApiRequest, NextApiResponse } from 'next';
import ytdl from 'ytdl-core';

export const config = {
  api: {
    responseLimit: false,
    bodyParser: {
      sizeLimit: '10mb',
    },
  },
};

function formatDuration(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;

  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }
  return `${minutes}:${secs.toString().padStart(2, '0')}`;
}

function formatFileSize(bytes: number): string {
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  if (bytes === 0) return '0 Byte';
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${Math.round(bytes / Math.pow(1024, i))} ${sizes[i]}`;
}

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  // Set CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  // Handle OPTIONS request
  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  if (req.method !== 'POST' && req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    let url: string;
    
    if (req.method === 'POST') {
      if (!req.body || !req.body.url) {
        return res.status(400).json({ error: 'URL is required in request body' });
      }
      url = req.body.url;
    } else {
      if (!req.query.url) {
        return res.status(400).json({ error: 'URL is required as query parameter' });
      }
      url = req.query.url as string;
    }

    // Clean the URL
    url = url.trim();

    // Validate YouTube URL
    if (!ytdl.validateURL(url)) {
      return res.status(400).json({ error: 'Invalid YouTube URL' });
    }

    // Get video info
    const info = await ytdl.getInfo(url);

    // If POST request, return video information
    if (req.method === 'POST') {
      const formats = ytdl.filterFormats(info.formats, 'videoandaudio')
        .filter(format => format.hasVideo && format.hasAudio);

      const qualityOptions = formats
        .map(format => ({
          quality: format.qualityLabel || 'Auto',
          itag: format.itag,
          container: format.container,
          fileSize: format.contentLength ? formatFileSize(parseInt(format.contentLength)) : 'Unknown'
        }))
        .filter((format, index, self) => 
          index === self.findIndex(t => t.quality === format.quality)
        )
        .sort((a, b) => {
          const getQualityNumber = (q: string) => parseInt(q.replace('p', '')) || 0;
          return getQualityNumber(b.quality) - getQualityNumber(a.quality);
        });

      return res.status(200).json({
        title: info.videoDetails.title,
        thumbnail: info.videoDetails.thumbnails[info.videoDetails.thumbnails.length - 1].url,
        duration: formatDuration(parseInt(info.videoDetails.lengthSeconds)),
        formats: qualityOptions.map(format => ({
          ...format,
          url: `/api/convert?url=${encodeURIComponent(url)}&itag=${format.itag}`
        }))
      });
    }

    // If GET request, stream the video
    const itag = req.query.itag ? parseInt(req.query.itag as string) : null;
    
    const format = itag 
      ? info.formats.find(f => f.itag === itag)
      : ytdl.chooseFormat(info.formats, { quality: 'highest', filter: 'videoandaudio' });

    if (!format) {
      return res.status(400).json({ error: 'No suitable format found' });
    }

    const sanitizedTitle = info.videoDetails.title.replace(/[^\w\s-]/g, '');
    const contentType = format.mimeType?.split(';')[0] || 'video/mp4';
    
    res.setHeader('Content-Type', contentType);
    res.setHeader('Content-Disposition', `attachment; filename="${sanitizedTitle}.${format.container}"`);

    const stream = ytdl(url, { format });

    // Handle stream errors
    stream.on('error', (error) => {
      console.error('Stream error:', error);
      if (!res.headersSent) {
        res.status(500).json({ error: 'Failed to stream video' });
      }
    });

    // Pipe the video stream to response
    stream.pipe(res);

  } catch (error) {
    console.error('Error processing video:', error);
    if (!res.headersSent) {
      return res.status(500).json({ 
        error: error instanceof Error ? error.message : 'Failed to process video'
      });
    }
  }
} 