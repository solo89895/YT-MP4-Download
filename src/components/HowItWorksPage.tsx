import React from 'react';
import Footer from './Footer';

const HowItWorksPage: React.FC = () => {
  return (
    <div className="min-h-screen bg-[#1a1a1a] flex flex-col">
      {/* Navigation */}
      <nav className="bg-[#111] border-b border-[#333] sticky top-0 z-50">
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
              <a href="/how-it-works" className="text-[#99cc00] text-sm font-medium transition-colors">How It Works</a>
              <a href="/legal" className="text-gray-300 hover:text-white text-sm font-medium transition-colors">Legal</a>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex-1">
        <div className="max-w-4xl mx-auto px-4 py-12 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h1 className="text-4xl font-bold text-white mb-4">
              How to Download Videos
            </h1>
            <p className="text-xl text-gray-400">
              Follow these simple steps to download your favorite videos
            </p>
          </div>

          <div className="space-y-8">
            {/* Step 1 */}
            <div className="bg-[#242424] rounded-xl p-6 relative">
              <div className="absolute -left-3 -top-3 w-10 h-10 bg-[#99cc00] rounded-full flex items-center justify-center text-black font-bold">
                1
              </div>
              <h3 className="text-xl font-semibold text-white mb-4 pl-8">Copy the Video URL</h3>
              <p className="text-gray-400 mb-4">
                Go to the video you want to download on YouTube, Facebook, Instagram, or Twitter. Copy the URL from your browser's address bar.
              </p>
              <div className="bg-[#1a1a1a] rounded-lg p-4">
                <code className="text-[#99cc00]">https://www.youtube.com/watch?v=example</code>
              </div>
            </div>

            {/* Step 2 */}
            <div className="bg-[#242424] rounded-xl p-6 relative">
              <div className="absolute -left-3 -top-3 w-10 h-10 bg-[#99cc00] rounded-full flex items-center justify-center text-black font-bold">
                2
              </div>
              <h3 className="text-xl font-semibold text-white mb-4 pl-8">Paste the URL</h3>
              <p className="text-gray-400 mb-4">
                Return to FastDownloadLK and paste the video URL into the download box at the top of the page.
              </p>
              <div className="bg-[#1a1a1a] rounded-lg p-4 flex items-center gap-4">
                <div className="flex-1 bg-[#333] rounded h-10"></div>
                <div className="w-24 bg-[#99cc00] rounded h-10"></div>
              </div>
            </div>

            {/* Step 3 */}
            <div className="bg-[#242424] rounded-xl p-6 relative">
              <div className="absolute -left-3 -top-3 w-10 h-10 bg-[#99cc00] rounded-full flex items-center justify-center text-black font-bold">
                3
              </div>
              <h3 className="text-xl font-semibold text-white mb-4 pl-8">Choose Quality</h3>
              <p className="text-gray-400 mb-4">
                Select your preferred video quality from the available options. We offer various formats from 1080p to 144p.
              </p>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                {['1080p', '720p', '480p', '360p'].map((quality) => (
                  <div key={quality} className="bg-[#1a1a1a] rounded p-2 text-center text-gray-300">
                    {quality}
                  </div>
                ))}
              </div>
            </div>

            {/* Step 4 */}
            <div className="bg-[#242424] rounded-xl p-6 relative">
              <div className="absolute -left-3 -top-3 w-10 h-10 bg-[#99cc00] rounded-full flex items-center justify-center text-black font-bold">
                4
              </div>
              <h3 className="text-xl font-semibold text-white mb-4 pl-8">Download</h3>
              <p className="text-gray-400 mb-4">
                Click the download button and wait for the process to complete. The video will be saved to your default downloads folder.
              </p>
              <div className="flex items-center gap-2 text-[#99cc00]">
                <svg className="w-5 h-5" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clipRule="evenodd" />
                </svg>
                <span>Download Complete!</span>
              </div>
            </div>
          </div>

          {/* Tips Section */}
          <div className="mt-12 bg-[#242424] rounded-xl p-6">
            <h2 className="text-2xl font-bold text-white mb-6">Pro Tips</h2>
            <ul className="space-y-4">
              <li className="flex items-start gap-3">
                <svg className="w-6 h-6 text-[#99cc00] flex-shrink-0 mt-1" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
                <p className="text-gray-300">For the best quality, always choose the highest resolution available that your device can support.</p>
              </li>
              <li className="flex items-start gap-3">
                <svg className="w-6 h-6 text-[#99cc00] flex-shrink-0 mt-1" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
                <p className="text-gray-300">Make sure you have a stable internet connection for uninterrupted downloads.</p>
              </li>
              <li className="flex items-start gap-3">
                <svg className="w-6 h-6 text-[#99cc00] flex-shrink-0 mt-1" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
                <p className="text-gray-300">Check your available storage space before downloading large video files.</p>
              </li>
            </ul>
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
};

export default HowItWorksPage; 