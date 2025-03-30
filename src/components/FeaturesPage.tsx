import React from 'react';
import Footer from './Footer';

const FeaturesPage: React.FC = () => {
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
              <a href="/features" className="text-[#99cc00] text-sm font-medium transition-colors">Features</a>
              <a href="/how-it-works" className="text-gray-300 hover:text-white text-sm font-medium transition-colors">How It Works</a>
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
              Powerful Features for Video Downloads
            </h1>
            <p className="text-xl text-gray-400">
              Everything you need for seamless video downloads
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Multiple Platform Support */}
            <div className="bg-[#242424] rounded-xl p-6 hover:bg-[#2a2a2a] transition-colors">
              <div className="w-12 h-12 bg-[#99cc00]/20 rounded-lg flex items-center justify-center mb-4">
                <svg className="w-6 h-6 text-[#99cc00]" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">Multiple Platform Support</h3>
              <p className="text-gray-400">
                Download videos from YouTube, Facebook, Instagram, Twitter, and more platforms with ease.
              </p>
            </div>

            {/* High Quality Downloads */}
            <div className="bg-[#242424] rounded-xl p-6 hover:bg-[#2a2a2a] transition-colors">
              <div className="w-12 h-12 bg-[#99cc00]/20 rounded-lg flex items-center justify-center mb-4">
                <svg className="w-6 h-6 text-[#99cc00]" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">High Quality Downloads</h3>
              <p className="text-gray-400">
                Choose from multiple quality options including 1080p, 720p, and more for the perfect balance of quality and size.
              </p>
            </div>

            {/* Fast Downloads */}
            <div className="bg-[#242424] rounded-xl p-6 hover:bg-[#2a2a2a] transition-colors">
              <div className="w-12 h-12 bg-[#99cc00]/20 rounded-lg flex items-center justify-center mb-4">
                <svg className="w-6 h-6 text-[#99cc00]" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">Fast Downloads</h3>
              <p className="text-gray-400">
                Experience lightning-fast downloads with our optimized servers and compression technology.
              </p>
            </div>

            {/* No Registration Required */}
            <div className="bg-[#242424] rounded-xl p-6 hover:bg-[#2a2a2a] transition-colors">
              <div className="w-12 h-12 bg-[#99cc00]/20 rounded-lg flex items-center justify-center mb-4">
                <svg className="w-6 h-6 text-[#99cc00]" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">No Registration Required</h3>
              <p className="text-gray-400">
                Start downloading immediately - no account creation or login required.
              </p>
            </div>
          </div>

          {/* Additional Features List */}
          <div className="mt-16">
            <h2 className="text-2xl font-bold text-white mb-8 text-center">More Great Features</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {[
                "Automatic format detection",
                "Progress tracking",
                "Resume interrupted downloads",
                "Browser extension support",
                "Mobile-friendly interface",
                "24/7 availability",
                "Secure downloads",
                "Regular updates"
              ].map((feature, index) => (
                <div key={index} className="flex items-center gap-2 text-gray-300">
                  <svg className="w-5 h-5 text-[#99cc00]" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                  {feature}
                </div>
              ))}
            </div>
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
};

export default FeaturesPage; 