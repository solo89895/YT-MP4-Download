import React from 'react';

const LegalPage: React.FC = () => {
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
              <a href="/how-it-works" className="text-gray-300 hover:text-white text-sm font-medium transition-colors">How It Works</a>
              <a href="/legal" className="text-gray-300 hover:text-white text-sm font-medium transition-colors">Legal</a>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex-1">
        <div className="max-w-4xl mx-auto px-4 py-12 sm:px-6 lg:px-8">
          <div className="space-y-12">
            {/* Terms of Service Section */}
            <section>
              <h1 className="text-3xl font-bold text-white mb-8">Terms of Service</h1>
              <div className="prose prose-invert max-w-none">
                <h2 className="text-xl font-semibold text-[#99cc00] mb-4">1. Acceptance of Terms</h2>
                <p className="text-gray-300 mb-4">
                  By accessing and using FastDownloadLK, you accept and agree to be bound by the terms and provisions of this agreement.
                </p>

                <h2 className="text-xl font-semibold text-[#99cc00] mb-4">2. Use License</h2>
                <p className="text-gray-300 mb-4">
                  This service is provided for personal, non-commercial use only. You agree not to use this service for any illegal or unauthorized purposes.
                </p>

                <h2 className="text-xl font-semibold text-[#99cc00] mb-4">3. Disclaimer</h2>
                <p className="text-gray-300 mb-4">
                  FastDownloadLK is not responsible for any content downloaded through our service. Users are responsible for ensuring they have the right to download content.
                </p>

                <h2 className="text-xl font-semibold text-[#99cc00] mb-4">4. Service Modifications</h2>
                <p className="text-gray-300 mb-4">
                  We reserve the right to modify or discontinue the service at any time without notice.
                </p>
              </div>
            </section>

            {/* Privacy Policy Section */}
            <section>
              <h1 className="text-3xl font-bold text-white mb-8">Privacy Policy</h1>
              <div className="prose prose-invert max-w-none">
                <h2 className="text-xl font-semibold text-[#99cc00] mb-4">1. Information Collection</h2>
                <p className="text-gray-300 mb-4">
                  We collect minimal information necessary to provide our service. This includes video URLs and basic usage statistics.
                </p>

                <h2 className="text-xl font-semibold text-[#99cc00] mb-4">2. Use of Information</h2>
                <p className="text-gray-300 mb-4">
                  The information we collect is used solely to provide and improve our service. We do not sell or share your personal information.
                </p>

                <h2 className="text-xl font-semibold text-[#99cc00] mb-4">3. Cookies</h2>
                <p className="text-gray-300 mb-4">
                  We use cookies to enhance your experience. You can choose to disable cookies in your browser settings.
                </p>

                <h2 className="text-xl font-semibold text-[#99cc00] mb-4">4. Security</h2>
                <p className="text-gray-300 mb-4">
                  We implement reasonable security measures to protect your information. However, no internet transmission is completely secure.
                </p>

                <h2 className="text-xl font-semibold text-[#99cc00] mb-4">5. Changes to Privacy Policy</h2>
                <p className="text-gray-300 mb-4">
                  We may update this privacy policy from time to time. Continued use of the service after changes constitutes acceptance of the new policy.
                </p>
              </div>
            </section>

            {/* Contact Section */}
            <section className="bg-[#242424] rounded-xl p-6">
              <h2 className="text-xl font-semibold text-white mb-4">Contact Us</h2>
              <p className="text-gray-300 mb-4">
                If you have any questions about these Terms, please contact us at{' '}
                <a href="mailto:anyshope.lk@gmail.com" className="text-[#99cc00] hover:underline">
                  anyshope.lk@gmail.com
                </a>
              </p>
            </section>
          </div>
        </div>
      </main>
    </div>
  );
};

export default LegalPage; 