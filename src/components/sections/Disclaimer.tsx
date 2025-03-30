
import React from "react";
import { AlertTriangle, Info, Shield } from "lucide-react";
import { motion } from "framer-motion";

const Disclaimer = () => {
  return (
    <section id="disclaimer" className="section bg-black py-20">
      <div className="container">
        <motion.div 
          className="max-w-3xl mx-auto"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          <div className="glassmorphism rounded-2xl p-8 border border-gray-800">
            <div className="flex items-start gap-4 mb-6">
              <div className="flex-shrink-0 w-10 h-10 bg-brand-500/20 rounded-full flex items-center justify-center text-brand-400">
                <Shield className="w-5 h-5" />
              </div>
              <div>
                <h2 className="text-2xl font-bold text-white mb-2">Legal Disclaimer</h2>
                <p className="text-gray-300">
                  Please read the following important information regarding the use of our service
                </p>
              </div>
            </div>
            
            <div className="space-y-4 text-gray-300">
              <p>
                vidconvertly is designed to help users download and convert content for personal use only. We respect copyright laws and intellectual property rights.
              </p>
              
              <div className="bg-gray-800/50 border-l-4 border-brand-500 p-4 rounded-r-lg">
                <div className="flex items-start gap-3">
                  <Info className="w-5 h-5 text-brand-400 flex-shrink-0 mt-0.5" />
                  <p className="text-sm">
                    <strong className="text-brand-400">Important:</strong> Users of this service are responsible for ensuring they have the right to download and use any content. Downloading copyrighted material without permission may violate copyright laws.
                  </p>
                </div>
              </div>
              
              <p>
                Our service does not host any copyrighted content. We simply provide a technical service that helps users convert publicly available videos to different formats for personal use.
              </p>
              
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: 0.2 }}
              >
                <h3 className="text-lg font-medium text-white mt-6">Permitted Use</h3>
                <ul className="list-disc pl-5 space-y-2 mt-2">
                  <li>Download videos you have created yourself</li>
                  <li>Download videos that are in the public domain</li>
                  <li>Download videos when you have received permission from the copyright holder</li>
                  <li>Download videos allowed under fair use doctrines for educational or personal purposes</li>
                </ul>
              </motion.div>
              
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: 0.4 }}
              >
                <h3 className="text-lg font-medium text-white mt-6">Prohibited Use</h3>
                <ul className="list-disc pl-5 space-y-2 mt-2">
                  <li>Downloading copyrighted videos without permission</li>
                  <li>Redistributing downloaded content commercially</li>
                  <li>Using the service for any illegal purposes</li>
                  <li>Attempting to bypass any platform's terms of service</li>
                </ul>
              </motion.div>
              
              <p className="text-sm text-gray-400 mt-6">
                By using our service, you acknowledge that you have read and understood this disclaimer and agree to use the service responsibly and legally.
              </p>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default Disclaimer;
