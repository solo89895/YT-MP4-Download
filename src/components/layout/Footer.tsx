
import React from "react";
import { Download } from "lucide-react";
import { motion } from "framer-motion";

const Footer = () => {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: { 
      y: 0, 
      opacity: 1,
      transition: { duration: 0.5 }
    }
  };

  return (
    <footer className="bg-black border-t border-gray-800">
      <div className="container py-12">
        <motion.div 
          className="grid grid-cols-1 md:grid-cols-4 gap-8"
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
        >
          <motion.div className="md:col-span-1" variants={itemVariants}>
            <a href="/" className="flex items-center gap-2 hover-scale inline-block">
              <span className="text-white font-semibold text-xl flex items-center">
                <Download className="w-5 h-5 text-brand-500 mr-1" />
                <span className="text-brand-400">Fast</span>
                <span>Download</span>
                <span className="text-brand-400 ml-1">LK</span>
              </span>
            </a>
            <p className="mt-4 text-sm text-gray-300">
              Convert and download videos from various platforms with ease. Fast, free, and simple.
            </p>
          </motion.div>
          
          <motion.div className="md:col-span-3 grid grid-cols-2 md:grid-cols-3 gap-8" variants={itemVariants}>
            <div>
              <h3 className="text-white font-medium mb-4">Platform</h3>
              <ul className="space-y-2">
                <li>
                  <a href="#features" className="text-sm text-gray-400 hover:text-brand-400 transition-colors">
                    Features
                  </a>
                </li>
                <li>
                  <a href="#how-it-works" className="text-sm text-gray-400 hover:text-brand-400 transition-colors">
                    How It Works
                  </a>
                </li>
                <li>
                  <a href="#" className="text-sm text-gray-400 hover:text-brand-400 transition-colors">
                    FAQs
                  </a>
                </li>
              </ul>
            </div>
            
            <div>
              <h3 className="text-white font-medium mb-4">Legal</h3>
              <ul className="space-y-2">
                <li>
                  <a href="#disclaimer" className="text-sm text-gray-400 hover:text-brand-400 transition-colors">
                    Disclaimer
                  </a>
                </li>
                <li>
                  <a href="#" className="text-sm text-gray-400 hover:text-brand-400 transition-colors">
                    Terms of Service
                  </a>
                </li>
                <li>
                  <a href="#" className="text-sm text-gray-400 hover:text-brand-400 transition-colors">
                    Privacy Policy
                  </a>
                </li>
              </ul>
            </div>
            
            <div>
              <h3 className="text-white font-medium mb-4">Supported</h3>
              <ul className="space-y-2">
                <li>
                  <a href="#" className="text-sm text-gray-400 hover:text-brand-400 transition-colors">
                    YouTube
                  </a>
                </li>
                <li>
                  <a href="#" className="text-sm text-gray-400 hover:text-brand-400 transition-colors">
                    Facebook
                  </a>
                </li>
                <li>
                  <a href="#" className="text-sm text-gray-400 hover:text-brand-400 transition-colors">
                    Instagram
                  </a>
                </li>
                <li>
                  <a href="#" className="text-sm text-gray-400 hover:text-brand-400 transition-colors">
                    Twitter
                  </a>
                </li>
                <li>
                  <a href="#" className="text-sm text-gray-400 hover:text-brand-400 transition-colors">
                    TikTok
                  </a>
                </li>
              </ul>
            </div>
          </motion.div>
        </motion.div>
        
        <div className="mt-12 pt-8 border-t border-gray-800 flex flex-col md:flex-row justify-between items-center">
          <p className="text-sm text-gray-400">
            © {new Date().getFullYear()} Fast Download LK. All rights reserved.
          </p>
          <div className="mt-4 md:mt-0">
            <p className="text-xs text-gray-500">
              Fast Download LK does not host any copyrighted content and respects intellectual property rights.
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
