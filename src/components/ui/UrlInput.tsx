
import React, { useState } from "react";
import { Search, Loader2, Download } from "lucide-react";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";

interface UrlInputProps {
  onSubmit: (url: string) => void;
  className?: string;
}

const UrlInput = ({ onSubmit, className }: UrlInputProps) => {
  const [url, setUrl] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!url.trim()) return;
    
    setIsProcessing(true);
    
    // Simulate processing
    setTimeout(() => {
      onSubmit(url);
      setIsProcessing(false);
      
      // For demo purposes, show a toast or alert
      alert("This is a frontend demo. In a real application, this would process the URL.");
    }, 1500);
  };

  return (
    <motion.form 
      onSubmit={handleSubmit} 
      className={cn("w-full max-w-3xl mx-auto", className)}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="relative group">
        <motion.div 
          className="absolute inset-0 bg-gradient-to-r from-brand-500/30 to-brand-600/30 rounded-full blur-lg opacity-50 group-hover:opacity-70 transition-opacity"
          animate={{ 
            opacity: [0.5, 0.7, 0.5], 
          }}
          transition={{ 
            duration: 2,
            repeat: Infinity,
            ease: "easeInOut" 
          }}
        ></motion.div>
        <div className="relative bg-gray-900/90 border border-gray-700 rounded-full shadow-lg flex items-center overflow-hidden">
          <div className="flex-shrink-0 pl-4">
            <Search className="h-6 w-6 text-brand-400" />
          </div>
          <input
            type="text"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="Paste video URL here..."
            className="flex-grow py-4 px-4 bg-transparent focus:outline-none text-gray-100 placeholder:text-gray-500"
          />
          <motion.button
            type="submit"
            disabled={isProcessing || !url.trim()}
            className={cn(
              "flex-shrink-0 bg-brand-600 hover:bg-brand-700 text-black font-bold py-3 px-6 m-1 rounded-full transition-all neon-glow flex items-center justify-center",
              (isProcessing || !url.trim()) && "opacity-70 cursor-not-allowed"
            )}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            {isProcessing ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                <span>Processing</span>
              </>
            ) : (
              <>
                <Download className="mr-2 h-4 w-4" />
                <span>Convert</span>
              </>
            )}
          </motion.button>
        </div>
      </div>
      <p className="text-center mt-3 text-sm text-gray-400">
        Supports YouTube, Facebook, Instagram, Twitter, and TikTok
      </p>
    </motion.form>
  );
};

export default UrlInput;
