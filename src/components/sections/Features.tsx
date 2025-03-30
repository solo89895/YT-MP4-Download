import React from "react";
import FeatureCard from "@/components/ui/FeatureCard";
import { Download, Sliders, Clock, Zap, Shield, Music } from "lucide-react";
import { motion } from "framer-motion";

const features = [
  {
    icon: <Download className="w-6 h-6" />,
    title: "Multiple Platforms",
    description: "Download videos from YouTube, Facebook, Instagram, Pinterest, and TikTok with a single tool."
  },
  {
    icon: <Sliders className="w-6 h-6" />,
    title: "Format Selection",
    description: "Choose from various formats including MP4, MP3, WebM, and more to suit your needs."
  },
  {
    icon: <Clock className="w-6 h-6" />,
    title: "Instant Processing",
    description: "Get your converted videos instantly without waiting in long queues or delays."
  },
  {
    icon: <Zap className="w-6 h-6" />,
    title: "High Quality",
    description: "Download videos in high definition quality up to 4K resolution when available."
  },
  {
    icon: <Shield className="w-6 h-6" />,
    title: "Safe & Secure",
    description: "No registration required. We don't store your files or personal information."
  },
  {
    icon: <Music className="w-6 h-6" />,
    title: "Audio Extraction",
    description: "Extract audio from videos and download as MP3 files with a single click."
  }
];

const Features = () => {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  return (
    <section id="features" className="section bg-black py-20">
      <div className="container">
        <motion.div 
          className="text-center max-w-2xl mx-auto mb-16"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
            Designed for <span className="text-gradient">Simplicity</span>
          </h2>
          <p className="text-gray-300">
            Enjoy a seamless video conversion experience with our powerful, yet elegantly simple features
          </p>
        </motion.div>
        
        <motion.div 
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
        >
          {features.map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
            >
              <FeatureCard
                icon={feature.icon}
                title={feature.title}
                description={feature.description}
                className="h-full"
              />
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
};

export default Features;
