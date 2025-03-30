import { motion } from "framer-motion";
import { IconType } from "react-icons";
import { FiMonitor, FiList, FiZap, FiStar, FiShield, FiMusic } from "react-icons/fi";

interface FeatureCardProps {
  title: string;
  description: string;
  icon: IconType;
  delay: number;
}

const FeatureCard = ({ title, description, icon: Icon, delay }: FeatureCardProps) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay }}
      viewport={{ once: true }}
      whileHover={{ scale: 1.02 }}
      className="bg-black p-8 rounded-2xl border border-[#1DB954]/20 hover:border-[#1DB954]/40 transition-all duration-300"
    >
      <div className="flex flex-col items-center text-center space-y-4">
        <motion.div 
          className="p-4 bg-[#1DB954]/10 rounded-xl"
          whileHover={{ scale: 1.1 }}
          transition={{ type: "spring", stiffness: 300 }}
        >
          <Icon className="w-8 h-8 text-[#1DB954]" />
        </motion.div>
        <h3 className="text-xl font-bold text-white">
          {title}
        </h3>
        <p className="text-gray-400">
          {description}
        </p>
      </div>
    </motion.div>
  );
};

const Features = () => {
  const features = [
    {
      title: "Multiple Platforms",
      description: "Download videos from YouTube, Facebook, Instagram, Twitter, and more",
      icon: FiMonitor,
    },
    {
      title: "Format Selection",
      description: "Choose from MP4, MP3, and other popular formats for your downloads",
      icon: FiList,
    },
    {
      title: "Instant Processing",
      description: "Fast and efficient video processing with real-time conversion",
      icon: FiZap,
    },
    {
      title: "High Quality",
      description: "Download videos in HD quality up to 4K resolution",
      icon: FiStar,
    },
    {
      title: "Safe & Secure",
      description: "100% safe downloads with no registration required",
      icon: FiShield,
    },
    {
      title: "Audio Extraction",
      description: "Extract high-quality MP3 audio from any video",
      icon: FiMusic,
    },
  ];

  return (
    <section id="features" className="py-24 px-4 bg-black">
      <div className="max-w-7xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
            Our <span className="text-[#1DB954]">Features</span>
          </h2>
          <p className="text-xl text-gray-400">
            Everything you need for seamless video downloads
          </p>
        </motion.div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <FeatureCard
              key={feature.title}
              {...feature}
              delay={index * 0.1}
            />
          ))}
        </div>
      </div>
    </section>
  );
};

export default Features; 