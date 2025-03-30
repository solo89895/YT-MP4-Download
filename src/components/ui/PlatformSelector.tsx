
import React from "react";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";

interface Platform {
  id: string;
  name: string;
  icon: React.ReactNode;
}

interface PlatformSelectorProps {
  platforms: Platform[];
  selectedPlatform: string;
  onSelect: (platformId: string) => void;
  className?: string;
  style?: React.CSSProperties;
}

const PlatformSelector = ({
  platforms,
  selectedPlatform,
  onSelect,
  className,
  style,
}: PlatformSelectorProps) => {
  return (
    <div 
      className={cn("flex flex-wrap justify-center gap-3", className)}
      style={style}
    >
      {platforms.map((platform, index) => (
        <motion.button
          key={platform.id}
          onClick={() => onSelect(platform.id)}
          className={cn(
            "flex items-center gap-2 px-4 py-2 rounded-full border transition-all duration-300",
            selectedPlatform === platform.id
              ? "border-brand-500 bg-brand-500/20 text-brand-400 neon-glow"
              : "border-gray-700 bg-gray-800/70 text-gray-300 hover:border-brand-500/50 hover:bg-gray-800"
          )}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: index * 0.1 }}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.98 }}
        >
          <span className="text-brand-400">{platform.icon}</span>
          <span className="font-medium text-sm">{platform.name}</span>
        </motion.button>
      ))}
    </div>
  );
};

export default PlatformSelector;
