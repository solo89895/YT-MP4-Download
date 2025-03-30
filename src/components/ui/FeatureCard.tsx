
import React from "react";
import { cn } from "@/lib/utils";

interface FeatureCardProps {
  icon: React.ReactNode;
  title: string;
  description: string;
  className?: string;
  style?: React.CSSProperties;
}

const FeatureCard = ({ icon, title, description, className, style }: FeatureCardProps) => {
  return (
    <div 
      className={cn(
        "bg-white border border-brand-100 rounded-2xl p-6 shadow-sm hover:shadow-md transition-all duration-300 hover:translate-y-[-4px]", 
        className
      )}
      style={style}
    >
      <div className="w-12 h-12 bg-brand-50 rounded-xl flex items-center justify-center text-brand-500 mb-4">
        {icon}
      </div>
      <h3 className="text-xl font-medium text-brand-900 mb-2">{title}</h3>
      <p className="text-brand-600 text-sm leading-relaxed">{description}</p>
    </div>
  );
};

export default FeatureCard;
