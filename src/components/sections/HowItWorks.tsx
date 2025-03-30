import { motion } from "framer-motion";
import { FiLink, FiSettings, FiDownload } from "react-icons/fi";

const steps = [
  {
    number: "01",
    icon: FiLink,
    title: "Paste URL",
    description: "Copy and paste the video URL from any supported platform"
  },
  {
    number: "02",
    icon: FiSettings,
    title: "Select Format",
    description: "Choose your preferred format and quality settings"
  },
  {
    number: "03",
    icon: FiDownload,
    title: "Download",
    description: "Click convert and download your video instantly"
  }
];

const HowItWorks = () => {
  return (
    <section id="how-it-works" className="py-24 px-4 bg-black">
      <div className="max-w-6xl mx-auto">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
            How It <span className="text-[#1DB954]">Works</span>
          </h2>
          <p className="text-xl text-gray-400">
            Convert your favorite videos in three simple steps
          </p>
        </motion.div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-12">
          {steps.map((step, index) => (
            <motion.div 
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.2 }}
              className="relative"
            >
              <div className="flex flex-col items-center text-center">
                <div className="relative mb-6">
                  <div className="w-20 h-20 bg-[#1DB954] bg-opacity-10 rounded-full flex items-center justify-center mb-6">
                    <step.icon className="w-8 h-8 text-[#1DB954]" />
                  </div>
                  <span className="absolute -top-2 -right-2 w-8 h-8 bg-[#1DB954] text-black text-sm font-bold rounded-full flex items-center justify-center">
                    {step.number}
                  </span>
                </div>
                
                <h3 className="text-xl font-bold text-white mb-3">{step.title}</h3>
                <p className="text-gray-400">{step.description}</p>
              </div>

              {index < steps.length - 1 && (
                <div className="hidden md:block absolute top-24 left-full w-full h-[2px] bg-gradient-to-r from-[#1DB954]/20 to-transparent transform -translate-x-1/2" />
              )}
            </motion.div>
          ))}
        </div>
        
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.6 }}
          className="mt-16 text-center"
        >
          <button 
            onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
            className="bg-[#1DB954] hover:bg-[#1ed760] text-white font-medium py-3 px-8 rounded-full transition-all duration-300 flex items-center gap-2 mx-auto"
          >
            <span>Try It Now</span>
            <FiDownload className="w-4 h-4" />
          </button>
        </motion.div>
      </div>
    </section>
  );
};

export default HowItWorks;
