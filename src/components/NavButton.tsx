import { motion } from "framer-motion";

interface NavButtonProps {
  href: string;
  children: React.ReactNode;
}

const NavButton = ({ href, children }: NavButtonProps) => {
  const handleClick = (e: React.MouseEvent) => {
    e.preventDefault();
    const element = document.querySelector(href);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <motion.a
      href={href}
      onClick={handleClick}
      className="relative px-4 py-2 cursor-pointer group text-white hover:text-[#1DB954] transition-colors"
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
    >
      <span className="relative z-10">{children}</span>
      
      {/* Animated underline */}
      <motion.div
        className="absolute bottom-0 left-0 h-[2px] bg-[#1DB954] rounded-full"
        initial={{ width: "0%" }}
        whileHover={{ width: "100%" }}
        transition={{ duration: 0.2 }}
      />

      {/* Hover glow effect */}
      <motion.div
        className="absolute inset-0 bg-[#1DB954] rounded-full opacity-0 blur-xl"
        whileHover={{ opacity: 0.1 }}
        transition={{ duration: 0.2 }}
      />
    </motion.a>
  );
};

export default NavButton; 