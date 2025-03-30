import { motion } from "framer-motion";
import { FiShield, FiCheck } from "react-icons/fi";

const LegalPoint = ({ text }: { text: string }) => (
  <motion.div
    initial={{ opacity: 0, x: -20 }}
    whileInView={{ opacity: 1, x: 0 }}
    viewport={{ once: true }}
    className="flex items-start gap-3"
  >
    <FiCheck className="w-5 h-5 text-[#1DB954] mt-1 flex-shrink-0" />
    <p className="text-gray-400">{text}</p>
  </motion.div>
);

const Legal = () => {
  const legalPoints = [
    "We do not store or host any copyrighted content on our servers.",
    "This service is intended for personal use only with content you have the right to download.",
    "Users are responsible for complying with local copyright laws and regulations.",
    "We respect intellectual property rights and comply with DMCA guidelines.",
    "Some features might not be available in all regions due to local restrictions.",
  ];

  return (
    <section id="legal" className="py-24 px-4 bg-black">
      <div className="max-w-4xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-12"
        >
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-[#1DB954] bg-opacity-10 mb-6">
            <FiShield className="w-8 h-8 text-[#1DB954]" />
          </div>
          <h2 className="text-3xl font-bold text-white mb-4">
            Legal Information
          </h2>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            We are committed to providing a safe and legal video conversion service
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="bg-[#0A0A0A] border border-gray-800 rounded-2xl p-8 space-y-6"
        >
          {legalPoints.map((point, index) => (
            <LegalPoint key={index} text={point} />
          ))}
        </motion.div>

        <motion.p
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="text-sm text-gray-500 text-center mt-8"
        >
          By using our service, you agree to comply with our terms and conditions.
          For more information, please read our full Terms of Service and Privacy Policy.
        </motion.p>
      </div>
    </section>
  );
};

export default Legal; 