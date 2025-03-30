import React from 'react';

const Footer: React.FC = () => {
  return (
    <div className="bg-[#111] text-xs text-gray-400 py-2 px-4 text-center border-t border-[#333]">
      By using our service you accept our{' '}
      <a href="/legal" className="text-[#99cc00] hover:underline">Terms of Service</a>
      {' '}and{' '}
      <a href="/legal" className="text-[#99cc00] hover:underline">Privacy Policy</a>
    </div>
  );
};

export default Footer; 