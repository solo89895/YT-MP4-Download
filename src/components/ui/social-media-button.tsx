import { Button } from "./button"
import { Share2 } from "lucide-react"

interface SocialMediaButtonProps {
  url: string
  title: string
}

export function SocialMediaButton({ url, title }: SocialMediaButtonProps) {
  const shareOptions = [
    {
      name: 'Facebook',
      url: `https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(url)}`,
      icon: 'facebook'
    },
    {
      name: 'Twitter',
      url: `https://twitter.com/intent/tweet?url=${encodeURIComponent(url)}&text=${encodeURIComponent(title)}`,
      icon: 'twitter'
    },
    {
      name: 'WhatsApp',
      url: `https://wa.me/?text=${encodeURIComponent(title + ' ' + url)}`,
      icon: 'whatsapp'
    },
    {
      name: 'Telegram',
      url: `https://t.me/share/url?url=${encodeURIComponent(url)}&text=${encodeURIComponent(title)}`,
      icon: 'telegram'
    }
  ]

  return (
    <div className="relative group">
      <Button variant="outline" size="icon" className="rounded-full">
        <Share2 className="h-4 w-4" />
      </Button>
      <div className="absolute right-0 mt-2 w-48 bg-white rounded-md shadow-lg py-1 hidden group-hover:block z-50">
        {shareOptions.map((option) => (
          <a
            key={option.name}
            href={option.url}
            target="_blank"
            rel="noopener noreferrer"
            className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
          >
            {option.name}
          </a>
        ))}
      </div>
    </div>
  )
} 