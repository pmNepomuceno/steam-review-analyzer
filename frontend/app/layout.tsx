import type { Metadata, Viewport } from "next";
import { Figtree } from "next/font/google";
import Link from "next/link";
import type { ReactNode } from "react";

import "./globals.css";

// Steam's own face (Motiva Sans) is Valve's; Figtree is a close, freely licensed match.
const sans = Figtree({ subsets: ["latin"], variable: "--font-sans", display: "swap" });

export const metadata: Metadata = {
  title: "Steam Review Analyzer",
  description: "What Steam reviews say about performance, price, bugs, story and gameplay",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover", // lets the header paint under the notch; content pads back via env()
  // The header is dark in both color schemes, so the browser chrome matches it in both.
  themeColor: "#171d25",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={sans.variable}>
      <body>
        <header className="site-header">
          <div>
            <Link href="/">
              <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden fill="none">
                <rect x="3" y="12" width="4" height="9" rx="1" fill="#66c0f4" />
                <rect x="10" y="7" width="4" height="14" rx="1" fill="#dfe8ef" />
                <rect x="17" y="3" width="4" height="18" rx="1" fill="#d9693a" />
              </svg>
              Steam Review <span>Analyzer</span>
            </Link>
          </div>
        </header>
        <main>{children}</main>
      </body>
    </html>
  );
}
