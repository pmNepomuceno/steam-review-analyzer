import type { NextConfig } from "next";

// The browser calls /api/*, which Next proxies to FastAPI: same origin, so no CORS setup.
const apiUrl = process.env.API_URL ?? "http://localhost:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${apiUrl}/:path*` }];
  },
};

export default nextConfig;
