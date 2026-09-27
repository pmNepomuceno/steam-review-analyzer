FROM node:22-slim

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

# Next bakes the /api rewrite target into the build, so API_URL must be set here, not at run
# time. No default: a missing value would build fine and send every request to localhost.
ARG API_URL
ENV API_URL=$API_URL NEXT_TELEMETRY_DISABLED=1

COPY . .
RUN test -n "$API_URL" || (echo "API_URL build arg is required (--build-arg API_URL=...)" >&2 && exit 1)
RUN npm run build

EXPOSE 3000
CMD ["npm", "start"]
