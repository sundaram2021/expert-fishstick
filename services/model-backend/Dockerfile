# syntax=docker/dockerfile:1
# ---------- build stage ----------
FROM node:20-bookworm-slim AS build
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY tsconfig.json ./
COPY src ./src
RUN npm run build
RUN npm prune --omit=dev

# ---------- runtime stage ----------
FROM node:20-bookworm-slim
ENV NODE_ENV=production
# Real-model mode (MODEL_MODE=real) downloads its ONNX weights here on first
# start; the default stub mode needs no model files at all.
ENV TRANSFORMERS_CACHE_DIR=/tmp/hf-cache
WORKDIR /app

COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/dist ./dist
COPY package.json ./

USER node
EXPOSE 8081
CMD ["node", "dist/index.js"]
