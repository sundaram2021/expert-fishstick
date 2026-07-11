# syntax=docker/dockerfile:1
# ---------- build stage ----------
# bookworm-slim (glibc) rather than alpine: onnxruntime-node ships glibc binaries.
FROM node:20-bookworm-slim AS build
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY tsconfig.json ./
COPY src ./src
RUN npm run build

# Bake the embedding model into the image at build time: containers start with
# zero runtime dependency on huggingface.co (fast, deterministic cold starts).
ENV TRANSFORMERS_CACHE_DIR=/app/.hf-cache
RUN node dist/scripts/warmup-model.js

RUN npm prune --omit=dev

# ---------- runtime stage ----------
FROM node:20-bookworm-slim
ENV NODE_ENV=production
ENV TRANSFORMERS_CACHE_DIR=/app/.hf-cache
WORKDIR /app

COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/dist ./dist
COPY --from=build /app/.hf-cache ./.hf-cache
COPY package.json ./

USER node
EXPOSE 8080
CMD ["node", "dist/index.js"]
