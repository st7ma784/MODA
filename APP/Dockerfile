# MODA Mobile App — Flutter build/serve container
# Stage 1: Build the Flutter web app (used as a lightweight preview server)
# The primary mobile artifacts (APK/IPA) are produced by the CI runner directly.

ARG FLUTTER_VERSION=3.22.3

FROM ghcr.io/cirruslabs/flutter:${FLUTTER_VERSION} AS builder

WORKDIR /app

# Copy project files
COPY . .

# Fetch dependencies (no-op if pubspec.yaml absent — project not yet scaffolded)
RUN if [ -f pubspec.yaml ]; then \
      flutter pub get && \
      flutter build web --release; \
    else \
      mkdir -p build/web && \
      echo "<h1>MODA Mobile — coming soon</h1>" > build/web/index.html; \
    fi

# Stage 2: Serve the web build with nginx (useful for previewing UI in Rancher)
FROM nginx:1.27-alpine AS runtime

COPY --from=builder /app/build/web /usr/share/nginx/html

COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -qO- http://localhost/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
