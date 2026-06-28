# Build stage — compiles the single static-ish binary.
FROM rust:1-slim-bookworm AS builder
WORKDIR /app

# rusqlite (bundled) needs a C compiler; reqwest default-tls needs OpenSSL.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential pkg-config libssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release

# Runtime stage — minimal image with just the binary and its TLS deps.
FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libssl3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/joule /usr/local/bin/joule

# Persist the request log outside the container if desired: -v joule:/data
ENV JOULE_DB=/data/joule.db
VOLUME ["/data"]
EXPOSE 8080

ENTRYPOINT ["joule"]
CMD ["serve", "--listen", "0.0.0.0:8080"]
